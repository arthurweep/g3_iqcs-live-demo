import os
import io
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 配置matplotlib字体为英文，保证云端图表兼容性
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 参数中文映射
PARAM_MAP = {
    "product_id": "产品ID",
    "F_cut_act": "刀头实际压力 (N)",
    "v_cut_act": "切割实际速度 (mm/s)",
    "F_break_peak": "崩边力峰值 (N)",
    "v_wheel_act": "磨轮线速度 (m/s)",
    "F_wheel_act": "磨轮压紧力 (N)",
    "P_cool_act": "冷却水压力 (bar)",
    "t_glass_meas": "玻璃厚度 (mm)",
    "pressure_speed_ratio": "压速比",
    "stress_indicator": "应力指标",
    "energy_density": "能量密度"
}

# 全局缓存
model_cache = {}

# --- 辅助函数 ---

def fig_to_base64(fig):
    """将matplotlib图像转换为base64字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_features_df(params_dict, feature_names):
    """根据输入参数字典创建包含衍生特征的DataFrame"""
    df = pd.DataFrame([params_dict])
    
    # 安全地计算衍生特征，避免除零错误
    if 'F_cut_act' in df and 'v_cut_act' in df and df['v_cut_act'].iloc[0] != 0:
        df['pressure_speed_ratio'] = df['F_cut_act'] / df['v_cut_act']
    else:
        df['pressure_speed_ratio'] = np.nan

    if 'F_break_peak' in df and 't_glass_meas' in df and df['t_glass_meas'].iloc[0] != 0:
        df['stress_indicator'] = df['F_break_peak'] / df['t_glass_meas']
    else:
        df['stress_indicator'] = np.nan
        
    if 'F_cut_act' in df and 'v_cut_act' in df:
        df['energy_density'] = df['F_cut_act'] * df['v_cut_act']
    else:
        df['energy_density'] = np.nan

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 使用训练时计算的特征均值填充缺失值
    if 'feature_defaults' in model_cache:
        df = df.fillna(model_cache['feature_defaults'])
        
    return df[feature_names]

def get_adjustment_candidates(clf, current_params, feature_names, shap_values, target_threshold, golden_baseline):
    """
    第一步：识别所有可供调整的候选参数。
    """
    # 检查当前样本是否已经合格
    current_df = create_features_df(current_params, feature_names)
    initial_prob = clf.predict_proba(current_df.values)[0, 1]
    if initial_prob >= target_threshold:
        return {'candidates': [], 'message': "样本当前预测已合格，无需调整。"}

    # 优先级1：检查超出黄金基线的参数
    params_out_of_baseline = []
    current_values_array = current_df.values.flatten()
    for i, feature in enumerate(feature_names):
        # 只检查原始（非衍生）特征
        if not any(k in feature for k in ['ratio', 'indicator', 'density']):
            mean = golden_baseline['mean'].get(feature)
            std = golden_baseline['std'].get(feature)
            if mean is not None and std is not None:
                lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
                current_val = current_values_array[i]
                if not (lower_bound <= current_val <= upper_bound):
                    params_out_of_baseline.append({
                        'name': feature,
                        'display_name': PARAM_MAP.get(feature, feature),
                        'reason': f"当前值 {current_val:.2f}，超出黄金基线 [{lower_bound:.2f}, {upper_bound:.2f}]"
                    })
    
    if params_out_of_baseline:
        return {
            'candidates': params_out_of_baseline,
            'message': "检测到有参数超出黄金基线范围，请选择一个进行优先修正。"
        }

    # 优先级2：根据SHAP值推荐负面影响最大的参数
    shap_impact = sorted([(feature_names[i], shap_values[i]) for i in range(len(shap_values))], key=lambda x: x[1])
    
    shap_candidates = []
    # 获取最多3个负面影响最大的原始特征
    for feature_name, impact in shap_impact:
        if impact < 0 and not any(k in feature_name for k in ['ratio', 'indicator', 'density']):
            shap_candidates.append({
                'name': feature_name,
                'display_name': PARAM_MAP.get(feature_name, feature_name),
                'reason': f"SHAP分析显示其对结果有负面影响 (影响值: {impact:.3f})"
            })
        if len(shap_candidates) >= 3:
            break

    if shap_candidates:
        return {
            'candidates': shap_candidates,
            'message': "根据SHAP分析，以下参数对预测结果有较大负面影响，请选择一个进行优化。"
        }

    return {'candidates': [], 'message': "未能找到明确的可调整参数。"}


def calculate_single_adjustment(clf, current_params, feature_names, selected_param, golden_baseline):
    """
    第二步：根据用户选择的单个参数，计算具体的调整建议。
    """
    initial_df = create_features_df(current_params, feature_names)
    initial_prob = clf.predict_proba(initial_df.values)[0, 1]
    original_val = current_params[selected_param]

    # 检查所选参数是否是基线问题
    mean = golden_baseline['mean'].get(selected_param)
    std = golden_baseline['std'].get(selected_param)
    is_baseline_issue = False
    if mean is not None and std is not None:
        lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
        if not (lower_bound <= original_val <= upper_bound):
            is_baseline_issue = True
            
    # --- 方案A：修复基线问题 ---
    if is_baseline_issue:
        target_val = mean
        adjusted_params = current_params.copy()
        adjusted_params[selected_param] = target_val
        adjusted_df = create_features_df(adjusted_params, feature_names)
        new_prob = clf.predict_proba(adjusted_df.values)[0, 1]
        
        adjustments = {
            selected_param: {
                'original_value': float(original_val),
                'suggested_value': float(target_val),
                'change': float(target_val - original_val)
            }
        }
        message = f"将超出基线的参数 '{PARAM_MAP.get(selected_param)}' 调整至黄金工艺均值。调整后，预测合格概率从 {initial_prob:.2%} 变为 {new_prob:.2%}。"
        return adjustments, new_prob, message

    # --- 方案B：基于SHAP的迭代优化 ---
    best_new_val, max_prob = original_val, initial_prob
    search_range = np.linspace(-0.1, 0.1, 21) if original_val != 0 else np.linspace(-1, 1, 21) # 动态搜索范围
    
    for step_ratio in search_range:
        if step_ratio == 0: continue
        
        temp_params = current_params.copy()
        change = original_val * step_ratio if original_val != 0 else step_ratio
        temp_params[selected_param] += change

        temp_df = create_features_df(temp_params, feature_names)
        new_prob = clf.predict_proba(temp_df.values)[0, 1]

        if new_prob > max_prob:
            max_prob, best_new_val = new_prob, temp_params[selected_param]

    if max_prob > initial_prob + 1e-6: # 确认有实质性改善
        adjustments = {
            selected_param: {
                'original_value': float(original_val),
                'suggested_value': float(best_new_val),
                'change': float(best_new_val - original_val)
            }
        }
        message = f"根据您的选择和SHAP分析，优化参数 '{PARAM_MAP.get(selected_param)}'。调整后，预测合格概率可从 {initial_prob:.2%} 提升至 {max_prob:.2%}"
        return adjustments, max_prob, message

    # --- 备用方案 ---
    return {}, initial_prob, f"尝试调整参数 '{PARAM_MAP.get(selected_param)}'，但未能找到有效的改进方案。"

# --- Flask 路由 ---

@app.route('/', methods=['GET'])
def index():
    model_ready = bool(model_cache.get('is_ready', False))
    displayable_params = []
    if model_ready and 'param_map' in model_cache:
        # 仅显示可供用户输入的原始特征
        displayable_params = [
            (k, v) for k, v in model_cache['param_map'].items()
            if not any(x in k for x in ['product_id', 'ratio', 'indicator', 'density'])
        ]
    return render_template('index.html', model_ready=model_ready, cache=model_cache, displayable_params=displayable_params)


@app.route('/train', methods=['POST'])
def train():
    global model_cache
    model_cache.clear()
    if 'file' not in request.files:
        flash("未选择文件", "error")
        return redirect(url_for('index'))
    file = request.files['file']
    if not file or file.filename == '':
        flash("文件无效或文件名为空", "error")
        return redirect(url_for('index'))
    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
        
        # 特征工程
        df['pressure_speed_ratio'] = df['F_cut_act'] / df['v_cut_act']
        df['stress_indicator'] = df['F_break_peak'] / df['t_glass_meas']
        df['energy_density'] = df['F_cut_act'] * df['v_cut_act']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        features_to_use = [f for f in PARAM_MAP.keys() if f != "product_id"]
        X = df[features_to_use].copy()
        y = df["OK_NG"]
        
        # 数据预处理
        feature_defaults = X.mean().to_dict()
        X = X.fillna(feature_defaults)

        # 处理样本不平衡
        count_ok = y.value_counts().get(1, 0)
        count_ng = y.value_counts().get(0, 0)
        scale_pos_weight_value = count_ng / count_ok if count_ok > 0 else 1 # 修正权重计算
        
        # 模型训练
        clf = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                gamma=0.1, random_state=42, use_label_encoder=False,
                                scale_pos_weight=scale_pos_weight_value, eval_metric='logloss')
        clf.fit(X, y)
        
        # 优化分类阈值
        probs_ok = clf.predict_proba(X)[:, 1]
        best_f1_macro, best_thresh = 0.0, 0.5
        for t in np.arange(0.01, 1.0, 0.01):
            f1 = f1_score(y, (probs_ok >= t).astype(int), average='macro', zero_division=0)
            if f1 > best_f1_macro:
                best_f1_macro, best_thresh = f1, t
        
        # 评估最终模型
        y_pred_final = (probs_ok >= best_thresh).astype(int)
        metrics = {
            'accuracy': accuracy_score(y, y_pred_final),
            'recall_ng': recall_score(y, y_pred_final, pos_label=0, zero_division=0),
            'precision_ng': precision_score(y, y_pred_final, pos_label=0, zero_division=0),
            'f1_ng': f1_score(y, y_pred_final, pos_label=0, zero_division=0)
        }
        
        # 生成特征重要性图
        fig_importance = plt.figure()
        xgb.plot_importance(clf, max_num_features=10, ax=plt.gca(), title='Feature Importance')
        plt.tight_layout()
        
        # 缓存所有结果
        model_cache = {
            'model': clf,
            'features': features_to_use,
            'best_threshold': best_thresh,
            'feature_defaults': feature_defaults,
            'golden_baseline': {
                'mean': df[df['OK_NG']==1][features_to_use].mean().to_dict(),
                'std': df[df['OK_NG']==1][features_to_use].std().to_dict()
            },
            'knowledge_base': pd.DataFrame(df[df['OK_NG']==1]),
            'feature_plot': fig_to_base64(fig_importance),
            'metrics': metrics,
            'filename': file.filename,
            'is_ready': True,
            'param_map': PARAM_MAP
        }
        flash(f"文件 '{file.filename}' 上传成功，模型已完成训练！", "success")
    except Exception as e:
        app.logger.error(f"处理文件时出错: {e}", exc_info=True)
        flash(f"处理文件时出错: {e}", "error")

    return redirect(url_for('index'))


@app.route('/api/process_monitor', methods=['POST'])
def process_monitor():
    if not model_cache.get('is_ready'):
        return jsonify({'error': '请先上传数据并训练模型'}), 400
    data, warnings, gb = request.get_json(), [], model_cache['golden_baseline']
    for p, v in data.items():
        if p in gb['mean'] and gb['mean'][p] is not None:
            m, s = gb['mean'][p], gb['std'][p]
            if not (m - 3*s <= v <= m + 3*s):
                warnings.append(f"参数 '{PARAM_MAP.get(p,p)}' ({v:.2f}) 超出黄金基线范围 [{m-3*s:.2f}, {m+3*s:.2f}]")
    return jsonify({'status': 'warning' if warnings else 'ok', 'messages': warnings or ['所有参数均在黄金基线范围内']})


@app.route('/api/recommend_params', methods=['POST'])
def recommend_params():
    if not model_cache.get('is_ready'):
        return jsonify({'error': '请先上传数据并训练模型'}), 400
    product_id, kb, features = request.get_json().get('product_id'), model_cache['knowledge_base'], model_cache['features']
    cases = kb[kb['product_id'] == product_id] if product_id else pd.DataFrame()
    
    if cases.empty:
        params = kb[features].mean().to_dict()
        msg = f"知识库无产品'{product_id}'案例，返回通用建议。" if product_id else "未提供产品ID，返回通用建议。"
    else:
        params = cases[features].mean().to_dict()
        msg = f"为产品'{product_id}'生成了推荐参数。"
        
    return jsonify({'product_id': product_id, 'recommended_params': params, 'message': msg, 'param_map': PARAM_MAP})


@app.route('/api/predict', methods=['POST'])
def predict():
    if not model_cache.get('is_ready'):
        return jsonify({'error': '请先上传数据并训练模型'}), 400
    data = request.get_json()
    try:
        features = model_cache['features']
        input_vector = create_features_df(data, features) # 使用辅助函数

        baseline_warnings = []
        gb = model_cache['golden_baseline']
        for feature in features:
            if not any(k in feature for k in ['ratio', 'indicator', 'density']):
                val = input_vector[feature].iloc[0]
                if feature in gb['mean'] and gb['mean'][feature] is not None:
                    m, s = gb['mean'][feature], gb['std'][feature]
                    if not (m - 3*s <= val <= m + 3*s):
                        baseline_warnings.append(f"参数 '{PARAM_MAP.get(feature, feature)}' ({val:.2f}) 超出黄金工艺基线范围 [{m-3*s:.2f}, {m+3*s:.2f}]")
        
        model = model_cache['model']
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_vector)
        prob_ok = model.predict_proba(input_vector)[0, 1]
        model_says_ng = bool(prob_ok < model_cache['best_threshold'])

        final_status = "ok"
        verdict_reason = []
        if model_says_ng:
            final_status = "ng"
            verdict_reason.append(f"AI模型预测合格概率仅为 {prob_ok:.2%}，低于阈值 {model_cache['best_threshold']:.2%}。")
            verdict_reason.extend(baseline_warnings)
        elif baseline_warnings:
            final_status = "warning"
            verdict_reason.extend(baseline_warnings)
            verdict_reason.append("虽然AI预测合格，但存在过程异常，有潜在质量风险。")
        else:
            verdict_reason.append("所有参数均在基线内，且AI模型预测合格。")

        fig = plt.figure()
        shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_vector.iloc[0].values, feature_names=features), show=False, max_display=10)
        plt.tight_layout()
        
        return jsonify({
            'status': final_status,
            'prob': float(prob_ok),
            'threshold': float(model_cache['best_threshold']),
            'waterfall_plot': fig_to_base64(fig),
            'input_data': data,
            'shap_values': shap_values[0].tolist(),
            'verdict_reason': verdict_reason
        })
    except Exception as e:
        app.logger.error(f"预测失败: {e}", exc_info=True)
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/get_adjustment_candidates', methods=['POST'])
def api_get_adjustment_candidates():
    """新API：获取调整候选参数列表"""
    if not model_cache.get('is_ready'):
        return jsonify({'error': '请先上传数据并训练模型'}), 400
    
    data = request.get_json()
    try:
        input_data = data.get('input_data')
        shap_values = data.get('shap_values')
        if not all([input_data, shap_values is not None]):
            return jsonify({'error': '缺少 input_data 或 shap_values'}), 400

        result = get_adjustment_candidates(
            model_cache['model'], input_data, model_cache['features'], 
            shap_values, model_cache['best_threshold'], model_cache['golden_baseline']
        )
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"获取调整候选参数时出错: {e}", exc_info=True)
        return jsonify({'error': f'获取候选参数失败: {str(e)}'}), 500


@app.route('/api/adjust', methods=['POST'])
def adjust():
    """修改后的API：根据用户选择的参数进行调整"""
    if not model_cache.get('is_ready'):
        return jsonify({'error': '请先上传数据并训练模型'}), 400
    
    data = request.get_json()
    try:
        input_data = data.get('input_data')
        selected_parameter = data.get('selected_parameter') # 接收用户选择的参数
        if not all([input_data, selected_parameter]):
            return jsonify({'error': '缺少 input_data 或 selected_parameter'}), 400

        adjustments, final_prob, message = calculate_single_adjustment(
            model_cache['model'], input_data, model_cache['features'],
            selected_parameter, model_cache['golden_baseline']
        )
        
        return jsonify({
            'adjustments': adjustments,
            'final_prob_after_adjustment': float(final_prob),
            'message': message,
            'param_map': model_cache['param_map'] # 传递参数映射给前端
        })

    except Exception as e:
        app.logger.error(f"参数调整时出错: {e}", exc_info=True)
        return jsonify({'error': f'参数调整失败: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)


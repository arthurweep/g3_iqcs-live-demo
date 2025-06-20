import os
import io
import base64
import logging
import random
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

# 参数名映射保持不变
PARAM_MAP = {
    "product_id": "产品ID", "F_cut_act": "刀头实际压力 (N)", "v_cut_act": "切割实际速度 (mm/s)",
    "F_break_peak": "崩边力峰值 (N)", "v_wheel_act": "磨轮线速度 (m/s)", "F_wheel_act": "磨轮压紧力 (N)",
    "P_cool_act": "冷却水压力 (bar)", "t_glass_meas": "玻璃厚度 (mm)", "pressure_speed_ratio": "压速比",
    "stress_indicator": "应力指标", "energy_density": "能量密度"
}

# 全局模型缓存保持不变
model_cache = {}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# generate_tabular_adjustment_suggestions 函数保持不变，无需修改
def generate_tabular_adjustment_suggestions(current_params, feature_names, shap_values, golden_baseline):
    suggestions_data = []
    problematic_params = {}
    current_df = pd.DataFrame([current_params])
    current_df['pressure_speed_ratio'] = current_df['F_cut_act'] / current_df['v_cut_act']
    current_df['stress_indicator'] = current_df['F_break_peak'] / current_df['t_glass_meas']
    current_df['energy_density'] = current_df['F_cut_act'] * current_df['v_cut_act']
    current_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    current_df = current_df.fillna(model_cache.get('feature_defaults', {}))
    current_values_array = current_df[feature_names].values.flatten()
    negative_shap_total = 0
    for i, feature in enumerate(feature_names):
        if any(k in feature for k in ['ratio', 'indicator', 'density']): continue
        is_out_of_baseline = False
        is_shap_negative = shap_values[i] < -0.001
        mean = golden_baseline['mean'].get(feature)
        std = golden_baseline['std'].get(feature)
        current_val = current_values_array[i]
        if mean is not None and std is not None:
            upper_bound, lower_bound = mean + 3 * std, mean - 3 * std
            if not (lower_bound <= current_val <= upper_bound):
                is_out_of_baseline = True
        if is_out_of_baseline or is_shap_negative:
            problematic_params[feature] = {
                'shap_value': shap_values[i], 'is_out_of_baseline': is_out_of_baseline,
                'current_value': current_val, 'target_value': mean if mean is not None else current_val
            }
            if is_shap_negative:
                negative_shap_total += abs(shap_values[i])
    if not problematic_params: return [], "AI模型分析未找到明确的可优化参数。", ""
    sorted_problems = sorted(problematic_params.items(), key=lambda item: item[1]['shap_value'])
    baseline_warning_param = None
    for feature_name, details in sorted_problems:
        if details['is_out_of_baseline']:
            baseline_warning_param = PARAM_MAP.get(feature_name, feature_name)
            break
    for feature_name, details in sorted_problems:
        adjustment = details['target_value'] - details['current_value']
        contribution = (abs(details['shap_value']) / negative_shap_total) * 100 if negative_shap_total > 0 and details['shap_value'] < 0 else 0
        suggestions_data.append({
            "display_name": PARAM_MAP.get(feature_name, feature_name), "current_value": details['current_value'],
            "adjustment_amount": adjustment, "target_value": details['target_value'], "contribution": contribution
        })
    message = f"共生成 {len(suggestions_data)} 条优化建议，按影响力排序。"
    footer_note = f"优先修正超出基线的参数 '{baseline_warning_param}'。" if baseline_warning_param else "所有建议均基于提升合格率的AI模型分析。"
    return suggestions_data, message, footer_note

# index 函数保持不变，无需修改
@app.route('/', methods=['GET'])
def index():
    model_ready = bool(model_cache.get('is_ready', False))
    displayable_params = []
    random_initial_values = {}
    if model_ready and 'param_map' in model_cache:
        displayable_params = [(k, v) for k, v in model_cache['param_map'].items() if not any(x in k for x in ['product_id', 'ratio', 'indicator', 'density'])]
        gb = model_cache.get('golden_baseline', {})
        for key, _ in displayable_params:
            mean = gb.get('mean', {}).get(key, 1)
            std = gb.get('std', {}).get(key, mean * 0.1)
            random_val = mean + random.uniform(-1.5, 1.5) * std
            random_initial_values[key] = round(random_val, 2)
    return render_template('index.html', model_ready=model_ready, cache=model_cache, displayable_params=displayable_params, random_initial_values=random_initial_values)

# --- !!! 核心修改 1：增强/train接口，添加数据分析和模型解读 !!! ---
@app.route('/train', methods=['POST'])
def train():
    global model_cache, PARAM_MAP
    model_cache.clear()
    if 'file' not in request.files:
        flash("未选择文件", "error"); return redirect(url_for('index'))
    file = request.files['file']
    if not file or file.filename == '':
        flash("文件无效或文件名为空", "error"); return redirect(url_for('index'))
    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
        df['pressure_speed_ratio'] = df['F_cut_act'] / df['v_cut_act']
        df['stress_indicator'] = df['F_break_peak'] / df['t_glass_meas']
        df['energy_density'] = df['F_cut_act'] * df['v_cut_act']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_to_use = [f for f in PARAM_MAP.keys() if f != "product_id"]
        X = df[features_to_use].copy()
        y = df["OK_NG"]
        X = X.fillna(X.mean())

        # --- 新增：数据分析 ---
        total_samples = len(df)
        count_ok = int(y.value_counts().get(1, 0))
        count_ng = int(y.value_counts().get(0, 0))
        data_analysis = {
            'total_samples': total_samples, 'count_ok': count_ok, 'count_ng': count_ng,
            'ok_percentage': (count_ok / total_samples) * 100 if total_samples > 0 else 0,
            'ng_percentage': (count_ng / total_samples) * 100 if total_samples > 0 else 0
        }

        scale_pos_weight_value = count_ng / count_ok if count_ok > 0 else 1
        clf = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, 
            gamma=0.1, random_state=42, use_label_encoder=False, 
            scale_pos_weight=scale_pos_weight_value, eval_metric='logloss'
        )
        clf.fit(X, y)
        probs_ok = clf.predict_proba(X)[:, 1]
        best_f1_macro, best_thresh = 0.0, 0.5
        for t in np.arange(0.01, 1.0, 0.01):
            f1 = f1_score(y, (probs_ok >= t).astype(int), average='macro', zero_division=0)
            if f1 > best_f1_macro: best_f1_macro, best_thresh = f1, t
        fig_importance = plt.figure()
        xgb.plot_importance(clf, max_num_features=10, ax=plt.gca())
        plt.tight_layout()
        y_pred_final = (probs_ok >= best_thresh).astype(int)
        metrics = {
            'accuracy': accuracy_score(y, y_pred_final), 'recall_ng': recall_score(y, y_pred_final, pos_label=0),
            'precision_ng': precision_score(y, y_pred_final, pos_label=0), 'f1_ng': f1_score(y, y_pred_final, pos_label=0)
        }
        ok_df = df[df['OK_NG'] == 1]
        model_cache = {
            'model': clf, 'features': features_to_use, 'best_threshold': best_thresh,
            'feature_defaults': X.mean().to_dict(),
            'golden_baseline': {'mean': ok_df[features_to_use].mean().to_dict(), 'std': ok_df[features_to_use].std().to_dict()},
            'knowledge_base': pd.DataFrame(ok_df),
            'feature_plot': fig_to_base64(fig_importance), 'metrics': metrics,
            'data_analysis': data_analysis, # --- 新增：将数据分析结果存入缓存 ---
            'filename': file.filename, 'is_ready': True, 'param_map': PARAM_MAP
        }
        flash(f"文件 '{file.filename}' 上传成功，模型已完成训练！", "success")
    except Exception as e:
        flash(f"处理文件时出错: {e}", "error")
    return redirect(url_for('index'))

# 其他API路由 (/api/process_monitor, /api/recommend_params, /api/predict, /api/adjust) 保持不变
@app.route('/api/process_monitor', methods=['POST'])
def process_monitor():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data, warnings, gb = request.get_json(), [], model_cache['golden_baseline']
    for p, v in data.items():
        if p in gb['mean'] and p in gb['std']:
            m, s = gb['mean'][p], gb['std'][p]
            if not (m - 3*s <= v <= m + 3*s):
                warnings.append(f"参数 '{PARAM_MAP.get(p,p)}' ({v:.2f}) 超出黄金基线范围 [{m-3*s:.2f}, {m+3*s:.2f}]")
    return jsonify({'status': 'warning' if warnings else 'ok', 'messages': warnings or ['所有参数均在黄金基线范围内']})

@app.route('/api/recommend_params', methods=['POST'])
def recommend_params():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    product_id, kb, features = request.get_json().get('product_id'), model_cache['knowledge_base'], model_cache['features']
    cases = kb[kb['product_id'] == product_id]
    if cases.empty:
        params, msg = kb[features].mean().to_dict(), f"知识库无产品'{product_id}'案例，返回通用建议。"
    else:
        params, msg = cases[features].mean().to_dict(), f"为产品'{product_id}'生成了推荐参数。"
    return jsonify({'product_id': product_id, 'recommended_params': params, 'message': msg, 'param_map': PARAM_MAP})

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data = request.get_json()
    try:
        features = model_cache['features']
        input_df = pd.DataFrame([data])
        input_df['pressure_speed_ratio'] = input_df['F_cut_act'] / input_df['v_cut_act']
        input_df['stress_indicator'] = input_df['F_break_peak'] / input_df['t_glass_meas']
        input_df['energy_density'] = input_df['F_cut_act'] * input_df['v_cut_act']
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df = input_df.fillna(model_cache['feature_defaults'])
        input_vector = input_df[features]
        baseline_warnings = []
        gb = model_cache['golden_baseline']
        for feature in features:
            if not any(k in feature for k in ['ratio', 'indicator', 'density']) and feature in gb['mean']:
                val = input_vector[feature].iloc[0]
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
            verdict_reason.append(f"AI模型预测合格概率仅为 {prob_ok:.2%}，低于阈值。")
            verdict_reason.extend(baseline_warnings)
        elif baseline_warnings:
            final_status = "warning"
            verdict_reason.append("虽然AI预测合格，但存在过程异常，有潜在质量风险。")
            verdict_reason.extend(baseline_warnings)
        else:
            verdict_reason.append("所有参数均在基线内，且AI模型预测合格。")
        fig = plt.figure()
        shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_vector.iloc[0], feature_names=features), show=False, max_display=10)
        plt.tight_layout()
        return jsonify({
            'status': final_status, 'prob': float(prob_ok), 'threshold': float(model_cache['best_threshold']),
            'waterfall_plot': fig_to_base64(fig), 'input_data': data, 'shap_values': shap_values[0].tolist(),
            'verdict_reason': verdict_reason
        })
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/adjust', methods=['POST'])
def adjust():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data = request.get_json()
    try:
        features, gb = model_cache['features'], model_cache['golden_baseline']
        input_data, shap_values = data.get('input_data'), data.get('shap_values')
        suggestions, message, footer_note = generate_tabular_adjustment_suggestions(input_data, features, shap_values, gb)
        return jsonify({ 'suggestions_table': suggestions, 'message': message, 'footer_note': footer_note })
    except Exception as e:
        app.logger.error(f"参数调整时出错: {e}", exc_info=True)
        return jsonify({'error': f'参数调整失败: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

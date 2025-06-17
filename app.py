# app.py
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

# --- 初始化与配置 ---
app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量来缓存模型和相关产物
model_cache = {}

# --- 辅助函数 ---
def fig_to_base64(fig):
    """将matplotlib图像转换为base64字符串以便在HTML中显示"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- 参数调整建议的核心算法 ---
def calculate_adjustment_suggestions(clf, current_values_array, feature_names, target_threshold):
    """
    一个稳健的参数调整建议算法，寻找能最大化提升“合格概率”的单步调整。
    """
    app.logger.info(f"开始为NG样本计算调整建议... 目标合格概率: > {target_threshold:.4f}")
    
    original_values = np.array(current_values_array, dtype=float).flatten()
    current_values = original_values.copy()
    
    initial_prob = clf.predict_proba(current_values.reshape(1, -1))[0, 1]
    
    if initial_prob >= target_threshold:
        return {}, initial_prob, "样本当前预测已合格，无需调整。"

    adjustments_made = {}
    max_iterations = 100
    
    for i in range(max_iterations):
        current_prob = clf.predict_proba(current_values.reshape(1, -1))[0, 1]
        if current_prob >= target_threshold:
            app.logger.info(f"迭代 {i+1}: 已达到目标概率 {current_prob:.4f}")
            break

        best_move = {'feature_idx': -1, 'new_value': None, 'prob_gain': -1e9}

        for j in range(len(feature_names)):
            # 只调整基础的7个参数，不调整复合特征
            if any(k in feature_names[j] for k in ['ratio', 'indicator', 'density']):
                continue

            original_val = original_values[j]
            # 尝试小幅增加和减少
            for step_ratio in [0.01, -0.01, 0.05, -0.05]: 
                temp_values = current_values.copy()
                change = original_val * step_ratio if original_val != 0 else 0.1 * step_ratio
                temp_values[j] += change
                
                # 调整后需要重新计算复合特征
                temp_df = pd.DataFrame([temp_values], columns=feature_names)
                temp_df['pressure_speed_ratio'] = temp_df['F_cut_act'] / temp_df['v_cut_act']
                temp_df['stress_indicator'] = temp_df['F_break_peak'] / temp_df['t_glass_meas']
                temp_df['energy_density'] = temp_df['F_cut_act'] * temp_df['v_cut_act']
                temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                temp_df = temp_df.fillna(model_cache.get('feature_defaults', {}))
                
                new_prob = clf.predict_proba(temp_df.values)[0, 1]
                prob_gain = new_prob - current_prob
                
                if prob_gain > best_move['prob_gain']:
                    best_move = {'feature_idx': j, 'new_value': temp_values[j], 'prob_gain': prob_gain}
        
        if best_move['feature_idx'] != -1 and best_move['prob_gain'] > 1e-6:
            idx = best_move['feature_idx']
            feature_name = feature_names[idx]
            old_val = current_values[idx]
            new_val = best_move['new_value']
            
            current_values[idx] = new_val
            adjustments_made[feature_name] = {
                'original_value': float(original_values[idx]),
                'suggested_value': float(new_val),
                'change': float(new_val - original_values[idx]),
                'prob_contribution': float(best_move['prob_gain'])
            }
            app.logger.info(f"迭代 {i+1}: 建议调整 '{feature_name}'")
        else:
            app.logger.info(f"迭代 {i+1}: 未找到能显著提升概率的调整，终止。")
            break

    final_prob = clf.predict_proba(pd.DataFrame([current_values], columns=feature_names).values)[0, 1]
    
    if not adjustments_made:
        message = "未能找到有效的参数调整建议。该样本可能是一个“顽固”的缺陷。"
    elif final_prob >= target_threshold:
        message = f"调整建议已生成。调整后，样本预测合格概率可提升至 {final_prob:.2%}"
    else:
        message = f"已找到部分优化建议，但仍未达到目标阈值。调整后预测合格概率为 {final_prob:.2%}"

    return adjustments_made, final_prob, message

# --- 路由与功能 ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', model_ready=bool(model_cache.get('is_ready', False)), cache=model_cache)

@app.route('/train', methods=['POST'])
def train():
    global model_cache
    model_cache.clear() 
    app.logger.info("--- 开始处理文件上传和实时模型训练 ---")

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
        
        features_to_use = ["F_cut_act", "v_cut_act", "F_break_peak", "v_wheel_act", "F_wheel_act", "P_cool_act", "t_glass_meas", "pressure_speed_ratio", "stress_indicator", "energy_density"]
        X = df[features_to_use].copy()
        y = df["OK_NG"]
        X = X.fillna(X.mean())

        count_ok, count_ng = y.value_counts().get(1, 0), y.value_counts().get(0, 0)
        scale_pos_weight_value = count_ok / count_ng if count_ng > 0 else 1
        
        clf = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, gamma=0.1, random_state=42, use_label_encoder=False, scale_pos_weight=scale_pos_weight_value, eval_metric='logloss')
        clf.fit(X, y)
        
        probs_ok = clf.predict_proba(X)[:, 1]
        best_f1_macro, best_thresh = 0.0, 0.5
        for t in np.arange(0.01, 1.0, 0.01):
            f1 = f1_score(y, (probs_ok >= t).astype(int), average='macro', zero_division=0)
            if f1 > best_f1_macro: best_f1_macro, best_thresh = f1, t
        
        fig_importance = plt.figure(); xgb.plot_importance(clf, max_num_features=10, ax=plt.gca()); plt.tight_layout()
        y_pred_final = (probs_ok >= best_thresh).astype(int)
        metrics = {'accuracy': accuracy_score(y, y_pred_final), 'recall_ng': recall_score(y, y_pred_final, pos_label=0), 'precision_ng': precision_score(y, y_pred_final, pos_label=0), 'f1_ng': f1_score(y, y_pred_final, pos_label=0)}

        model_cache = {
            'model': clf, 'features': features_to_use, 'best_threshold': best_thresh,
            'feature_defaults': X.mean().to_dict(),
            'golden_baseline': {'mean': df[df['OK_NG']==1][features_to_use].mean().to_dict(), 'std': df[df['OK_NG']==1][features_to_use].std().to_dict()},
            'knowledge_base': pd.DataFrame(df[df['OK_NG']==1].to_dict('records')),
            'feature_plot': fig_to_base64(fig_importance), 'metrics': metrics,
            'filename': file.filename, 'is_ready': True
        }
        flash(f"文件 '{file.filename}' 上传成功，模型已完成训练！", "success")
    except Exception as e:
        flash(f"处理文件时出错: {e}", "error")

    return redirect(url_for('index'))

# --- API 接口 ---
@app.route('/api/process_monitor', methods=['POST'])
def process_monitor():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data, warnings, gb = request.get_json(), [], model_cache['golden_baseline']
    for p, v in data.items():
        if p in gb['mean']:
            m, s = gb['mean'][p], gb['std'][p]
            if not (m - 3*s <= v <= m + 3*s): warnings.append(f"参数'{p}'({v:.2f})超出范围[{m-3*s:.2f}, {m+3*s:.2f}]")
    return jsonify({'status': 'warning' if warnings else 'ok', 'messages': warnings or ['所有参数均在黄金基线范围内']})

@app.route('/api/recommend_params', methods=['POST'])
def recommend_params():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    product_id, kb = request.get_json().get('product_id'), model_cache['knowledge_base']
    cases = kb[kb['product_id'] == product_id]
    if cases.empty:
        params, msg = kb[model_cache['features']].mean().to_dict(), f"知识库无产品'{product_id}'案例，返回通用建议。"
    else:
        params, msg = cases[model_cache['features']].mean().to_dict(), f"为产品'{product_id}'生成了推荐参数。"
    return jsonify({'product_id': product_id, 'recommended_params': params, 'message': msg})

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data = request.get_json()
    try:
        input_df = pd.DataFrame([data])
        input_df['pressure_speed_ratio'] = input_df['F_cut_act'] / input_df['v_cut_act']
        input_df['stress_indicator'] = input_df['F_break_peak'] / input_df['t_glass_meas']
        input_df['energy_density'] = input_df['F_cut_act'] * input_df['v_cut_act']
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df = input_df.fillna(model_cache['feature_defaults'])
        input_vector = input_df[model_cache['features']]
        
        model = model_cache['model']; explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_vector)
        prob_ok = model.predict_proba(input_vector)[0, 1]
        is_ng = bool(prob_ok < model_cache['best_threshold'])
        
        fig = plt.figure(); shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_vector.iloc[0], feature_names=model_cache['features']), show=False, max_display=10);
        return jsonify({'prob': float(prob_ok), 'is_ng': is_ng, 'threshold': float(model_cache['best_threshold']), 'waterfall_plot': fig_to_base64(fig), 'input_data': data})
    except Exception as e: return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/adjust', methods=['POST'])
def adjust():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data = request.get_json()
    try:
        clf, features, threshold = model_cache['model'], model_cache['features'], model_cache['best_threshold']
        input_data = data.get('input_data')
        
        current_df = pd.DataFrame([input_data])
        current_df['pressure_speed_ratio'] = current_df['F_cut_act'] / current_df['v_cut_act']
        current_df['stress_indicator'] = current_df['F_break_peak'] / current_df['t_glass_meas']
        current_df['energy_density'] = current_df['F_cut_act'] * current_df['v_cut_act']
        current_df.replace([np.inf, -np.inf], np.nan, inplace=True); current_df = current_df.fillna(model_cache['feature_defaults'])
        current_values_array = current_df[features].values

        adjustments, final_prob, message = calculate_adjustment_suggestions(clf, current_values_array, features, threshold)
        return jsonify({'adjustments': adjustments, 'final_prob_after_adjustment': float(final_prob), 'message': message})
    except Exception as e: return jsonify({'error': f'参数调整失败: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

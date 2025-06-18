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

# 配置matplotlib字体为英文，保证云端图表兼容性
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# 参数中英文映射表
PARAM_MAP = {
    "product_id": "产品ID", "F_cut_act": "刀头实际压力 (N)", "v_cut_act": "切割实际速度 (mm/s)",
    "F_break_peak": "崩边力峰值 (N)", "v_wheel_act": "磨轮线速度 (m/s)", "F_wheel_act": "磨轮压紧力 (N)",
    "P_cool_act": "冷却水压力 (bar)", "t_glass_meas": "玻璃厚度 (mm)", "pressure_speed_ratio": "压速比",
    "stress_indicator": "应力指标", "energy_density": "能量密度"
}

model_cache = {}

# --- 辅助函数 ---
def fig_to_base64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- 参数调整建议的核心算法 ---
def calculate_adjustment_suggestions_v2(clf, current_params, feature_names, shap_values, target_threshold, golden_baseline):
    # (此函数逻辑保持不变)
    app.logger.info(f"开始为NG样本计算调整建议 (V2)... 目标合格概率: > {target_threshold:.4f}")
    current_df = pd.DataFrame([current_params])
    current_df['pressure_speed_ratio'] = current_df['F_cut_act'] / current_df['v_cut_act']
    current_df['stress_indicator'] = current_df['F_break_peak'] / current_df['t_glass_meas']
    current_df['energy_density'] = current_df['F_cut_act'] * current_df['v_cut_act']
    current_df.replace([np.inf, -np.inf], np.nan, inplace=True); current_df = current_df.fillna(model_cache.get('feature_defaults', {}))
    current_values_array = current_df[feature_names].values.flatten()
    initial_prob = clf.predict_proba(current_values_array.reshape(1, -1))[0, 1]

    if initial_prob >= target_threshold: return {}, initial_prob, "样本当前预测已合格，无需调整。", "N/A"

    params_out_of_baseline = []
    for i, feature in enumerate(feature_names):
        if not any(k in feature for k in ['ratio', 'indicator', 'density']):
            mean, std = golden_baseline['mean'].get(feature), golden_baseline['std'].get(feature)
            if mean is not None and std is not None:
                upper, lower = mean + 3 * std, mean - 3 * std
                if not (lower <= current_values_array[i] <= upper):
                    params_out_of_baseline.append({'idx': i, 'name': feature, 'current': current_values_array[i], 'target': mean, 'dist': abs(current_values_array[i] - mean)})
    
    if params_out_of_baseline:
        param_to_fix = max(params_out_of_baseline, key=lambda x: x['dist'])
        idx = param_to_fix['idx']; adjusted_values = current_values_array.copy(); adjusted_values[idx] = param_to_fix['target']
        temp_df = pd.DataFrame([adjusted_values], columns=feature_names)
        temp_df['pressure_speed_ratio'] = temp_df['F_cut_act'] / temp_df['v_cut_act']; temp_df['stress_indicator'] = temp_df['F_break_peak'] / temp_df['t_glass_meas']; temp_df['energy_density'] = temp_df['F_cut_act'] * temp_df['v_cut_act']
        final_prob = clf.predict_proba(temp_df.fillna(model_cache.get('feature_defaults', {})).values)[0, 1]
        adjustments = {param_to_fix['name']: {'original_value': float(param_to_fix['current']), 'suggested_value': float(param_to_fix['target']), 'change': float(param_to_fix['target'] - param_to_fix['current']), 'prob_contribution': float(final_prob - initial_prob)}}
        optimization_thought = f"优化思路：检测到参数“{PARAM_MAP.get(param_to_fix['name'])}”已严重偏离其正常工艺范围（黄金基线）。系统采取**常识优先策略**，首先建议将其修正回基线均值，这是恢复稳定生产最直接有效的方法。"
        message = f"调整后，预测合格概率可提升至 {final_prob:.2%}"; return adjustments, final_prob, message, optimization_thought

    shap_impact = sorted([(feature_names[i], shap_values[i]) for i in range(len(shap_values))], key=lambda x: x[1])
    for feature_name, impact in shap_impact:
        if impact < 0 and not any(k in feature_name for k in ['ratio', 'indicator', 'density']):
            idx, original_val = feature_names.index(feature_name), current_values_array[idx]
            best_new_val, max_prob = original_val, initial_prob
            for step_ratio in np.linspace(-0.1, 0.1, 21):
                if step_ratio == 0: continue
                temp_values = current_values_array.copy(); temp_values[idx] += original_val * step_ratio if original_val != 0 else 0.1 * step_ratio
                temp_df = pd.DataFrame([temp_values], columns=feature_names)
                new_prob = clf.predict_proba(temp_df.fillna(model_cache.get('feature_defaults', {})).values)[0, 1]
                if new_prob > max_prob: max_prob, best_new_val = new_prob, temp_values[idx]
            if max_prob > initial_prob + 1e-6:
                adjustments = {feature_name: {'original_value': float(original_val), 'suggested_value': float(best_new_val), 'change': float(best_new_val - original_val), 'prob_contribution': float(max_prob - initial_prob)}}
                optimization_thought = f"优化思路：所有参数均在正常范围内。系统启动**归因分析策略**，发现参数“{PARAM_MAP.get(feature_name)}”是对本次不合格结果负面影响最大的因素（据SHAP分析）。因此，系统优先对该参数进行微调，以寻求最大概率提升。"
                message = f"调整后，预测合格概率可提升至 {max_prob:.2%}"; return adjustments, max_prob, message, optimization_thought
    return {}, initial_prob, "未能找到有效的参数调整建议。", "系统在当前状态下未找到明确的单步优化路径。"

# --- 路由与功能 ---
@app.route('/', methods=['GET'])
def index():
    model_ready = bool(model_cache.get('is_ready', False))
    displayable_params = []
    if model_ready:
        # 【BUG修复】这里的 `x` 变量名被修正为 `substring`
        displayable_params = [
            (k, v) for k, v in model_cache['param_map'].items() 
            if not any(substring in k for substring in ['product_id', 'ratio', 'indicator', 'density'])
        ]
    return render_template('index.html', model_ready=model_ready, cache=model_cache, displayable_params=displayable_params)

@app.route('/train', methods=['POST'])
def train():
    global model_cache, PARAM_MAP
    model_cache.clear() 
    if 'file' not in request.files: flash("未选择文件", "error"); return redirect(url_for('index'))
    file = request.files['file']
    if not file or file.filename == '': flash("文件无效或文件名为空", "error"); return redirect(url_for('index'))
    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
        df['pressure_speed_ratio'] = df['F_cut_act'] / df['v_cut_act']
        df['stress_indicator'] = df['F_break_peak'] / df['t_glass_meas']
        df['energy_density'] = df['F_cut_act'] * df['v_cut_act']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_to_use = [f for f in PARAM_MAP.keys() if f != "product_id"]
        X = df[features_to_use].copy(); y = df["OK_NG"]; X = X.fillna(X.mean())
        count_ok, count_ng = y.value_counts().get(1, 0), y.value_counts().get(0, 0)
        scale_pos_weight_value = count_ok / count_ng if count_ng > 0 else 1
        clf = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, gamma=0.1, random_state=42, use_label_encoder=False, scale_pos_weight=scale_pos_weight_value, eval_metric='logloss')
        clf.fit(X, y)
        probs_ok = clf.predict_proba(X)[:, 1]
        best_f1_macro, best_thresh = 0.0, 0.5
        for t in np.arange(0.01, 1.0, 0.01):
            f1 = f1_score(y, (probs_ok >= t).astype(int), average='macro', zero_division=0)
            if f1 > best_f1_macro: best_f1_macro, best_thresh = f1, t
        clf.get_booster().feature_names = [PARAM_MAP.get(f, f) for f in features_to_use]
        fig_importance = plt.figure(); xgb.plot_importance(clf, max_num_features=10, ax=plt.gca()); plt.tight_layout()
        clf.get_booster().feature_names = None
        y_pred_final = (probs_ok >= best_thresh).astype(int)
        metrics = {'accuracy': accuracy_score(y, y_pred_final), 'recall_ng': recall_score(y, y_pred_final, pos_label=0), 'precision_ng': precision_score(y, y_pred_final, pos_label=0), 'f1_ng': f1_score(y, y_pred_final, pos_label=0)}
        model_cache = {'model': clf, 'features': features_to_use, 'best_threshold': best_thresh, 'feature_defaults': X.mean().to_dict(), 'golden_baseline': {'mean': df[df['OK_NG']==1][features_to_use].mean().to_dict(), 'std': df[df['OK_NG']==1][features_to_use].std().to_dict()}, 'knowledge_base': pd.DataFrame(df[df['OK_NG']==1]), 'feature_plot': fig_to_base64(fig_importance), 'metrics': metrics, 'filename': file.filename, 'is_ready': True, 'param_map': PARAM_MAP}
        flash(f"文件 '{file.filename}' 上传成功，模型已完成训练！", "success")
    except Exception as e: flash(f"处理文件时出错: {e}", "error")
    return redirect(url_for('index'))

@app.route('/api/process_monitor', methods=['POST'])
def process_monitor():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data, warnings, gb = request.get_json(), [], model_cache['golden_baseline']
    for p, v in data.items():
        if p in gb['mean']:
            m, s = gb['mean'][p], gb['std'][p]
            if not (m - 3*s <= v <= m + 3*s): warnings.append(f"参数 '{PARAM_MAP.get(p,p)}' ({v:.2f}) 超出黄金基线范围 [{m-3*s:.2f}, {m+3*s:.2f}]")
    return jsonify({'status': 'warning' if warnings else 'ok', 'messages': warnings or ['所有参数均在黄金基线范围内']})

@app.route('/api/recommend_params', methods=['POST'])
def recommend_params():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    product_id, kb, features = request.get_json().get('product_id'), model_cache['knowledge_base'], model_cache['features']
    cases = kb[kb['product_id'] == product_id]
    if cases.empty: params, msg = kb[features].mean().to_dict(), f"知识库无产品'{product_id}'案例，返回通用建议。"
    else: params, msg = cases[features].mean().to_dict(), f"为产品'{product_id}'生成了推荐参数。"
    return jsonify({'product_id': product_id, 'recommended_params': params, 'message': msg, 'param_map': PARAM_MAP})

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data = request.get_json()
    try:
        features = model_cache['features']
        input_df = pd.DataFrame([data]); input_df['pressure_speed_ratio'] = input_df['F_cut_act'] / input_df['v_cut_act']; input_df['stress_indicator'] = input_df['F_break_peak'] / input_df['t_glass_meas']; input_df['energy_density'] = input_df['F_cut_act'] * input_df['v_cut_act']
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True); input_df = input_df.fillna(model_cache['feature_defaults']); input_vector = input_df[features]
        baseline_warnings = []
        gb = model_cache['golden_baseline']
        for feature in features:
            if not any(k in feature for k in ['ratio', 'indicator', 'density']):
                val = input_vector[feature].iloc[0]
                if feature in gb['mean']:
                    m, s = gb['mean'][feature], gb['std'][feature]
                    if not (m - 3*s <= val <= m + 3*s): baseline_warnings.append(f"参数 '{PARAM_MAP.get(feature, feature)}' ({val:.2f}) 超出黄金工艺基线范围 [{m-3*s:.2f}, {m+3*s:.2f}]")
        model = model_cache['model']; explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_vector)
        prob_ok = model.predict_proba(input_vector)[0, 1]
        model_says_ng = bool(prob_ok < model_cache['best_threshold'])
        final_status = "ok"; verdict_reason = []
        if model_says_ng: final_status, verdict_reason = "ng", [f"AI模型预测合格概率仅为 {prob_ok:.2%}，低于阈值。"] + baseline_warnings
        elif baseline_warnings: final_status, verdict_reason = "warning", baseline_warnings + ["虽然AI预测合格，但存在过程异常，有潜在质量风险。"]
        else: verdict_reason.append("所有参数均在基线内，且AI模型预测合格。")
        display_features = [PARAM_MAP.get(f, f) for f in features]
        fig = plt.figure(); shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_vector.iloc[0], feature_names=display_features), show=False, max_display=10)
        return jsonify({'status': final_status, 'prob': float(prob_ok), 'threshold': float(model_cache['best_threshold']), 'waterfall_plot': fig_to_base64(fig), 'input_data': data, 'shap_values': shap_values[0].tolist(), 'verdict_reason': verdict_reason})
    except Exception as e: return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/adjust', methods=['POST'])
def adjust():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data = request.get_json()
    try:
        clf, features, threshold, gb = model_cache['model'], model_cache['features'], model_cache['best_threshold'], model_cache['golden_baseline']
        input_data, shap_values = data.get('input_data'), data.get('shap_values')
        adjustments, final_prob, message, thought = calculate_adjustment_suggestions_v2(clf, input_data, features, shap_values, threshold, gb)
        return jsonify({'adjustments': adjustments, 'final_prob_after_adjustment': float(final_prob), 'message': message, 'optimization_thought': thought, 'param_map': PARAM_MAP})
    except Exception as e:
        app.logger.error(f"参数调整时出错: {e}", exc_info=True)
        return jsonify({'error': f'参数调整失败: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

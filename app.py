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
    app.logger.info(f"开始为NG样本计算调整建议 (V2)... 目标合格概率: > {target_threshold:.4f}")
    
    current_df = pd.DataFrame([current_params])
    current_df['pressure_speed_ratio'] = current_df['F_cut_act'] / current_df['v_cut_act']
    current_df['stress_indicator'] = current_df['F_break_peak'] / current_df['t_glass_meas']
    current_df['energy_density'] = current_df['F_cut_act'] * current_df['v_cut_act']
    current_df.replace([np.inf, -np.inf], np.nan, inplace=True); current_df = current_df.fillna(model_cache.get('feature_defaults', {}))
    current_values_array = current_df[feature_names].values.flatten()
    initial_prob = clf.predict_proba(current_values_array.reshape(1, -1))[0, 1]

    if initial_prob >= target_threshold: return {}, initial_prob, "样本当前预测已合格，无需调整。"

    # 策略1: 优先修正超出黄金基线的参数
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
        idx = param_to_fix['idx']
        adjusted_values = current_values_array.copy()
        adjusted_values[idx] = param_to_fix['target']
        
        # 重新计算复合特征并预测
        temp_df = pd.DataFrame([adjusted_values], columns=feature_names)
        temp_df['pressure_speed_ratio'] = temp_df['F_cut_act'] / temp_df['v_cut_act']
        temp_df['stress_indicator'] = temp_df['F_break_peak'] / temp_df['t_glass_meas']
        temp_df['energy_density'] = temp_df['F_cut_act'] * temp_df['v_cut_act']
        final_prob = clf.predict_proba(temp_df.fillna(model_cache.get('feature_defaults', {})).values)[0, 1]
        
        adjustments = {param_to_fix['name']: {'original_value': float(param_to_fix['current']), 'suggested_value': float(param_to_fix['target']), 'change': float(param_to_fix['target'] - param_to_fix['current']), 'prob_contribution': float(final_prob - initial_prob)}}
        optimization_thought = f"优化思路：检测到参数“{PARAM_MAP.get(param_to_fix['name'])}”已严重偏离其正常工艺范围（黄金基线）。系统采取**常识优先策略**，首先建议将其修正回基线均值，这是恢复稳定生产最直接有效的方法。"
        message = f"调整后，预测合格概率可提升至 {final_prob:.2%}"
        return adjustments, final_prob, message, optimization_thought

    # 策略2: 根据SHAP值进行调整
    shap_impact = sorted([(feature_names[i], shap_values[i]) for i in range(len(shap_values))], key=lambda x: x[1])
    for feature_name, impact in shap_impact:
        if impact < 0 and not any(k in feature_name for k in ['ratio', 'indicator', 'density']):
            idx, original_val = feature_names.index(feature_name), current_values_array[idx]
            best_new_val, max_prob = original_val, initial_prob
            for step_ratio in np.linspace(-0.1, 0.1, 21):
                if step_ratio == 0: continue
                temp_values = current_values_array.copy()
                temp_values[idx] += original_val * step_ratio if original_val != 0 else 0.1 * step_ratio
                temp_df = pd.DataFrame([temp_values], columns=feature_names)
                # ... (重新计算复合特征和预测) ...
                new_prob = clf.predict_proba(temp_df.fillna(model_cache.get('feature_defaults', {})).values)[0, 1]
                if new_prob > max_prob: max_prob, best_new_val = new_prob, temp_values[idx]

            if max_prob > initial_prob + 1e-6:
                adjustments = {feature_name: {'original_value': float(original_val), 'suggested_value': float(best_new_val), 'change': float(best_new_val - original_val), 'prob_contribution': float(max_prob - initial_prob)}}
                optimization_thought = f"优化思路：所有参数均在正常范围内。系统启动**归因分析策略**，发现参数“{PARAM_MAP.get(feature_name)}”是对本次不合格结果负面影响最大的因素（据SHAP分析）。因此，系统优先对该参数进行微调，以寻求最大概率提升。"
                message = f"调整后，预测合格概率可提升至 {max_prob:.2%}"
                return adjustments, max_prob, message, optimization_thought

    return {}, initial_prob, "未能找到有效的参数调整建议。", "系统在当前状态下未找到明确的单步优化路径。可能需要工艺员进行多参数复合调整或检查其他外部因素。"

# --- 路由与功能 ---
@app.route('/', methods=['GET'])
def index():
    model_ready = bool(model_cache.get('is_ready', False))
    displayable_params = []
    if model_ready:
        displayable_params = [(k, v) for k, v in model_cache['param_map'].items() if not any(x in k for x in ['product_id', 'ratio', 'indicator', 'density'])]
    return render_template('index.html', model_ready=model_ready, cache=model_cache, displayable_params=displayable_params)

@app.route('/train', methods=['POST'])
def train():
    # ... (代码与上一版本相同) ...
    return redirect(url_for('index'))

@app.route('/api/process_monitor', methods=['POST'])
def process_monitor():
    # ... (代码与上一版本相同) ...
    return jsonify({'status': 'warning' if warnings else 'ok', 'messages': warnings or ['所有参数均在黄金基线范围内']})

@app.route('/api/recommend_params', methods=['POST'])
def recommend_params():
    # ... (代码与上一版本相同) ...
    return jsonify({'product_id': product_id, 'recommended_params': params, 'message': msg, 'param_map': PARAM_MAP})

@app.route('/api/predict', methods=['POST'])
def predict():
    # ... (代码与上一版本相同，确保返回 status, verdict_reason, input_data, shap_values) ...
    return jsonify({'status': final_status, 'prob': float(prob_ok), 'threshold': float(model_cache['best_threshold']), 'waterfall_plot': fig_to_base64(fig), 'input_data': data, 'shap_values': shap_values[0].tolist(), 'verdict_reason': verdict_reason})

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

import os
import io
import base64
import logging
import random
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import NearestNeighbors

# 配置
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']; plt.rcParams['axes.unicode_minus'] = False
app = Flask(__name__); app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据映射与缓存
PARAM_MAP = {
    "product_id": "产品ID", "F_cut_act": "刀头实际压力 (N)", "v_cut_act": "切割实际速度 (mm/s)",
    "F_break_peak": "崩边力峰值 (N)", "v_wheel_act": "磨轮线速度 (m/s)", "F_wheel_act": "磨轮压紧力 (N)",
    "P_cool_act": "冷却水压力 (bar)", "t_glass_meas": "玻璃厚度 (mm)", "pressure_speed_ratio": "压速比",
    "stress_indicator": "应力指标", "energy_density": "能量密度"
}
model_cache = {}

# 辅助函数和建议生成函数 (保持不变)
def fig_to_base64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight'); plt.close(fig); return base64.b64encode(buf.getvalue()).decode('utf-8')
def generate_tabular_adjustment_suggestions(current_params, feature_names, shap_values, golden_baseline):
    suggestions_data, problematic_params, negative_shap_total = [], {}, 0
    current_df = pd.DataFrame([current_params])
    current_df['pressure_speed_ratio'] = current_df['F_cut_act'] / current_df['v_cut_act']; current_df['stress_indicator'] = current_df['F_break_peak'] / current_df['t_glass_meas']; current_df['energy_density'] = current_df['F_cut_act'] * current_df['v_cut_act']
    current_df.replace([np.inf, -np.inf], np.nan, inplace=True); current_df = current_df.fillna(model_cache.get('feature_defaults', {}))
    current_values_array = current_df[feature_names].values.flatten()
    for i, feature in enumerate(feature_names):
        if any(k in feature for k in ['ratio', 'indicator', 'density']): continue
        is_out_of_baseline, is_shap_negative = False, shap_values[i] < -0.001
        mean, std, current_val = golden_baseline['mean'].get(feature), golden_baseline['std'].get(feature), current_values_array[i]
        if mean is not None and std is not None and not (mean - 3 * std <= current_val <= mean + 3 * std): is_out_of_baseline = True
        if is_out_of_baseline or is_shap_negative:
            problematic_params[feature] = {'shap_value': shap_values[i], 'current_value': current_val, 'target_value': mean if mean is not None else current_val}
            if is_shap_negative: negative_shap_total += abs(shap_values[i])
    if not problematic_params: return [], "AI模型分析未找到明确的可优化参数。", ""
    sorted_problems = sorted(problematic_params.items(), key=lambda item: item[1]['shap_value'])
    for feature_name, details in sorted_problems:
        adjustment = details['target_value'] - details['current_value']
        contribution = (abs(details['shap_value']) / negative_shap_total) * 100 if negative_shap_total > 0 and details['shap_value'] < 0 else 0
        suggestions_data.append({"display_name": PARAM_MAP.get(feature_name, feature_name), "current_value": details['current_value'], "adjustment_amount": adjustment, "target_value": details['target_value'], "contribution": contribution})
    return suggestions_data, f"共生成 {len(suggestions_data)} 条优化建议。", "基于AI模型分析，修正以下参数可最大化提升合格率。"

@app.route('/', methods=['GET'])
def index():
    model_ready = bool(model_cache.get('is_ready', False)); displayable_params = []; baseline_ranges = {}
    if model_ready:
        all_params = model_cache.get('param_map', {}); gb = model_cache.get('golden_baseline', {})
        for key, name in all_params.items():
            if not any(x in key for x in ['product_id', 'ratio', 'indicator', 'density']):
                displayable_params.append((key, name))
                mean, std = gb.get('mean', {}).get(key), gb.get('std', {}).get(key)
                if mean is not None and std is not None: baseline_ranges[key] = f"[{mean - 3 * std:.2f} ~ {mean + 3 * std:.2f}]"
    return render_template('index.html', model_ready=model_ready, cache=model_cache, displayable_params=displayable_params, baseline_ranges=baseline_ranges)

@app.route('/train', methods=['POST'])
def train():
    global model_cache, PARAM_MAP; model_cache.clear()
    if 'file' not in request.files: flash("未选择文件", "error"); return redirect(url_for('index'))
    file = request.files['file']
    if not file or file.filename == '': flash("文件无效或文件名为空", "error"); return redirect(url_for('index'))
    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
        df['pressure_speed_ratio'] = df['F_cut_act'] / df['v_cut_act']; df['stress_indicator'] = df['F_break_peak'] / df['t_glass_meas']; df['energy_density'] = df['F_cut_act'] * df['v_cut_act']
        df.replace([np.inf, -np.inf], np.nan, inplace=True); features_to_use = [f for f in PARAM_MAP.keys() if f != "product_id"]; X = df[features_to_use].copy(); y = df["OK_NG"]; X = X.fillna(X.mean())
        total_samples, count_ok, count_ng = len(df), int(y.value_counts().get(1, 0)), int(y.value_counts().get(0, 0))
        data_analysis = {'total_samples': total_samples, 'count_ok': count_ok, 'count_ng': count_ng, 'ok_percentage': (count_ok / total_samples) * 100 if total_samples > 0 else 0, 'ng_percentage': (count_ng / total_samples) * 100 if total_samples > 0 else 0}
        scale_pos_weight_value = count_ng / count_ok if count_ok > 0 else 1
        clf = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, gamma=0.1, random_state=42, use_label_encoder=False, scale_pos_weight=scale_pos_weight_value, eval_metric='logloss')
        clf.fit(X, y); probs_ok = clf.predict_proba(X)[:, 1]; best_f1_macro, best_thresh = 0.0, 0.5
        for t in np.arange(0.01, 1.0, 0.01):
            f1 = f1_score(y, (probs_ok >= t).astype(int), average='macro', zero_division=0)
            if f1 > best_f1_macro: best_f1_macro, best_thresh = f1, t
        fig_importance = plt.figure(); xgb.plot_importance(clf, max_num_features=10, ax=plt.gca()); plt.tight_layout()
        y_pred_final = (probs_ok >= best_thresh).astype(int)
        metrics = {'accuracy': accuracy_score(y, y_pred_final), 'recall_ng': recall_score(y, y_pred_final, pos_label=0), 'precision_ng': precision_score(y, y_pred_final, pos_label=0), 'f1_ng': f1_score(y, y_pred_final, pos_label=0)}
        ok_df = df[df['OK_NG'] == 1].copy(); ng_df = df[df['OK_NG'] == 0].copy()
        knn_model, ok_df_features = None, ok_df[features_to_use]
        if not ok_df_features.empty:
            knn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree'); knn_model.fit(ok_df_features)
        model_cache = {
            'model': clf, 'features': features_to_use, 'best_threshold': best_thresh, 'feature_defaults': X.mean().to_dict(),
            'golden_baseline': {'mean': ok_df[features_to_use].mean().to_dict(), 'std': ok_df[features_to_use].std().to_dict()},
            'ok_df_for_knn': ok_df_features, 'knn_model': knn_model, 'feature_plot': fig_to_base64(fig_importance), 
            'ng_samples_for_sim': ng_df[features_to_use].to_dict('records'), # --- 缓存不合格品样本 ---
            'metrics': metrics, 'data_analysis': data_analysis, 'filename': file.filename, 'is_ready': True, 'param_map': PARAM_MAP
        }
        flash(f"文件 '{file.filename}' 上传成功，分类器与推荐模型已完成训练！", "success")
    except Exception as e: flash(f"处理文件时出错: {e}", "error")
    return redirect(url_for('index'))

# --- !!! 核心修改 2: 增强的失效模拟 !!! ---
@app.route('/api/simulation/generate', methods=['GET'])
def generate_simulated_data():
    if not model_cache.get('is_ready'): return jsonify({'error': '模型未训练'}), 400
    params, warnings, gb = {}, [], model_cache['golden_baseline']
    features = [f for f in PARAM_MAP.keys() if f not in ['product_id', 'pressure_speed_ratio', 'stress_indicator', 'energy_density']]
    is_failure_scenario = random.random() < (1/6)
    
    if is_failure_scenario and model_cache.get('ng_samples_for_sim'):
        # 使用真实的不合格品作为模板
        base_params = random.choice(model_cache['ng_samples_for_sim'])
        for feature in features:
            # 在真实不合格品基础上增加少量噪音
            params[feature] = base_params.get(feature, 0) * random.uniform(0.98, 1.02)
    else:
        # 正常生成合格品
        for feature in features:
            mean, std = gb.get('mean', {}).get(feature, 1), gb.get('std', {}).get(feature, 0.1)
            params[feature] = mean + random.uniform(-1, 1) * std
            
    for feature, val in params.items():
        mean, std = gb.get('mean', {}).get(feature, 1), gb.get('std', {}).get(feature, 0.1)
        if not (mean - 3 * std <= val <= mean + 3 * std): warnings.append(feature)
    return jsonify({'params': params, 'warnings': warnings, 'product_id': f"SIM-{str(uuid.uuid4())[:8]}", 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

@app.route('/api/simulation/get_full_prediction', methods=['POST'])
def get_full_prediction():
    if not model_cache.get('is_ready'): return jsonify({'error': '模型未训练'}), 400
    data = request.get_json()
    features = model_cache['features']; input_df = pd.DataFrame([data['params']])
    input_df['pressure_speed_ratio'] = input_df['F_cut_act'] / input_df['v_cut_act']; input_df['stress_indicator'] = input_df['F_break_peak'] / input_df['t_glass_meas']; input_df['energy_density'] = input_df['F_cut_act'] * input_df['v_cut_act']
    input_df.replace([np.inf, -np.inf], np.nan, inplace=True); input_df = input_df.fillna(model_cache['feature_defaults']); input_vector = input_df[features]
    model = model_cache['model']; explainer = shap.TreeExplainer(model); shap_values_raw = explainer.shap_values(input_vector)
    shap_values = shap_values_raw[0] if isinstance(shap_values_raw, list) and len(shap_values_raw) > 0 and isinstance(shap_values_raw[0], np.ndarray) else shap_values_raw
    if shap_values.ndim > 1: shap_values = shap_values.flatten()
    prob_ok = model.predict_proba(input_vector)[0, 1]; model_says_ng = bool(prob_ok < model_cache['best_threshold']); final_status = "ng" if model_says_ng else "ok"
    baseline_warnings = []; gb = model_cache['golden_baseline']
    for feature, val in data['params'].items():
        if feature in gb['mean']:
            m, s = gb['mean'][feature], gb['std'][feature]
            if not (m - 3*s <= val <= m + 3*s): baseline_warnings.append(f"参数 '{PARAM_MAP.get(feature, feature)}' 超出黄金基线。")
    verdict_reason = [f"AI模型预测合格概率为 {prob_ok:.2%}"] + baseline_warnings
    fig = plt.figure()
    shap.plots.waterfall(shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=input_vector.iloc[0], feature_names=features), show=False, max_display=10); plt.tight_layout()
    return jsonify({'id': data['product_id'], 'params': data['params'], 'status': final_status, 'prob': float(prob_ok), 'threshold': float(model_cache['best_threshold']), 'shap_values': shap_values.tolist(), 'verdict_reason': verdict_reason, 'waterfall_plot': fig_to_base64(fig)})

@app.route('/api/recommend_params', methods=['POST'])
def recommend_params():
    if not model_cache.get('is_ready'): return jsonify({'error': '模型未训练'}), 400
    knn_model, ok_df = model_cache.get('knn_model'), model_cache.get('ok_df_for_knn')
    if knn_model is None or ok_df is None or ok_df.empty: return jsonify({'error': '推荐模型不可用，合格品数据不足。'}), 400
    query_dict = model_cache['feature_defaults']; query_df = pd.DataFrame([query_dict])[model_cache['features']]
    distances, indices = knn_model.kneighbors(query_df)
    recommended_params_df = ok_df.iloc[indices[0]].mean(); params = recommended_params_df.to_dict()
    msg = f"已通过机器学习(KNN)从{len(indices[0])}个最优历史案例中生成推荐参数。"
    return jsonify({'recommended_params': params, 'message': msg, 'param_map': PARAM_MAP})

@app.route('/api/adjust', methods=['POST'])
def adjust():
    if not model_cache.get('is_ready'): return jsonify({'error': '模型未训练'}), 400
    data = request.get_json()
    try:
        features, gb = model_cache['features'], model_cache['golden_baseline']
        input_data, shap_values = data.get('input_data'), data.get('shap_values')
        suggestions, message, footer_note = generate_tabular_adjustment_suggestions(input_data, features, np.array(shap_values), gb)
        return jsonify({ 'suggestions_table': suggestions, 'message': message, 'footer_note': footer_note })
    except Exception as e:
        app.logger.error(f"参数调整时出错: {e}", exc_info=True)
        return jsonify({'error': f'参数调整失败: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)


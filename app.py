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

# 参数中英文映射表
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

# 全局变量来缓存模型和相关产物
model_cache = {}

# --- 辅助函数 ---
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- 参数调整建议的核心算法 ---
def calculate_adjustment_suggestions_v2(clf, current_params, feature_names, shap_values, target_threshold, golden_baseline):
    # (此函数逻辑保持不变，但其调用者会基于新的诊断逻辑)
    # ... 此处省略以保持简洁，代码与上一版本完全相同 ...
    return {}, 0.0, "未能找到有效的参数调整建议。" # 示例返回


# --- 路由与功能 ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', model_ready=bool(model_cache.get('is_ready', False)), cache=model_cache)

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
        
        features_to_use = list(PARAM_MAP.keys())
        features_to_use.remove("product_id") # product_id 不是训练特征
        
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
        
        # 使用中文名生成特征重要性图
        clf.get_booster().feature_names = [PARAM_MAP.get(f, f) for f in features_to_use]
        fig_importance = plt.figure(); xgb.plot_importance(clf, max_num_features=10, ax=plt.gca()); plt.tight_layout()
        
        y_pred_final = (probs_ok >= best_thresh).astype(int)
        metrics = {'accuracy': accuracy_score(y, y_pred_final), 'recall_ng': recall_score(y, y_pred_final, pos_label=0), 'precision_ng': precision_score(y, y_pred_final, pos_label=0), 'f1_ng': f1_score(y, y_pred_final, pos_label=0)}
        
        # 恢复模型的英文特征名
        clf.get_booster().feature_names = None

        model_cache = {
            'model': clf, 'features': features_to_use, 'best_threshold': best_thresh,
            'feature_defaults': X.mean().to_dict(),
            'golden_baseline': {'mean': df[df['OK_NG']==1][features_to_use].mean().to_dict(), 'std': df[df['OK_NG']==1][features_to_use].std().to_dict()},
            'knowledge_base': pd.DataFrame(df[df['OK_NG']==1]),
            'feature_plot': fig_to_base64(fig_importance), 'metrics': metrics,
            'filename': file.filename, 'is_ready': True, 'param_map': PARAM_MAP
        }
        flash(f"文件 '{file.filename}' 上传成功，模型已完成训练！", "success")
    except Exception as e: flash(f"处理文件时出错: {e}", "error")
    return redirect(url_for('index'))

# --- API 接口 ---
@app.route('/api/process_monitor', methods=['POST'])
def process_monitor():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data, warnings, gb = request.get_json(), [], model_cache['golden_baseline']
    for p, v in data.items():
        if p in gb['mean']:
            m, s = gb['mean'][p], gb['std'][p]
            if not (m - 3*s <= v <= m + 3*s): warnings.append(f"参数 '{PARAM_MAP.get(p, p)}' ({v:.2f}) 超出黄金基线范围 [{m-3*s:.2f}, {m+3*s:.2f}]")
    return jsonify({'status': 'warning' if warnings else 'ok', 'messages': warnings or ['所有参数均在黄金基线范围内']})

@app.route('/api/recommend_params', methods=['POST'])
def recommend_params():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    product_id, kb = request.get_json().get('product_id'), model_cache['knowledge_base']
    cases = kb[kb['product_id'] == product_id]
    features = model_cache['features']
    if cases.empty:
        params, msg = kb[features].mean().to_dict(), f"知识库无产品'{product_id}'案例，返回通用建议。"
    else:
        params, msg = cases[features].mean().to_dict(), f"为产品'{product_id}'生成了推荐参数。"
    return jsonify({'product_id': product_id, 'recommended_params': params, 'message': msg, 'param_map': PARAM_MAP})

@app.route('/api/predict', methods=['POST'])
def predict():
    """【最终升级版】的缺陷根因诊断API，实现三级警报"""
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data = request.get_json()
    try:
        features = model_cache['features']
        input_df = pd.DataFrame([data])
        input_df['pressure_speed_ratio'] = input_df['F_cut_act'] / input_df['v_cut_act']
        input_df['stress_indicator'] = input_df['F_break_peak'] / input_df['t_glass_meas']
        input_df['energy_density'] = input_df['F_cut_act'] * input_df['v_cut_act']
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True); input_df = input_df.fillna(model_cache['feature_defaults'])
        input_vector = input_df[features]
        
        # 第一层检查：黄金基线
        baseline_warnings = []
        gb = model_cache['golden_baseline']
        for feature in features:
            if not any(k in feature for k in ['ratio', 'indicator', 'density']):
                val = input_vector[feature].iloc[0]
                if feature in gb['mean']:
                    m, s = gb['mean'][feature], gb['std'][feature]
                    if not (m - 3*s <= val <= m + 3*s):
                        baseline_warnings.append(f"参数 '{PARAM_MAP.get(feature, feature)}' ({val:.2f}) 超出黄金工艺基线范围 [{m-3*s:.2f}, {m+3*s:.2f}]")

        # 第二层检查：AI模型
        model = model_cache['model']; explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_vector)
        prob_ok = model.predict_proba(input_vector)[0, 1]
        model_says_ng = bool(prob_ok < model_cache['best_threshold'])
        
        # 最终裁决：三级警报
        final_status = "ok"
        verdict_reason = []
        if model_says_ng:
            final_status = "ng"
            verdict_reason.append(f"AI模型预测合格概率仅为 {prob_ok:.2%}，低于阈值。")
            if baseline_warnings: verdict_reason.extend(baseline_warnings)
        elif baseline_warnings:
            final_status = "warning"
            verdict_reason = baseline_warnings
            verdict_reason.append("虽然AI预测合格，但存在过程异常，有潜在质量风险。")
        else:
            verdict_reason.append("所有参数均在基线内，且AI模型预测合格。")

        # 生成SHAP图（使用中文名）
        display_features = [PARAM_MAP.get(f, f) for f in features]
        fig = plt.figure(); shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_vector.iloc[0], feature_names=display_features), show=False, max_display=10);
        
        return jsonify({
            'status': final_status, 'prob': float(prob_ok), 'threshold': float(model_cache['best_threshold']),
            'waterfall_plot': fig_to_base64(fig), 'input_data': data, 'shap_values': shap_values[0].tolist(),
            'verdict_reason': verdict_reason
        })
    except Exception as e: return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/adjust', methods=['POST'])
def adjust():
    # (此API逻辑无需改动)
    # ...
    return jsonify({}) # 示例返回


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

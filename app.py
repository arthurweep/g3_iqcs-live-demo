import os
import io
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

model_cache = {}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/', methods=['GET'])
def index():
    model_ready = bool(model_cache.get('is_ready', False))
    displayable_params = []
    if model_ready and 'param_map' in model_cache:
        displayable_params = [
            (k, v) for k, v in model_cache['param_map'].items()
            if not any(x in k for x in ['product_id', 'ratio', 'indicator', 'density'])
        ]
    return render_template('index.html', model_ready=model_ready, cache=model_cache, displayable_params=displayable_params)

@app.route('/train', methods=['POST'])
def train():
    global model_cache, PARAM_MAP
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
        df['pressure_speed_ratio'] = df['F_cut_act'] / df['v_cut_act']
        df['stress_indicator'] = df['F_break_peak'] / df['t_glass_meas']
        df['energy_density'] = df['F_cut_act'] * df['v_cut_act']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_to_use = [f for f in PARAM_MAP.keys() if f != "product_id"]
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
        fig_importance = plt.figure()
        xgb.plot_importance(clf, max_num_features=10, ax=plt.gca())
        plt.tight_layout()
        y_pred_final = (probs_ok >= best_thresh).astype(int)
        metrics = {'accuracy': accuracy_score(y, y_pred_final), 'recall_ng': recall_score(y, y_pred_final, pos_label=0), 'precision_ng': precision_score(y, y_pred_final, pos_label=0), 'f1_ng': f1_score(y, y_pred_final, pos_label=0)}
        model_cache = {'model': clf, 'features': features_to_use, 'best_threshold': best_thresh, 'feature_defaults': X.mean().to_dict(), 'golden_baseline': {'mean': df[df['OK_NG']==1][features_to_use].mean().to_dict(), 'std': df[df['OK_NG']==1][features_to_use].std().to_dict()}, 'knowledge_base': pd.DataFrame(df[df['OK_NG']==1]), 'feature_plot': fig_to_base64(fig_importance), 'metrics': metrics, 'filename': file.filename, 'is_ready': True, 'param_map': PARAM_MAP}
        flash(f"文件 '{file.filename}' 上传成功，模型已完成训练！", "success")
    except Exception as e:
        flash(f"处理文件时出错: {e}", "error")
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

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
from sklearn.utils.class_weight import compute_sample_weight

# --- 初始化与配置 ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # 用于flash消息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量来缓存模型和相关产物，这是我们服务器的“记忆”
model_cache = {}

# --- 辅助函数 ---
def fig_to_base64(fig):
    """将matplotlib图像转换为base64字符串以便在HTML中显示"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- 路由与功能 ---

@app.route('/', methods=['GET'])
def index():
    """渲染主页面"""
    # 页面会根据model_cache中是否有内容来动态显示
    return render_template('index.html', model_ready=bool(model_cache.get('is_ready', False)), cache=model_cache)

@app.route('/train', methods=['POST'])
def train():
    """处理文件上传并实时训练模型"""
    global model_cache
    
    # 在每次训练前清空旧的缓存
    model_cache.clear() 
    app.logger.info("--- 开始处理文件上传和实时模型训练 ---")

    if 'file' not in request.files:
        flash("未选择文件", "error")
        return redirect(url_for('index'))

    file = request.files['file']
    if not file or file.filename == '':
        flash("文件无效或文件名为空", "error")
        return redirect(url_for('index'))

    try:
        # --- 1. 加载上传的数据 ---
        df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
        app.logger.info(f"成功加载上传的数据集，共 {len(df)} 条记录。")
        app.logger.info(f"OK/NG 分布情况:\n{df['OK_NG'].value_counts().to_string()}")

        # --- 2. 强化特征工程 ---
        app.logger.info("正在创建基于物理机理的特征...")
        df['pressure_speed_ratio'] = df['F_cut_act'] / df['v_cut_act']
        df['stress_indicator'] = df['F_break_peak'] / df['t_glass_meas']
        df['energy_density'] = df['F_cut_act'] * df['v_cut_act']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # --- 3. 数据预处理 ---
        features_to_use = [
            "F_cut_act", "v_cut_act", "F_break_peak", "v_wheel_act",
            "F_wheel_act", "P_cool_act", "t_glass_meas",
            "pressure_speed_ratio", "stress_indicator", "energy_density"
        ]
        X = df[features_to_use].copy()
        y = df["OK_NG"]
        X = X.fillna(X.mean())

        # --- 4. 核心算法升级：处理数据不平衡 ---
        app.logger.info("正在训练XGBoost模型...")
        count_ok = (y == 1).sum()
        count_ng = (y == 0).sum()
        scale_pos_weight_value = count_ok / count_ng if count_ng > 0 else 1
        app.logger.info(f"计算出的 scale_pos_weight 值为: {scale_pos_weight_value:.2f}")

        clf = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, gamma=0.1, random_state=42, use_label_encoder=False,
            scale_pos_weight=scale_pos_weight_value,
            eval_metric='logloss'
        )
        clf.fit(X, y)
        app.logger.info("模型训练完成。")

        # --- 5. 寻找最佳F1阈值 ---
        app.logger.info("正在寻找最佳F1分数阈值...")
        probs_ok = clf.predict_proba(X)[:, 1]
        best_f1_macro, best_thresh = 0.0, 0.5
        for t_candidate in np.arange(0.01, 1.0, 0.01):
            y_pred = (probs_ok >= t_candidate).astype(int)
            f1_macro_current = f1_score(y, y_pred, average='macro', zero_division=0)
            if f1_macro_current > best_f1_macro:
                best_f1_macro = f1_macro_current
                best_thresh = t_candidate
        app.logger.info(f"找到最佳阈值: {best_thresh:.2f}")

        # --- 6. 生成图表和指标 ---
        # 特征重要性图
        fig_importance = plt.figure()
        xgb.plot_importance(clf, max_num_features=10, ax=plt.gca())
        plt.tight_layout()
        feature_plot_base64 = fig_to_base64(fig_importance)

        # 计算详细指标
        y_pred_final = (probs_ok >= best_thresh).astype(int)
        metrics = {
            'accuracy': accuracy_score(y, y_pred_final),
            'recall_ok': recall_score(y, y_pred_final, pos_label=1),
            'recall_ng': recall_score(y, y_pred_final, pos_label=0),
            'precision_ok': precision_score(y, y_pred_final, pos_label=1),
            'precision_ng': precision_score(y, y_pred_final, pos_label=0),
            'f1_ok': f1_score(y, y_pred_final, pos_label=1),
            'f1_ng': f1_score(y, y_pred_final, pos_label=0),
        }

        # --- 7. 将所有产物存入全局缓存 ---
        app.logger.info("正在将所有训练产物存入内存缓存...")
        model_cache = {
            'model': clf,
            'features': features_to_use,
            'best_threshold': best_thresh,
            'feature_defaults': X.mean().to_dict(),
            'golden_baseline': {
                'mean': df[df['OK_NG']==1][features_to_use].mean().to_dict(),
                'std': df[df['OK_NG']==1][features_to_use].std().to_dict()
            },
            'knowledge_base': pd.DataFrame(df[df['OK_NG']==1].to_dict('records')),
            'feature_plot': feature_plot_base64,
            'metrics': metrics,
            'filename': file.filename,
            'is_ready': True
        }
        flash(f"文件 '{file.filename}' 上传成功，模型已完成训练！", "success")

    except Exception as e:
        app.logger.error(f"处理文件时发生错误: {e}", exc_info=True)
        flash(f"处理文件时出错: {e}", "error")

    return redirect(url_for('index'))

# --- API 接口 ---
@app.route('/api/process_monitor', methods=['POST'])
def process_monitor():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data = request.get_json()
    warnings = []
    golden_baseline = model_cache['golden_baseline']
    for param, value in data.items():
        if param in golden_baseline['mean']:
            mean = golden_baseline['mean'][param]
            std = golden_baseline['std'][param]
            upper_bound = mean + 3 * std
            lower_bound = mean - 3 * std
            if not (lower_bound <= value <= upper_bound):
                warnings.append(f"参数 '{param}' 的值 ({value:.2f}) 超出黄金基线范围 [{lower_bound:.2f}, {upper_bound:.2f}]")
    if warnings: return jsonify({'status': 'warning', 'messages': warnings})
    return jsonify({'status': 'ok', 'message': '所有参数均在黄金基线范围内'})

@app.route('/api/recommend_params', methods=['POST'])
def recommend_params():
    if not model_cache.get('is_ready'): return jsonify({'error': '请先上传数据并训练模型'}), 400
    data = request.get_json()
    product_id = data.get('product_id')
    knowledge_base = model_cache['knowledge_base']
    successful_cases = knowledge_base[knowledge_base['product_id'] == product_id]
    if successful_cases.empty:
        recommended_params = knowledge_base[model_cache['features']].mean().to_dict()
        message = f"知识库中没有产品 '{product_id}' 的案例，返回所有合格品的平均值作为通用建议。"
    else:
        recommended_params = successful_cases[model_cache['features']].mean().to_dict()
        message = f"为产品 '{product_id}' 生成了推荐参数。"
    return jsonify({'product_id': product_id, 'recommended_params': recommended_params, 'message': message})

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

        model = model_cache['model']
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_vector)
        prob_ok = model.predict_proba(input_vector)[0, 1]
        is_ng = bool(prob_ok < model_cache['best_threshold'])
        
        fig = plt.figure()
        shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                            base_values=explainer.expected_value, 
                                            data=input_vector.iloc[0], 
                                            feature_names=model_cache['features']), 
                            show=False, max_display=10)
        waterfall_plot_base64 = fig_to_base64(fig)
        return jsonify({'prob': float(prob_ok), 'is_ng': is_ng, 'threshold': float(model_cache['best_threshold']), 'waterfall_plot': waterfall_plot_base64})
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)


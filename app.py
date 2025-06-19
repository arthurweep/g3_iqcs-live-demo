import os
import io
import base64
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, g
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from scipy.stats import t

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 支持中文

# --- Global Cache for Model and Data ---
# 使用g对象在请求上下文中存储缓存，比全局变量更安全
def get_model_cache():
    if 'model_cache' not in g:
        g.model_cache = {
            'model': None,
            'best_threshold': 0.5,
            'features': None,
            'feature_defaults': None,
            'golden_baseline': None,
            'knowledge_base': None,
            'model_performance': None
        }
    return g.model_cache

# --- Helper Functions ---

def train_model(df):
    """训练XGBoost模型并返回模型和相关信息"""
    # 1. 特征工程
    df['pressure_speed_ratio'] = df['F_cut_act'] / df['v_cut_act']
    df['stress_indicator'] = df['F_break_peak'] / df['t_glass_meas']
    df['energy_density'] = df['F_cut_act'] * df['v_cut_act']
    
    features = [col for col in df.columns if col not in ['time', 'Part_ID', 'quality_result']]
    X = df[features]
    y = df['quality_result']

    # 2. 处理类别不平衡
    scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1

    # 3. 训练XGBoost模型
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    model.fit(X, y)

    # 4. 寻找最佳阈值
    y_pred_proba = model.predict_proba(X)[:, 1]
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred_binary = (y_pred_proba > threshold).astype(int)
        f1 = f1_score(y, y_pred_binary, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # 5. 计算模型性能
    y_pred_final = (y_pred_proba > best_threshold).astype(int)
    performance = {
        'accuracy': accuracy_score(y, y_pred_final),
        'recall_ng': recall_score(y, y_pred_final, pos_label=0),
        'precision_ng': precision_score(y, y_pred_final, pos_label=0),
        'f1_score_ng': f1_score(y, y_pred_final, pos_label=0),
        'f1_macro': f1_score(y, y_pred_final, average='macro'),
        'best_threshold': best_threshold
    }
    
    # 6. 生成特征重要性图
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=15, height=0.8)
    plt.title('Feature Importance')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    importance_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return model, features, best_threshold, performance, importance_plot

def get_golden_baseline(df):
    """计算合格品的黄金工艺基线"""
    ok_df = df[df['quality_result'] == 1]
    if ok_df.empty:
        return None, None
    
    features = [col for col in df.columns if col not in ['time', 'Part_ID', 'quality_result']]
    baseline = {}
    n = len(ok_df)
    if n > 1:
        # 使用t分布计算置信区间，更适合小样本
        t_crit = t.ppf(df=n-1, q=1 - (0.0027 / 2)) # 对应3-sigma
        for col in features:
            mean = ok_df[col].mean()
            std = ok_df[col].std()
            margin_of_error = t_crit * std / np.sqrt(n)
            baseline[col] = {
                'mean': mean,
                'std': std,
                'lower': mean - 3 * std, # 使用3-sigma作为硬边界
                'upper': mean + 3 * std
            }
    else: # 样本太少，无法计算标准差
        for col in features:
            mean = ok_df[col].mean()
            baseline[col] = {'mean': mean, 'std': 0, 'lower': mean, 'upper': mean}
            
    return baseline, ok_df[features]

def get_adjustment_suggestion_logic(current_params_df, model_cache):
    """计算参数调整建议的核心逻辑"""
    model = model_cache['model']
    golden_baseline = model_cache['golden_baseline']
    best_threshold = model_cache['best_threshold']

    # 1. 计算原始合格概率
    original_prob = model.predict_proba(current_params_df)[0, 1]

    # 2. 检查哪些参数超出了黄金基线
    out_of_baseline_params = []
    if golden_baseline:
        for param, values in golden_baseline.items():
            if param in current_params_df.columns:
                current_val = current_params_df[param].iloc[0]
                if not (values['lower'] <= current_val <= values['upper']):
                    out_of_baseline_params.append({
                        'param': param,
                        'current': current_val,
                        'target': values['mean']
                    })

    adjusted_params_df = current_params_df.copy()
    suggestion_text = ""

    # 3. 优先调整超出基线的参数
    if out_of_baseline_params:
        suggestion_text = "以下参数偏离黄金基线，建议优先调整：\n"
        for p in out_of_baseline_params:
            suggestion_text += f"- **{p['param']}**: 从 `{p['current']:.2f}` 调整至黄金基线均值 `{p['target']:.2f}` 附近。\n"
            adjusted_params_df[p['param']] = p['target']
    
    # 4. 如果没有超出基线的，根据SHAP值调整影响最大的参数
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(current_params_df)
        
        # 找到负贡献最大的特征
        most_negative_idx = np.argmin(shap_values[0])
        param_to_adjust = current_params_df.columns[most_negative_idx]
        current_val = current_params_df[param_to_adjust].iloc[0]

        # 模拟调整（尝试向黄金基线均值移动10%）
        target_val = golden_baseline[param_to_adjust]['mean']
        adjustment = (target_val - current_val) * 0.1 # 小步调整
        new_val = current_val + adjustment
        
        suggestion_text = f"根据SHAP分析，参数 **{param_to_adjust}** 对不合格影响最大。建议进行微调：\n- **{param_to_adjust}**: 从 `{current_val:.2f}` 调整至 `{new_val:.2f}` 附近。\n"
        adjusted_params_df[param_to_adjust] = new_val

    # 5. 计算调整后的新概率
    adjusted_prob = model.predict_proba(adjusted_params_df)[0, 1]

    return {
        "suggestion": suggestion_text,
        "original_prob_ok": float(original_prob),
        "adjusted_prob_ok": float(adjusted_prob),
        "is_adjusted_ok": bool(adjusted_prob > best_threshold)
    }

# --- Flask Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        df = pd.read_csv(file)
        # 简单的数据清洗
        df.dropna(inplace=True)
        df.rename(columns={'OK/NG': 'quality_result'}, inplace=True)
        df['quality_result'] = df['quality_result'].apply(lambda x: 1 if x == 'OK' else 0)

        model, features, best_threshold, performance, importance_plot = train_model(df)
        
        # 更新缓存
        model_cache = get_model_cache()
        model_cache['model'] = model
        model_cache['features'] = features
        model_cache['best_threshold'] = best_threshold
        model_cache['feature_defaults'] = df[features].mean().to_dict()
        model_cache['golden_baseline'], model_cache['knowledge_base'] = get_golden_baseline(df)
        model_cache['model_performance'] = performance
        
        return jsonify({
            'message': '模型训练成功!',
            'performance': performance,
            'feature_importance_plot': importance_plot
        })
    except Exception as e:
        return jsonify({'error': f'训练失败: {str(e)}'}), 500

### MODIFIED ENDPOINT ###
@app.route('/api/predict', methods=['POST'])
def predict_api():
    model_cache = get_model_cache()
    if not model_cache['model']:
        return jsonify({'error': '模型未训练，请先上传数据训练模型'}), 400

    data = request.json
    try:
        # 使用缓存的特征和默认值来构建输入
        input_data = model_cache['feature_defaults'].copy()
        input_data.update(data)
        current_params_df = pd.DataFrame([input_data], columns=model_cache['features'])

        # 预测
        model = model_cache['model']
        prob_ok = model.predict_proba(current_params_df)[0, 1]
        best_threshold = model_cache['best_threshold']

        # 检查黄金基线
        golden_baseline = model_cache['golden_baseline']
        out_of_spec_params = []
        if golden_baseline:
            for param, values in golden_baseline.items():
                if param in current_params_df.columns:
                    current_val = current_params_df[param].iloc[0]
                    if not (values['lower'] <= current_val <= values['upper']):
                        out_of_spec_params.append(f"{param} (当前: {current_val:.2f}, 正常范围: {values['lower']:.2f}-{values['upper']:.2f})")

        # 判断最终状态
        if prob_ok < best_threshold:
            # NG: 不合格，生成SHAP图并返回诊断信息
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(current_params_df)
            
            plt.figure()
            # 指定字体以支持中文
            plt.rcParams['font.sans-serif'] = ['SimHei'] 
            plt.rcParams['axes.unicode_minus'] = False
            shap.plots.waterfall(shap_values[0], max_display=14, show=False)
            plt.title('不合格样本(NG)诊断分析瀑布图')
            plt.tight_layout()
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            shap_plot = base64.b64encode(img.getvalue()).decode()
            plt.close()

            return jsonify({
                "status": "NG",
                "message": f"产品预测为不合格 (NG)，合格概率仅为 {prob_ok:.2%}，低于阈值 {best_threshold:.2%}",
                "probability_ok": float(prob_ok),
                "shap_plot": shap_plot,
                "show_suggestion_button": True,  # 告知前端显示按钮
                "original_params": data          # 将原始参数传回，用于下一步请求
            })
        
        elif out_of_spec_params:
            # Warning: 预测合格但参数偏离
            return jsonify({
                "status": "Warning",
                "message": f"产品预测为合格 (OK)，但以下参数偏离黄金基线，存在质量风险: {'; '.join(out_of_spec_params)}",
                "probability_ok": float(prob_ok),
                "show_suggestion_button": False
            })
        
        else:
            # OK: 合格
            return jsonify({
                "status": "OK",
                "message": f"产品预测为合格 (OK)，合格概率为 {prob_ok:.2%}",
                "probability_ok": float(prob_ok),
                "show_suggestion_button": False
            })

    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

### NEW ENDPOINT ###
@app.route('/api/get_adjustment_suggestion', methods=['POST'])
def get_adjustment_suggestion_api():
    """
    接收NG样本的原始参数，计算并返回调整建议
    """
    model_cache = get_model_cache()
    if not model_cache['model']:
        return jsonify({'error': '模型未训练'}), 400

    data = request.json
    if 'original_params' not in data:
         return jsonify({'error': '请求中缺少 original_params'}), 400

    try:
        original_params = data['original_params']
        # 使用缓存的特征和默认值来构建输入
        input_data = model_cache['feature_defaults'].copy()
        input_data.update(original_params)
        current_params_df = pd.DataFrame([input_data], columns=model_cache['features'])

        # 调用核心逻辑函数
        suggestion_result = get_adjustment_suggestion_logic(current_params_df, model_cache)

        return jsonify(suggestion_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'获取建议失败: {str(e)}'}), 500


@app.route('/api/monitor', methods=['POST'])
def monitor_api():
    """产线心电图：实时监控参数是否在黄金基线内"""
    model_cache = get_model_cache()
    if not model_cache['golden_baseline']:
        return jsonify({'warning': '黄金基线未建立，无法监控'}), 400
    
    data = request.json
    golden_baseline = model_cache['golden_baseline']
    out_of_spec_params = []
    
    for param, value in data.items():
        if param in golden_baseline:
            spec = golden_baseline[param]
            if not (spec['lower'] <= value <= spec['upper']):
                out_of_spec_params.append(
                    f"{param} (当前: {value:.2f}, 正常范围: {spec['lower']:.2f}-{spec['upper']:.2f})"
                )
    
    if out_of_spec_params:
        return jsonify({
            'status': 'Abnormal',
            'message': '工艺参数异常波动!',
            'details': out_of_spec_params
        })
    else:
        return jsonify({
            'status': 'Normal',
            'message': '工艺参数稳定'
        })

@app.route('/api/recommend', methods=['POST'])
def recommend_api():
    """AI工艺员：根据产品ID推荐开机参数"""
    model_cache = get_model_cache()
    if not model_cache['knowledge_base'] is not None:
        return jsonify({'error': '知识库未建立'}), 400
        
    data = request.json
    part_id = data.get('Part_ID')
    
    if not part_id:
        return jsonify({'error': '请提供Part_ID'}), 400
        
    knowledge_base = model_cache['knowledge_base']
    # 这里的逻辑可以更复杂，例如使用K-NN。为简化，我们直接查找
    # 实际上，需要一个将Part_ID和工艺参数关联起来的原始数据表
    # 此处假设knowledge_base本身就是合格品的参数DF
    
    # 简化版：直接返回所有合格品的平均值作为通用建议
    # 在真实场景中，你会根据Part_ID去筛选
    recommendations = model_cache['golden_baseline']
    recommended_params = {k: v['mean'] for k, v in recommendations.items()}
    
    return jsonify({
        'message': f'为产品 {part_id} 生成的推荐参数',
        'recommended_params': recommended_params
    })

# --- Main Execution ---
if __name__ == '__main__':
    # 为了在某些环境下能找到中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


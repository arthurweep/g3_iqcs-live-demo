import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
from flask import Flask, request, jsonify, render_template

# ==============================================================================
# 配置区域 - Configuration Area
# ==============================================================================

# 设置Matplotlib后端，防止在无GUI的服务器上出错
matplotlib.use('Agg')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化Flask应用
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 确保中文字符正确显示

# 全局变量，用于存储训练好的模型、特征名称和SHAP解释器
model = None
feature_names = None
explainer = None
# 假设类别 0 是“合格”，1 是“不合格”
UNQUALIFIED_CLASS_INDEX = 1 

# --- !!! 关键配置：请根据您的实际工艺要求修改此处的黄金工艺基准 !!! ---
# 格式为: '参数名': {'min': 最小值, 'max': 最大值}
GOLDEN_BASELINE = {
    '参数A': {'min': 10, 'max': 20},
    '参数B': {'min': 5.0, 'max': 15.0},
    '参数C': {'min': 100, 'max': 120},
    '参数D': {'min': 0.8, 'max': 1.2},
    '参数E': {'min': 55, 'max': 65}
    # ... 请继续添加或修改其他参数
}


# ==============================================================================
# 辅助函数 - Helper Functions
# ==============================================================================

def check_baseline(parameters: dict) -> list:
    """
    检查输入参数是否超出黄金工艺基线。
    
    Args:
        parameters (dict): 包含参数名和值的字典。
    
    Returns:
        list: 包含所有超出基线参数的警告信息字符串列表。
    """
    warnings = []
    for param, value in parameters.items():
        if param in GOLDEN_BASELINE:
            baseline = GOLDEN_BASELINE[param]
            if not (baseline['min'] <= value <= baseline['max']):
                warning_msg = f"参数'{param}'当前值({value})超出黄金基线范围[{baseline['min']}, {baseline['max']}]"
                warnings.append(warning_msg)
    return warnings

def generate_multiple_suggestions(parameters: dict, shap_values: np.ndarray) -> list:
    """
    当预测不合格时，生成多个参数的优化建议。
    建议基于SHAP值和是否超出黄金基线。

    Args:
        parameters (dict): 输入的参数字典。
        shap_values (np.ndarray): 对应输入的SHAP值。

    Returns:
        list: 一个包含多个建议的列表，按重要性排序。
    """
    suggestions = []
    problematic_params = {}

    # 1. 识别所有问题参数
    for i, param in enumerate(feature_names):
        current_value = parameters.get(param)
        param_shap_value = shap_values[i]
        
        is_out_of_baseline = False
        if param in GOLDEN_BASELINE:
            baseline = GOLDEN_BASELINE[param]
            if not (baseline['min'] <= current_value <= baseline['max']):
                is_out_of_baseline = True
        
        # 如果SHAP值为正（推动向“不合格”），或者参数本身就超出了基线，则视为问题参数
        if param_shap_value > 0 or is_out_of_baseline:
            problematic_params[param] = {
                'current_value': current_value,
                'shap_value': param_shap_value,
                'is_out_of_baseline': is_out_of_baseline
            }

    if not problematic_params:
        return ["模型判定不合格，但未找到明确的可调整工艺参数。请检查所有输入是否准确。"]

    # 2. 对问题参数按SHAP值（影响力）降序排序
    sorted_problems = sorted(problematic_params.items(), key=lambda item: item[1]['shap_value'], reverse=True)

    # 3. 生成建议
    for param, details in sorted_problems:
        reason = []
        if details['shap_value'] > 0:
            reason.append(f"对不合格结果影响较大(SHAP值: {details['shap_value']:.3f})")
        if details['is_out_of_baseline']:
            reason.append("已超出黄金基线")
        
        suggestion_text = f"参数'{param}' (当前值: {details['current_value']}): "
        
        # 提供具体建议值
        if param in GOLDEN_BASELINE:
            baseline = GOLDEN_BASELINE[param]
            # 建议调整到基线的中点
            target_value = (baseline['min'] + baseline['max']) / 2
            suggestion_text += f"建议调整至黄金基线范围[{baseline['min']}, {baseline['max']}]内，例如 {target_value:.2f}。 "
        else:
            suggestion_text += "请基于工艺经验进行调整。 "
            
        suggestion_text += f"原因: {'; '.join(reason)}。"
        suggestions.append(suggestion_text)
        
    return suggestions


# ==============================================================================
# Flask路由 - Flask Routes
# ==============================================================================

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model_endpoint():
    """
    模型训练接口（示例）
    在实际应用中，您会从上传的文件中读取数据
    """
    global model, feature_names, explainer
    
    # --- 这是一个模拟的训练过程 ---
    logging.info("开始模型训练...")
    try:
        # 假设我们有一个 'training_data.csv' 文件
        # data = pd.read_csv('path/to/your/training_data.csv')
        # X = data.drop('quality_label', axis=1)
        # y = data['quality_label']
        
        # 使用模拟数据进行演示
        num_samples = 100
        data = {name: np.random.uniform(val['min'], val['max'], num_samples) for name, val in GOLDEN_BASELINE.items()}
        df = pd.DataFrame(data)
        # 模拟一个简单的质量标签逻辑
        df['quality_label'] = ((df['参数A'] > 15) & (df['参数C'] < 110)).astype(int)
        
        X = df.drop('quality_label', axis=1)
        y = df['quality_label']
        
        feature_names = X.columns.tolist()
        
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        
        logging.info("模型训练成功！")
        return jsonify({'status': 'success', 'message': '模型已成功训练并加载！'})
    except Exception as e:
        logging.error(f"模型训练失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    接收生产参数，进行质量判定，并返回包含预警和优化建议的结果
    """
    if not model or not explainer:
        return jsonify({'status': 'error', 'message': '模型未训练，请先训练模型！'}), 400

    try:
        # 从POST请求中获取JSON数据
        input_data = request.get_json()
        if not input_data:
            return jsonify({'status': 'error', 'message': '未收到输入数据'}), 400
        
        # 将输入数据转换为DataFrame
        input_df = pd.DataFrame([input_data])
        # 确保列顺序与训练时一致
        input_df = input_df[feature_names] 
        
        # 1. 检查参数是否超出黄金基线
        warnings = check_baseline(input_data)
        
        # 2. 模型预测
        prediction_proba = model.predict_proba(input_df)[0]
        prediction = np.argmax(prediction_proba) # 获取概率最高的类别索引
        
        result_status = "合格" if prediction == 0 else "不合格"
        confidence = prediction_proba[prediction] * 100
        
        # 3. 计算SHAP值
        shap_values = explainer.shap_values(input_df)
        
        # 对于二分类，shap_values可能是一个列表，我们取“不合格”类别的SHAP值
        shap_values_for_unqualified = shap_values[UNQUALIFIED_CLASS_INDEX] if isinstance(shap_values, list) else shap_values
        
        # 4. 准备返回的JSON数据
        response = {
            'status': 'success',
            'prediction': result_status,
            'confidence': f"{confidence:.2f}%",
            'warnings': warnings, # 无论合格与否，都返回预警信息
            'suggestions': []
        }
        
        # 5. 如果不合格，生成多参数优化建议
        if result_status == "不合格":
            suggestions = generate_multiple_suggestions(input_data, shap_values_for_unqualified[0])
            response['suggestions'] = suggestions
            
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"预测过程中发生错误: {e}")
        return jsonify({'status': 'error', 'message': f'预测失败: {str(e)}'}), 500


# ==============================================================================
# 启动应用 - Application Runner
# ==============================================================================

if __name__ == '__main__':
    # Render.com会设置PORT环境变量
    port = int(os.environ.get('PORT', 5000))
    # 0.0.0.0表示监听所有可用的网络接口，这对于云部署是必需的
    app.run(host='0.0.0.0', port=port, debug=True)


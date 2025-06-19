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
# 这里的示例值与前端UI图像中的默认输入值相对应，您需要根据实际工艺范围进行调整。
GOLDEN_BASELINE = {
    '参数A': {'min': 10.0, 'max': 20.0}, # 例如，基于15.0的范围
    '参数B': {'min': 5.0, 'max': 15.0},  # 例如，基于10.0的范围
    '参数C': {'min': 100, 'max': 120},   # 例如，基于110的范围
    '参数D': {'min': 0.8, 'max': 1.2},   # 例如，基于0.95的范围
    '参数E': {'min': 50, 'max': 70}     # 例如，基于60的范围
    # ... 请根据您的实际工艺数据，继续添加或修改其他参数
    # 确保这里的参数名（键）与前端HTML中input的'name'属性以及训练数据中的列名完全一致。
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
            # 检查是否超出范围
            if not (baseline['min'] <= value <= baseline['max']):
                warning_msg = (
                    f"参数'{param}'当前值({value})超出黄金基线范围"
                    f"[{baseline['min']}, {baseline['max']}]"
                )
                warnings.append(warning_msg)
    return warnings

def generate_multiple_suggestions(parameters: dict, shap_values: np.ndarray) -> list:
    """
    当预测不合格时，生成多个参数的优化建议。
    建议基于SHAP值和是否超出黄金基线。

    Args:
        parameters (dict): 输入的参数字典。
        shap_values (np.ndarray): 对应输入的SHAP值（通常为针对不合格类别的SHAP值）。

    Returns:
        list: 一个包含多个建议的列表，按重要性排序。
    """
    suggestions = []
    problematic_params = {}

    # 1. 识别所有问题参数
    # 确保 feature_names 已被模型训练初始化
    if not feature_names:
        return ["模型特征名称未定义，无法生成详细建议。"]
        
    for i, param in enumerate(feature_names):
        current_value = parameters.get(param)
        
        # 确保 shap_values 维度正确
        if i >= len(shap_values):
            logging.warning(f"SHAP值索引 {i} 超出 shap_values 范围。")
            continue

        param_shap_value = shap_values[i]
        
        is_out_of_baseline = False
        if param in GOLDEN_BASELINE and current_value is not None:
            baseline = GOLDEN_BASELINE[param]
            if not (baseline['min'] <= current_value <= baseline['max']):
                is_out_of_baseline = True
        
        # 如果SHAP值为正（推动向“不合格”），或者参数本身就超出了基线，则视为问题参数
        # 设定一个SHAP值阈值，避免影响微乎其微的参数
        shap_threshold = 0.01 # 可以根据实际情况调整此阈值
        if abs(param_shap_value) > shap_threshold or is_out_of_baseline:
            problematic_params[param] = {
                'current_value': current_value,
                'shap_value': param_shap_value,
                'is_out_of_baseline': is_out_of_baseline
            }

    if not problematic_params:
        return ["模型判定不合格，但未找到明确的可调整工艺参数。请检查所有输入是否准确，或尝试重新训练模型。"]

    # 2. 对问题参数按SHAP值（影响力，取绝对值降序）排序
    sorted_problems = sorted(problematic_params.items(), key=lambda item: abs(item[1]['shap_value']), reverse=True)

    # 3. 生成建议
    for param, details in sorted_problems:
        reason_parts = []
        if details['is_out_of_baseline']:
            reason_parts.append("已超出黄金基线")
        
        # 判断SHAP值方向，给予具体建议
        if details['shap_value'] > 0: # SHAP值大于0表示该参数向“不合格”方向推动
            reason_parts.append(f"对不合格结果有正向推动(SHAP值: {details['shap_value']:.3f})")
        elif details['shap_value'] < 0: # SHAP值小于0表示该参数向“合格”方向推动
             reason_parts.append(f"对不合格结果有负向推动(SHAP值: {details['shap_value']:.3f})")
            
        suggestion_text = f"参数'{param}' (当前值: {details['current_value']}): "
        
        # 提供具体建议值
        if param in GOLDEN_BASELINE:
            baseline = GOLDEN_BASELINE[param]
            # 建议调整到基线的中点，或者根据SHAP值方向微调
            target_value = (baseline['min'] + baseline['max']) / 2
            
            # 如果当前值高于基线上限，且SHAP值提示需要降低，建议降至上限
            if details['current_value'] > baseline['max'] and details['shap_value'] > 0:
                target_value = baseline['max']
                suggestion_text += f"建议将其值降低至黄金基线上限 {baseline['max']:.2f}。"
            # 如果当前值低于基线下限，且SHAP值提示需要升高，建议升至下限
            elif details['current_value'] < baseline['min'] and details['shap_value'] > 0:
                target_value = baseline['min']
                suggestion_text += f"建议将其值升高至黄金基线下限 {baseline['min']:.2f}。"
            # 如果在基线内但SHAP值异常，建议向有利方向微调
            elif baseline['min'] <= details['current_value'] <= baseline['max'] and details['shap_value'] > 0:
                 suggestion_text += f"建议微调至黄金基线内，如 {target_value:.2f}，以优化结果。"
            else: # 其他情况，建议回到基线中心
                suggestion_text += f"建议调整至黄金基线范围[{baseline['min']}, {baseline['max']}]内，例如 {target_value:.2f}。"
        else:
            suggestion_text += "请基于工艺经验进行调整（无基线参考）。 "
            
        suggestion_text += f" 原因: {'; '.join(reason_parts)}。"
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
    模型训练接口
    在实际应用中，您会从上传的文件中读取数据
    """
    global model, feature_names, explainer
    
    # --- 这是一个模拟的训练过程，请替换为您的实际数据加载和模型训练逻辑 ---
    logging.info("开始模型训练...")
    try:
        # 生产环境通常从数据库或文件加载真实数据
        # 例如：
        # data_path = 'data/your_training_data.csv'
        # if not os.path.exists(data_path):
        #     raise FileNotFoundError(f"训练数据文件未找到: {data_path}")
        # df = pd.read_csv(data_path)
        
        # 使用模拟数据进行演示，数据结构与GOLDEN_BASELINE匹配
        num_samples = 1000 # 增加样本量
        simulated_data = {}
        for name, val in GOLDEN_BASELINE.items():
            # 生成模拟数据时，允许一部分数据超出基线，以模拟真实情况
            range_span = val['max'] - val['min']
            simulated_data[name] = np.random.uniform(val['min'] - range_span*0.5, 
                                                     val['max'] + range_span*0.5, 
                                                     num_samples)
        df = pd.DataFrame(simulated_data)
        
        # 模拟一个更复杂的质量标签逻辑，增加“不合格”样本的代表性
        # 例如，某些参数超出了范围就容易不合格
        df['quality_label'] = 0 # 默认合格
        # 模拟不合格条件：如果参数A或C严重偏离基线，或参数B和D组合异常
        if '参数A' in df.columns and '参数C' in df.columns:
            df.loc[(df['参数A'] < GOLDEN_BASELINE['参数A']['min'] - 2) | 
                   (df['参数A'] > GOLDEN_BASELINE['参数A']['max'] + 2) |
                   (df['参数C'] < GOLDEN_BASELINE['参数C']['min'] - 5) | 
                   (df['参数C'] > GOLDEN_BASELINE['参数C']['max'] + 5), 'quality_label'] = 1
        if '参数B' in df.columns and '参数D' in df.columns:
            df.loc[(df['参数B'] < GOLDEN_BASELINE['参数B']['min'] - 1) & 
                   (df['参数D'] > GOLDEN_BASELINE['参数D']['max'] + 0.05), 'quality_label'] = 1
        
        logging.info(f"模拟数据生成完成，样本数: {len(df)}，合格样本: {len(df[df['quality_label'] == 0])}，不合格样本: {len(df[df['quality_label'] == 1])}")

        X = df.drop('quality_label', axis=1)
        y = df['quality_label']
        
        feature_names = X.columns.tolist()
        
        # 使用XGBoost进行分类，优化参数
        model = xgb.XGBClassifier(
            objective='binary:logistic', # 二分类逻辑回归
            eval_metric='logloss',       # 评估指标
            use_label_encoder=False,     # 抑制警告
            n_estimators=100,            # 迭代次数
            learning_rate=0.1,           # 学习率
            max_depth=5,                 # 树的最大深度
            subsample=0.8,               # 采样率
            colsample_bytree=0.8,        # 列采样率
            random_state=42              # 随机种子
        )
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        
        logging.info("模型训练成功！")
        return jsonify({'status': 'success', 'message': '模型已成功训练并加载！'})
    except Exception as e:
        logging.error(f"模型训练失败: {e}", exc_info=True) # 打印详细错误信息
        return jsonify({'status': 'error', 'message': f"模型训练失败: {str(e)} 请检查训练数据和配置。"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    接收生产参数，进行质量判定，并返回包含预警和优化建议的结果
    """
    if model is None or explainer is None:
        return jsonify({'status': 'error', 'message': '模型未训练或加载，请先进行模型训练！'}), 400

    try:
        # 从POST请求中获取JSON数据
        input_data = request.get_json()
        if not input_data:
            return jsonify({'status': 'error', 'message': '未收到输入数据，请提供JSON格式的参数。'}), 400
        
        # 确保所有预期的特征都在输入数据中，并填充缺失值
        processed_input_data = {}
        for name in feature_names:
            processed_input_data[name] = input_data.get(name, 0.0) # 缺失值用0.0填充，或根据实际情况调整

        input_df = pd.DataFrame([processed_input_data])
        # 确保列顺序与训练时一致
        input_df = input_df[feature_names] 
        
        # 1. 检查参数是否超出黄金基线
        warnings = check_baseline(processed_input_data) # 使用处理过的输入数据进行检查
        
        # 2. 模型预测
        prediction_proba = model.predict_proba(input_df)[0]
        prediction = np.argmax(prediction_proba) # 获取概率最高的类别索引 (0或1)
        
        result_status = "合格" if prediction == 0 else "不合格"
        confidence = prediction_proba[prediction] * 100
        
        # 3. 计算SHAP值
        # SHAP TreeExplainer.shap_values() 返回的是一个列表，每个元素对应一个类别
        # 对于二分类，shap_values[0] 是预测为类别0（合格）的SHAP值
        # shap_values[1] 是预测为类别1（不合格）的SHAP值
        # 我们关注的是导致“不合格”的SHAP值，所以取 UNQUALIFIED_CLASS_INDEX
        
        # 注意：SHAP值为np.ndarray，需要取第一行（因为我们只预测一个样本）
        shap_values_raw = explainer.shap_values(input_df.iloc[0]) # 获取单个样本的SHAP值
        
        # 如果 explainer.shap_values 返回一个列表（多输出模型或分类模型），则取对应类别的SHAP值
        shap_values_for_target_class = shap_values_raw[UNQUALIFIED_CLASS_INDEX] if isinstance(shap_values_raw, list) else shap_values_raw
        
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
            suggestions = generate_multiple_suggestions(processed_input_data, shap_values_for_target_class)
            response['suggestions'] = suggestions
            
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"预测过程中发生错误: {e}", exc_info=True) # 打印详细错误信息
        return jsonify({'status': 'error', 'message': f'预测失败: {str(e)}'}), 500


# ==============================================================================
# 启动应用 - Application Runner
# ==============================================================================

if __name__ == '__main__':
    # Render.com会设置PORT环境变量
    # 在本地开发时，如果没有设置PORT环境变量，会使用5000端口
    port = int(os.environ.get('PORT', 5000))
    # 0.0.0.0表示监听所有可用的网络接口，这对于云部署是必需的
    app.run(host='0.0.0.0', port=port, debug=True)


import os
import io
import base64
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for Matplotlib, crucial for server deployments
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, g
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from scipy.stats import t

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates') # Specify template folder explicitly
app.config['JSON_AS_ASCII'] = False # Supports Chinese characters in JSON responses

# --- Global Cache for Model and Data ---
model_cache = {
    'model': None,
    'best_threshold': 0.5,
    'features': None,
    'feature_defaults': None,
    'golden_baseline': None,
    'knowledge_base': None, # Stores OK samples for AI Craftsman
    'model_performance': None
}

# --- Helper Functions ---

def train_model_logic(df_input):
    """
    Handles the core logic for training the XGBoost model.
    Assumes df_input already has a 'quality_result' column with 0/1 values.
    """
    df = df_input.copy() # Work on a copy to avoid modifying original dataframe unexpectedly

    # Define all expected numerical features from your dataset
    # These are the columns that will be directly used by the model
    base_features = [
        'F_cut_act', 'v_cut_act', 'F_break_peak', 'v_wheel_act', 
        'F_wheel_act', 'P_cool_act', 't_glass_meas'
    ]

    # 1. Feature Engineering
    # Ensure these columns are always present or gracefully handled if missing in input
    # Add a check to avoid division by zero for v_cut_act and t_glass_meas
    df['pressure_speed_ratio'] = df['F_cut_act'] / df['v_cut_act'].replace(0, np.nan).fillna(1)
    df['stress_indicator'] = df['F_break_peak'] / df['t_glass_meas'].replace(0, np.nan).fillna(1)
    df['energy_density'] = df['F_cut_act'] * df['v_cut_act']
    
    # Combine base features and engineered features
    engineered_features = ['pressure_speed_ratio', 'stress_indicator', 'energy_density']
    
    # Filter for actual features present in the dataframe
    features = [col for col in base_features + engineered_features if col in df.columns and col not in ['product_id', 'quality_result']]

    X = df[features]
    y = df['quality_result']

    # 2. Handle Class Imbalance
    num_neg = (y == 0).sum() # Number of NG samples
    num_pos = (y == 1).sum() # Number of OK samples
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1 

    # 3. Train XGBoost Model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False, 
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=100, 
        learning_rate=0.1
    )
    model.fit(X, y)

    # 4. Find Best Threshold based on F1-score for evaluation
    y_pred_proba = model.predict_proba(X)[:, 1] 
    best_f1 = -1 
    best_threshold = 0.5 

    for threshold in np.linspace(0.1, 0.9, 100): 
        y_pred_binary = (y_pred_proba >= threshold).astype(int) 
        f1_mac = f1_score(y, y_pred_binary, average='macro', zero_division=0) 
        if f1_mac > best_f1:
            best_f1 = f1_mac
            best_threshold = threshold

    # 5. Calculate Model Performance Metrics
    y_pred_final = (y_pred_proba >= best_threshold).astype(int)
    performance = {
        'accuracy': accuracy_score(y, y_pred_final),
        'recall_ng': recall_score(y, y_pred_final, pos_label=0, zero_division=0), 
        'precision_ng': precision_score(y, y_pred_final, pos_label=0, zero_division=0), 
        'f1_score_ng': f1_score(y, y_pred_final, pos_label=0, zero_division=0), 
        'f1_macro': best_f1, 
        'best_threshold': best_threshold
    }
    
    # 6. Generate Feature Importance Plot
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=15, height=0.8, importance_type='gain') 
    plt.title('Feature Importance (Gain)', fontsize=16)
    plt.tight_layout()
    img_buffer = io.BytesIO() 
    plt.savefig(img_buffer, format='png', bbox_inches='tight') 
    img_buffer.seek(0) 
    importance_plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8') 
    plt.close() 

    return model, features, best_threshold, performance, importance_plot_base64

def get_golden_baseline_logic(df_full):
    """
    Calculates the golden baseline (mean +/- 3*std) for OK samples.
    Also returns the DataFrame of OK samples as knowledge_base.
    """
    ok_df = df_full[df_full['quality_result'] == 1].copy() 

    if ok_df.empty:
        return None, None 

    # All numerical features from your dataset
    base_features = [
        'F_cut_act', 'v_cut_act', 'F_break_peak', 'v_wheel_act', 
        'F_wheel_act', 'P_cool_act', 't_glass_meas'
    ]
    
    # Engineered features also have baselines
    engineered_features = ['pressure_speed_ratio', 'stress_indicator', 'energy_density']

    # Combine all features to calculate baselines
    all_features_for_baseline = [col for col in base_features + engineered_features if col in ok_df.columns]
    
    golden_baseline = {}
    if len(ok_df) > 1: 
        for col in all_features_for_baseline:
            mean = ok_df[col].mean()
            std = ok_df[col].std()
            golden_baseline[col] = {
                'mean': mean,
                'std': std,
                'lower': mean - 3 * std,
                'upper': mean + 3 * std
            }
    else: 
        for col in all_features_for_baseline:
            mean = ok_df[col].mean()
            golden_baseline[col] = {
                'mean': mean,
                'std': 0,
                'lower': mean,
                'upper': mean
            }
                
    # knowledge_base will be the dataframe of all OK samples' features
    knowledge_base = ok_df[all_features_for_baseline] 
    
    return golden_baseline, knowledge_base

def calculate_engineered_features(data_dict):
    """Calculates engineered features for a single sample (dictionary)."""
    # Create a mutable copy if data_dict is immutable
    data = data_dict.copy() 

    # Use .get() with default values to prevent KeyError if a base feature is missing
    f_cut_act = data.get('F_cut_act', 0)
    v_cut_act = data.get('v_cut_act', 1e-6) # Use a very small number to prevent division by zero
    f_break_peak = data.get('F_break_peak', 0)
    t_glass_meas = data.get('t_glass_meas', 1e-6) # Use a very small number to prevent division by zero

    data['pressure_speed_ratio'] = f_cut_act / v_cut_act
    data['stress_indicator'] = f_break_peak / t_glass_meas
    data['energy_density'] = f_cut_act * v_cut_act
    return data

def get_adjustment_suggestion_logic(current_params_raw, model_cache_data):
    """
    Calculates parameter adjustment suggestions for an NG sample.
    current_params_raw is the original raw input dictionary from the frontend.
    """
    model = model_cache_data['model']
    golden_baseline = model_cache_data['golden_baseline']
    best_threshold = model_cache_data['best_threshold']
    all_feature_names = model_cache_data['features'] # List of all features the model expects

    if not model or not golden_baseline:
        return {"error": "模型或黄金基线未准备好。"}

    # Start with a full set of features to ensure all are present for prediction
    # Use feature_defaults to fill missing columns if any
    input_data_with_defaults = model_cache_data['feature_defaults'].copy()
    input_data_with_defaults.update(current_params_raw) # Overwrite defaults with actual values
    
    # Calculate engineered features for the current sample
    current_params_processed = calculate_engineered_features(input_data_with_defaults)

    # Create a DataFrame for prediction, ensuring columns match training data
    # It's crucial that `all_feature_names` includes all base and engineered features
    current_params_df = pd.DataFrame([current_params_processed], columns=all_feature_names)

    # 1. Calculate Original Probability of OK
    original_prob = model.predict_proba(current_params_df)[:, 1][0] 

    # 2. Identify Parameters Outside Golden Baseline
    out_of_baseline_params = []
    # Create a copy for simulation. This will hold the "adjusted" parameters.
    adjusted_params_for_simulation = current_params_df.copy() 

    for param, values in golden_baseline.items():
        # Check if the parameter exists in the current sample and it's not an engineered feature
        # Only adjust base features to avoid complex inverse calculations for engineered features
        if param in current_params_raw and param in current_params_df.columns:
            current_val = current_params_df[param].iloc[0]
            if not (values['lower'] <= current_val <= values['upper']):
                out_of_baseline_params.append({
                    'param': param,
                    'current_val': current_val,
                    'target_mean': values['mean'],
                    'lower_bound': values['lower'],
                    'upper_bound': values['upper']
                })
                # Adjust the base feature in the simulation DataFrame to its golden mean
                adjusted_params_for_simulation[param] = values['mean']
    
    suggestion_text = ""
    
    if out_of_baseline_params:
        # Prioritize adjusting parameters that are out of golden baseline
        suggestion_text += "发现以下工艺参数偏离黄金基线，建议优先调整：\n"
        for p in out_of_baseline_params:
            suggestion_text += f"- **{p['param']}**: 当前值 `{p['current_val']:.2f}`，建议调整至黄金基线均值 `{p['target_mean']:.2f}` (范围: `{p['lower_bound']:.2f}` ~ `{p['upper_bound']:.2f}`)。\n"
        suggestion_text += "\n" # Add a line break for readability
    else:
        # If all parameters are within baseline, use SHAP to find most impactful negative feature
        explainer = shap.TreeExplainer(model)
        shap_values_obj = explainer(current_params_df)
        shap_values_array = shap_values_obj.values[0] 
        feature_names_shap = shap_values_obj.feature_names
        
        # Find the feature with the most negative contribution (pushing towards NG)
        most_negative_shap_idx = np.argmin(shap_values_array)
        param_to_adjust = feature_names_shap[most_negative_shap_idx]
        current_val_impacted = current_params_df[param_to_adjust].iloc[0]

        # Only suggest adjusting a base feature, not an engineered one
        if param_to_adjust not in current_params_raw:
             # If the most impacting feature is engineered, find its most impacting base component
             # This is a simplification; a more robust solution would involve more complex attribution
            relevant_base_params = [p for p in current_params_raw if p in all_feature_names] # Filter only base features
            if relevant_base_params:
                # Find the base feature with the largest impact among the base features
                base_shap_impacts = {p: shap_values_array[feature_names_shap.index(p)] for p in relevant_base_params if p in feature_names_shap}
                if base_shap_impacts:
                    param_to_adjust = min(base_shap_impacts, key=base_shap_impacts.get) # Parameter with most negative SHAP
                    current_val_impacted = current_params_df[param_to_adjust].iloc[0]
                else: # Fallback if no relevant base parameters
                    param_to_adjust = list(current_params_raw.keys())[0] # Just pick the first base param
                    current_val_impacted = current_params_df[param_to_adjust].iloc[0]
            else:
                param_to_adjust = list(current_params_raw.keys())[0] # Fallback if no relevant base parameters
                current_val_impacted = current_params_df[param_to_adjust].iloc[0]


        target_val_for_impacted = golden_baseline.get(param_to_adjust, {}).get('mean', current_val_impacted)
        # Apply a small step change towards the golden mean for initial suggestion (e.g., 10% of difference)
        step_size = (target_val_for_impacted - current_val_impacted) * 0.1 
        if abs(step_size) < 0.01 and abs(target_val_for_impacted - current_val_impacted) > 0.01: # Ensure a meaningful step if target is different
            step_size = 0.01 if target_val_for_impacted > current_val_impacted else -0.01
            
        new_val_impacted = current_val_impacted + step_size
        
        # Update the simulation DataFrame with this single adjusted base feature
        adjusted_params_for_simulation[param_to_adjust] = new_val_impacted

        suggestion_text += f"当前参数均在基线范围内，但根据AI分析，参数 **{param_to_adjust}** 对不合格预测贡献最大。建议进行微调：\n"
        suggestion_text += f"- **{param_to_adjust}**: 从 `{current_val_impacted:.2f}` 调整至 `{new_val_impacted:.2f}` 附近。\n"

    # Re-calculate ALL engineered features for the _adjusted_ parameters, if any base features changed
    # This is critical because changing a base feature (e.g., F_cut_act) also changes engineered features
    adjusted_params_dict_raw = {}
    for param_name in current_params_raw.keys(): # Only base features
        adjusted_params_dict_raw[param_name] = adjusted_params_for_simulation[param_name].iloc[0] # Get the (potentially adjusted) value

    adjusted_params_re_engineered = calculate_engineered_features(adjusted_params_dict_raw)
    
    # Ensure this DataFrame has all the features the model was trained on
    adjusted_params_final_df = pd.DataFrame([adjusted_params_re_engineered], columns=all_feature_names)

    # 3. Calculate Adjusted Probability of OK
    adjusted_prob = model.predict_proba(adjusted_params_final_df)[:, 1][0]

    return {
        "suggestion": suggestion_text,
        "original_prob_ok": float(original_prob),
        "adjusted_prob_ok": float(adjusted_prob),
        "is_adjusted_to_ok_threshold": bool(adjusted_prob >= best_threshold) 
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
        df.dropna(inplace=True) 

        # Robustly handle the quality result column
        found_quality_col = False
        
        # Prioritize 'OK_NG' based on your provided dataset
        if 'OK_NG' in df.columns:
            df.rename(columns={'OK_NG': 'quality_result'}, inplace=True)
            found_quality_col = True
        # Then check for 'OK/NG'
        elif 'OK/NG' in df.columns:
            df.rename(columns={'OK/NG': 'quality_result'}, inplace=True)
            found_quality_col = True
        elif 'quality_result' in df.columns:
            found_quality_col = True
        else: # Try other common names
            for col_cand in ['Quality', 'Result', 'Status']:
                if col_cand in df.columns:
                    df.rename(columns={col_cand: 'quality_result'}, inplace=True)
                    found_quality_col = True
                    break
        
        if not found_quality_col or 'quality_result' not in df.columns:
            raise ValueError("CSV文件中未找到预期的质量结果列。请确保列名为 'OK_NG'、'OK/NG'、'quality_result'、'Quality'、'Result' 或 'Status'。")

        # Convert quality_result column to 0/1 (0 for NG, 1 for OK)
        # Handle cases where 'OK_NG' might already be 0/1, or 'OK'/'NG' strings
        if df['quality_result'].dtype == 'object':
            df['quality_result'] = df['quality_result'].apply(lambda x: 1 if str(x).strip().upper() == 'OK' or str(x).strip() == '1' else 0)
        else: # Assume it's already numeric (0 or 1)
            df['quality_result'] = df['quality_result'].astype(int)

        # Proceed with training logic
        model, features, best_threshold, performance, importance_plot = train_model_logic(df)
        
        # Update global model_cache
        model_cache['model'] = model
        model_cache['features'] = features
        model_cache['best_threshold'] = best_threshold
        
        # Calculate feature defaults for prediction (mean of all features in the training data)
        # Use only base features for feature_defaults as they are direct inputs
        base_features_from_df = [
            'F_cut_act', 'v_cut_act', 'F_break_peak', 'v_wheel_act', 
            'F_wheel_act', 'P_cool_act', 't_glass_meas'
        ]
        model_cache['feature_defaults'] = df[base_features_from_df].mean().to_dict() 

        # Pass the full df to get_golden_baseline_logic, it will filter by quality_result=1
        model_cache['golden_baseline'], model_cache['knowledge_base'] = get_golden_baseline_logic(df)
        model_cache['model_performance'] = performance
        
        return jsonify({
            'message': '模型训练成功!',
            'performance': performance,
            'feature_importance_plot': importance_plot
        })
    except Exception as e:
        import traceback
        app.logger.error(f"Training failed: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'训练失败: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict_api():
    if not model_cache['model']:
        return jsonify({'error': '模型未训练，请先上传数据训练模型'}), 400

    data_raw = request.json # Raw input from frontend
    
    try:
        # Use feature defaults to ensure all required base features are present
        input_data_for_processing = model_cache['feature_defaults'].copy()
        input_data_for_processing.update(data_raw) # Overwrite defaults with actual input
        
        # Calculate engineered features for the current sample
        current_params_with_engineered = calculate_engineered_features(input_data_for_processing)

        # Create DataFrame for prediction, ensuring column order matches training data
        current_params_df = pd.DataFrame([current_params_with_engineered], columns=model_cache['features'])

        # Predict probability of being OK
        prob_ok = model_cache['model'].predict_proba(current_params_df)[:, 1][0]
        best_threshold = model_cache['best_threshold']

        # Check against golden baseline for warnings
        golden_baseline = model_cache['golden_baseline']
        out_of_spec_params = []
        if golden_baseline: 
            # Check only base features against the baseline, as engineered features are derived
            base_features_from_input = [col for col in data_raw.keys() if col in golden_baseline] # Parameters actually provided by user
            for param in base_features_from_input:
                values = golden_baseline[param]
                current_val = current_params_df[param].iloc[0] # Get value from the processed DF
                if not (values['lower'] <= current_val <= values['upper']):
                    out_of_spec_params.append(f"{param} (当前: {current_val:.2f}, 正常范围: {values['lower']:.2f}-{values['upper']:.2f})")

        # Determine final status
        status_message = ""
        prediction_status = ""
        shap_plot_base64 = None
        show_suggestion_button = False

        if prob_ok < best_threshold:
            prediction_status = "NG"
            status_message = f"产品预测为不合格 (NG)，合格概率仅为 {prob_ok:.2%}，低于阈值 {best_threshold:.2%}"
            
            # Generate SHAP plot for NG samples
            explainer = shap.TreeExplainer(model_cache['model'])
            shap_values_obj = explainer(current_params_df)
            
            plt.figure(figsize=(10, 6))
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif'] 
            plt.rcParams['axes.unicode_minus'] = False 
            shap.plots.waterfall(shap_values_obj[0], max_display=15, show=False) 
            plt.title('不合格样本(NG)诊断分析瀑布图', fontsize=16)
            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            shap_plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close()

            show_suggestion_button = True 
        
        elif out_of_spec_params:
            prediction_status = "Warning"
            status_message = f"产品预测为合格 (OK)，但以下参数偏离黄金基线，存在潜在质量风险: {'; '.join(out_of_spec_params)}"
            show_suggestion_button = False 
        
        else:
            prediction_status = "OK"
            status_message = f"产品预测为合格 (OK)，合格概率为 {prob_ok:.2%}"
            show_suggestion_button = False

        return jsonify({
            "status": prediction_status,
            "message": status_message,
            "probability_ok": float(prob_ok),
            "shap_plot": shap_plot_base64,
            "show_suggestion_button": show_suggestion_button,
            "original_params": data_raw # Pass raw input params back for suggestion request
        })

    except Exception as e:
        app.logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/get_adjustment_suggestion', methods=['POST'])
def get_adjustment_suggestion_api():
    """
    Receives raw NG sample parameters, calculates and returns adjustment suggestions.
    """
    if not model_cache['model']:
        return jsonify({'error': '模型未训练'}), 400

    data = request.json
    original_params_raw = data.get('original_params') 
    if not original_params_raw:
         return jsonify({'error': '请求中缺少 original_params'}), 400

    try:
        suggestion_result = get_adjustment_suggestion_logic(original_params_raw, model_cache)
        return jsonify(suggestion_result)

    except Exception as e:
        app.logger.error(f"Getting adjustment suggestion failed: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'获取建议失败: {str(e)}'}), 500

@app.route('/api/monitor', methods=['POST'])
def monitor_api():
    """AI Heartbeat: Real-time monitoring of parameters against golden baseline."""
    if not model_cache['golden_baseline']:
        return jsonify({'status': 'NotReady', 'message': '黄金基线未建立，无法监控'}), 400
    
    data_raw = request.json # Raw input from frontend
    
    # Calculate engineered features for the current monitor sample
    current_monitor_params = model_cache['feature_defaults'].copy() # Use defaults to ensure all features for baseline check
    current_monitor_params.update(data_raw)
    current_monitor_params_engineered = calculate_engineered_features(current_monitor_params)

    golden_baseline = model_cache['golden_baseline']
    out_of_spec_params = []
    
    # Iterate through golden baseline, checking relevant parameters
    for param_name, specs in golden_baseline.items():
        # Check if this parameter (base or engineered) is in the processed sample
        if param_name in current_monitor_params_engineered: 
            current_val = current_monitor_params_engineered[param_name]
            if not (specs['lower'] <= current_val <= specs['upper']):
                out_of_spec_params.append({
                    "param": param_name,
                    "current": current_val,
                    "range": f"[{specs['lower']:.2f} - {specs['upper']:.2f}]"
                })
    
    if out_of_spec_params:
        return jsonify({
            'status': 'Abnormal',
            'message': '工艺参数异常波动！',
            'details': out_of_spec_params
        })
    else:
        return jsonify({
            'status': 'Normal',
            'message': '工艺参数稳定'
        })

@app.route('/api/recommend', methods=['POST'])
def recommend_api():
    """AI Craftsman: Recommends start-up parameters based on Part_ID."""
    if model_cache['knowledge_base'] is None or model_cache['knowledge_base'].empty:
        return jsonify({'error': '知识库未建立或为空，请先训练模型并确保有合格数据'}), 400
        
    data = request.json
    part_id = data.get('Part_ID')
    
    if not part_id:
        return jsonify({'error': '请提供Part_ID'}), 400
        
    knowledge_base_df = model_cache['knowledge_base'] 
    golden_baseline = model_cache['golden_baseline'] 

    # For now, it returns the overall mean of all OK samples' base features.
    # To make it truly Part_ID specific, the original training data would need 'product_id'
    # and this knowledge_base_df would need to include it or be indexed by it.
    # Then filter knowledge_base_df by part_id, and calculate means for those specific samples.
    
    recommended_params = {}
    # Use only base features for recommendation as they are directly adjustable
    base_features_from_model = [
        'F_cut_act', 'v_cut_act', 'F_break_peak', 'v_wheel_act', 
        'F_wheel_act', 'P_cool_act', 't_glass_meas'
    ]
    if golden_baseline:
        for param in base_features_from_model:
            if param in golden_baseline: # Ensure it has a baseline
                recommended_params[param] = golden_baseline[param]['mean']
    
    return jsonify({
        'message': f'为产品 {part_id} 生成的推荐参数 (目前为所有合格品的平均值)',
        'recommended_params': recommended_params
    })

# --- Main Execution ---
if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


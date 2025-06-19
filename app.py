import os
import io
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 配置matplotlib字体为中文，并兼容云端环境（推荐在云端部署时安装中文字体或使用英文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flash messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义参数的中文映射，用于前端显示和解释
PARAM_MAP = {
    "product_id": "产品ID",
    "F_cut_act": "刀头实际压力 (N)",
    "v_cut_act": "切割实际速度 (mm/s)",
    "F_break_peak": "掰断力峰值 (N)", # Changed from "崩边力峰值" to be consistent with previous context
    "v_wheel_act": "磨轮线速度 (m/s)",
    "F_wheel_act": "磨轮压紧力 (N)",
    "P_cool_act": "冷却水压力 (bar)",
    "t_glass_meas": "玻璃厚度 (mm)",
    "pressure_speed_ratio": "压速比",
    "stress_indicator": "应力指标",
    "energy_density": "能量密度"
}

# 全局模型缓存
model_cache = {}

# --- Helper Functions ---

def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def calculate_engineered_features(data_dict):
    """Calculates engineered features for a single sample (dictionary)."""
    # Create a mutable copy
    data = data_dict.copy() 

    # Use .get() with default values to prevent KeyError if a base feature is missing
    # Use a very small number for division to avoid ZeroDivisionError, instead of directly 0
    f_cut_act = data.get('F_cut_act', 0)
    v_cut_act = data.get('v_cut_act', 1e-6) 
    f_break_peak = data.get('F_break_peak', 0)
    t_glass_meas = data.get('t_glass_meas', 1e-6) 

    data['pressure_speed_ratio'] = f_cut_act / v_cut_act
    data['stress_indicator'] = f_break_peak / t_glass_meas
    data['energy_density'] = f_cut_act * v_cut_act
    return data

def calculate_adjustment_suggestions_v2(clf, current_params_raw, feature_names, shap_values_list, target_threshold, golden_baseline):
    """
    Calculates parameter adjustment suggestions based on golden baseline and SHAP values.
    Accepts current_params_raw (dict) for proper engineered feature re-calculation.
    Accepts shap_values_list (list) as computed by backend.
    """
    # 1. Prepare current data for prediction (add engineered features, align to model's features)
    input_data_with_defaults = model_cache['feature_defaults'].copy() # Use full feature defaults from training
    input_data_with_defaults.update(current_params_raw) # Overwrite with current raw input values
    
    current_df_processed = calculate_engineered_features(input_data_with_defaults)
    
    # Align to model's feature order and convert to DataFrame
    current_df_aligned = pd.DataFrame([current_df_processed], columns=feature_names)
    current_df_aligned = current_df_aligned.fillna(model_cache.get('feature_defaults', {})) # Fill any NaNs after processing

    initial_prob = clf.predict_proba(current_df_aligned)[0, 1]

    if initial_prob >= target_threshold:
        return {}, initial_prob, "样本当前预测已合格，无需调整。"

    adjustments = {}
    message = ""
    adjusted_prob_simulated = initial_prob # Initialize simulated probability

    # --- Strategy 1: Prioritize parameters outside golden baseline ---
    params_out_of_baseline = []
    
    # Check only base features that are also in current_params_raw (i.e., user input)
    base_features_in_raw_input = [f for f in current_params_raw.keys() if f in golden_baseline['mean']]

    for feature in base_features_in_raw_input:
        mean = golden_baseline['mean'].get(feature)
        std = golden_baseline['std'].get(feature)
        current_val = current_params_raw[feature] # Use raw value for checking
        
        if mean is not None and std is not None:
            if std < 1e-6: # Handle cases with zero or very small standard deviation (exact match)
                if not (abs(current_val - mean) < 1e-3): # If not close enough to mean
                     params_out_of_baseline.append({
                        'name': feature, 'current_val': current_val, 'target_val': mean, 
                        'distance': abs(current_val - mean), 'range': f"[{mean:.2f} (± very small)]"
                    })
            else: # Standard deviation is significant
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                if not (lower_bound <= current_val <= upper_bound):
                    params_out_of_baseline.append({
                        'name': feature, 'current_val': current_val, 'target_val': mean, 
                        'distance': abs(current_val - mean), 'range': f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                    })

    if params_out_of_baseline:
        # Sort by distance to mean, fix the one furthest away first (or could pick largest negative SHAP contributor among them)
        param_to_fix = max(params_out_of_baseline, key=lambda x: x['distance'])
        
        # Simulate adjustment: create a new raw params dict with the suggested change
        simulated_params_raw = current_params_raw.copy()
        simulated_params_raw[param_to_fix['name']] = param_to_fix['target_val']
        
        # Recalculate engineered features for simulated raw params
        simulated_df_processed = calculate_engineered_features(simulated_params_raw)
        
        # Align simulated df to model's features and fill NaNs
        simulated_df_aligned = pd.DataFrame([simulated_df_processed], columns=feature_names)
        simulated_df_aligned = simulated_df_aligned.fillna(model_cache.get('feature_defaults', {}))

        adjusted_prob_simulated = clf.predict_proba(simulated_df_aligned)[0, 1]

        adjustments = {
            param_to_fix['name']: {
                'original_value': float(param_to_fix['current_val']),
                'suggested_value': float(param_to_fix['target_val']),
                'change': float(param_to_fix['target_val'] - param_to_fix['current_val']),
                'prob_increase': float(adjusted_prob_simulated - initial_prob) # Corrected prob increase
            }
        }
        message = f"优先修正超出基线的参数 '{PARAM_MAP.get(param_to_fix['name'], param_to_fix['name'])}' (范围: {param_to_fix['range']})。"
        return adjustments, adjusted_prob_simulated, message # Return simulated probability

    # --- Strategy 2: Use SHAP to find the most negatively impacting feature (if all within baseline) ---
    # `shap_values_list` here comes from the frontend, corresponding to `feature_names` passed here
    shap_impact = sorted([(feature_names[i], shap_values_list[i]) for i in range(len(shap_values_list))], key=lambda x: x[1])

    # Iterate through features, looking for one with negative impact (pushing towards NG)
    for feature_name, impact in shap_impact:
        # Only suggest adjusting a base feature (not engineered ones) and with negative impact
        if impact < 0 and feature_name in current_params_raw: 
            original_val = current_params_raw[feature_name]
            best_new_val = original_val
            max_prob = initial_prob
            
            # Determine initial step direction based on SHAP impact and relationship to mean
            step_direction = 0
            # If SHAP is negative, and current value is > mean, try decreasing (step_direction = -1)
            # If SHAP is negative, and current value is < mean, try increasing (step_direction = 1)
            if impact < 0: # This feature pushes towards NG
                if original_val > target_mean: step_direction = -1 
                elif original_val < target_mean: step_direction = 1
                else: step_direction = 1 # If already at mean, try increasing (arbitrary)
            
            step_values_range = np.linspace(0.01, 0.2, 5) # Try 1% to 20% deviation

            for step_magnitude in step_values_range:
                temp_params_raw = current_params_raw.copy()
                
                # Calculate a step based on original value or a fixed amount
                step_amount = original_val * step_magnitude if original_val != 0 else step_magnitude # Proportional or fixed
                
                if step_direction == 1:
                    temp_params_raw[feature_name] = original_val + step_amount
                elif step_direction == -1:
                    temp_params_raw[feature_name] = original_val - step_amount
                else: # Default if no clear direction, just a small change
                    temp_params_raw[feature_name] = original_val + step_amount

                # Recalculate engineered features for simulation
                temp_df_processed = calculate_engineered_features(temp_params_raw)
                temp_df_aligned = pd.DataFrame([temp_df_processed], columns=feature_names)
                temp_df_aligned = temp_df_aligned.fillna(model_cache.get('feature_defaults', {}))

                new_prob = clf.predict_proba(temp_df_aligned)[0, 1]

                if new_prob > max_prob:
                    max_prob = new_prob
                    best_new_val = temp_params_raw[feature_name] 
            
            if max_prob > initial_prob + 1e-6: # If a significant improvement found
                adjustments = {
                    feature_name: {
                        'original_value': float(original_val),
                        'suggested_value': float(best_new_val),
                        'change': float(best_new_val - original_val),
                        'prob_increase': float(max_prob - initial_prob) 
                    }
                }
                message = f"根据AI分析，建议调整负面影响最大的参数 '{PARAM_MAP.get(feature_name, feature_name)}'。调整后，预测合格概率可提升。"
                return adjustments, max_prob, message

    return {}, initial_prob, "未能找到有效的参数调整建议。"

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    # Pass cache data to template for conditional rendering and initial values
    model_ready = model_cache.get('is_ready', False)
    
    displayable_params = []
    if model_ready:
        # These are the base features that users directly input
        # They should correspond to the keys in PARAM_MAP that are NOT product_id or engineered features
        all_param_names = list(PARAM_MAP.keys())
        base_features_in_param_map = [p for p in all_param_names if not any(x in p for x in ['product_id', 'ratio', 'indicator', 'density'])]

        for param_name in base_features_in_param_map:
            displayable_params.append((param_name, PARAM_MAP.get(param_name, param_name)))
        
    return render_template('index.html', model_ready=model_ready, cache=model_cache, displayable_params=displayable_params)


@app.route('/train', methods=['POST'])
def train():
    global model_cache # Explicitly declare global to modify it

    if 'file' not in request.files:
        flash("未选择文件", "error")
        return redirect(url_for('index'))

    file = request.files['file']
    if not file or file.filename == '':
        flash("文件无效或文件名为空", "error")
        return redirect(url_for('index'))

    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
        df.dropna(inplace=True) # Drop rows with any NaN values

        # Robustly handle the quality result column
        # Prioritize 'OK_NG' based on your provided dataset
        if 'OK_NG' in df.columns:
            df['quality_result'] = df['OK_NG'].apply(lambda x: 1 if str(x).strip().upper() == '1' else 0) # Convert 1/0 string or int
        elif 'OK/NG' in df.columns: # Check for common alternative
            df['quality_result'] = df['OK/NG'].apply(lambda x: 1 if str(x).strip().upper() == 'OK' else 0)
        elif 'quality_result' in df.columns: # Check for already named
            df['quality_result'] = df['quality_result'].apply(lambda x: 1 if str(x).strip().upper() == 'OK' or str(x).strip().upper() == '1' else 0)
        else: # Try other common names and convert to 0/1
            found_col = False
            for col_cand in ['Quality', 'Result', 'Status']:
                if col_cand in df.columns:
                    df['quality_result'] = df[col_cand].apply(lambda x: 1 if str(x).strip().upper() == 'OK' or str(x).strip().upper() == '1' else 0)
                    found_col = True
                    break
            if not found_col:
                raise ValueError("CSV文件中未找到预期的质量结果列。请确保列名为 'OK_NG'、'OK/NG'、'quality_result'、'Quality'、'Result' 或 'Status'。")

        # Feature Engineering (applied to the full dataframe for consistency)
        df['pressure_speed_ratio'] = df['F_cut_act'] / df['v_cut_act'].replace(0, np.nan).fillna(1)
        df['stress_indicator'] = df['F_break_peak'] / df['t_glass_meas'].replace(0, np.nan).fillna(1)
        df['energy_density'] = df['F_cut_act'] * df['v_cut_act']
        
        # Replace inf/-inf with NaN (from division by zero) then fill all NaNs with column mean
        df.replace([np.inf, -np.inf], np.nan, inplace=True) 

        # Define all features used by the model (base + engineered)
        features_to_use_in_model = [
            'F_cut_act', 'v_cut_act', 'F_break_peak', 'v_wheel_act', 
            'F_wheel_act', 'P_cool_act', 't_glass_meas',
            'pressure_speed_ratio', 'stress_indicator', 'energy_density'
        ]
        
        # Filter features_to_use to only include columns actually present in the dataframe
        features_to_use_in_model = [f for f in features_to_use_in_model if f in df.columns]

        X = df[features_to_use_in_model].copy()
        y = df["quality_result"] # Use the cleaned 'quality_result' column

        # Fill any remaining NaNs in features with their column means (after feature engineering)
        X = X.fillna(X.mean())

        count_ok = y.value_counts().get(1, 0)
        count_ng = y.value_counts().get(0, 0)
        scale_pos_weight_value = count_ok / count_ng if count_ng > 0 else 1

        clf = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, gamma=0.1, random_state=42, use_label_encoder=False, scale_pos_weight=scale_pos_weight_value, eval_metric='logloss')
        clf.fit(X, y)

        probs_ok = clf.predict_proba(X)[:, 1]
        best_f1_macro, best_thresh = 0.0, 0.5
        for t in np.arange(0.01, 1.0, 0.01):
            f1 = f1_score(y, (probs_ok >= t).astype(int), average='macro', zero_division=0)
            if f1 > best_f1_macro:
                best_f1_macro, best_thresh = f1, t

        fig_importance = plt.figure()
        xgb.plot_importance(clf, max_num_features=10, ax=plt.gca(), importance_type='gain') # Use 'gain'
        plt.title('特征重要性 (Gain)', fontsize=16) # Chinese title
        plt.tight_layout()
        
        y_pred_final = (probs_ok >= best_thresh).astype(int)
        metrics = {
            'accuracy': accuracy_score(y, y_pred_final),
            'recall_ng': recall_score(y, y_pred_final, pos_label=0, zero_division=0),
            'precision_ng': precision_score(y, y_pred_final, pos_label=0, zero_division=0),
            'f1_ng': f1_score(y, y_pred_final, pos_label=0, zero_division=0)
        }

        # Calculate golden baseline for ALL features the model uses
        ok_df = df[df['quality_result'] == 1].copy() # Ensure this is a copy to avoid SettingWithCopyWarning
        golden_mean = ok_df[features_to_use_in_model].mean().to_dict()
        golden_std = ok_df[features_to_use_in_model].std().to_dict()

        model_cache = {
            'model': clf,
            'features': features_to_use_in_model, # Store the exact list of features used by the model
            'best_threshold': best_thresh,
            'feature_defaults': X.mean().to_dict(), # Defaults for all model features
            'golden_baseline': {'mean': golden_mean, 'std': golden_std},
            'knowledge_base': ok_df.copy(), # Store OK samples. Includes 'product_id' if present in original df.
            'feature_plot': fig_to_base64(fig_importance),
            'metrics': metrics,
            'filename': file.filename,
            'is_ready': True, # Flag indicating model is trained
            'param_map': PARAM_MAP # Store param map for display
        }
        flash(f"文件 '{file.filename}' 上传成功，模型已完成训练！", "success")

    except Exception as e:
        flash(f"处理文件时出错: {e}", "error")
        logging.error(f"Error during training: {e}", exc_info=True) # Log full traceback

    return redirect(url_for('index'))

@app.route('/api/process_monitor', methods=['POST'])
def process_monitor():
    if not model_cache.get('is_ready'):
        return jsonify({'error': '请先上传数据并训练模型'}), 400

    data_raw = request.get_json() # Raw input from frontend

    # Use feature defaults to ensure all required base features are present for engineered feature calculation
    input_data_for_processing = model_cache['feature_defaults'].copy()
    input_data_for_processing.update(data_raw) 
    
    # Calculate engineered features for the current monitor sample
    current_monitor_params_processed = calculate_engineered_features(input_data_for_processing)

    warnings = []
    gb = model_cache['golden_baseline']

    # Check only base features against their baselines
    for p, v in data_raw.items(): 
        if p in gb['mean']: # Ensure golden baseline has data for this parameter
            m, s = gb['mean'][p], gb['std'][p]
            if s is None or s < 1e-6: # Handle cases with zero or near-zero standard deviation
                if not (abs(v - m) < 1e-3): # If not close enough to mean
                    warnings.append({
                        "param": p,
                        "current": v,
                        "range": f"[{m:.2f} (± very small)]" # Indicate it's a tight range
                    })
            else:
                lower_bound = m - 3 * s
                upper_bound = m + 3 * s
                if not (lower_bound <= v <= upper_bound):
                    warnings.append({
                        "param": p,
                        "current": v,
                        "range": f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                    })
    
    # Also check engineered features if they are part of the golden baseline
    for eng_feat in ['pressure_speed_ratio', 'stress_indicator', 'energy_density']:
        if eng_feat in gb['mean'] and eng_feat in current_monitor_params_processed:
            m, s = gb['mean'][eng_feat], gb['std'][eng_feat]
            v = current_monitor_params_processed[eng_feat]
            if s is None or s < 1e-6:
                if not (abs(v - m) < 1e-3):
                    warnings.append({
                        "param": eng_feat,
                        "current": v,
                        "range": f"[{m:.2f} (± very small)]"
                    })
            else:
                lower_bound = m - 3 * s
                upper_bound = m + 3 * s
                if not (lower_bound <= v <= upper_bound):
                    warnings.append({
                        "param": eng_feat,
                        "current": v,
                        "range": f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                    })

    if warnings:
        # Convert warning params to their Chinese names for frontend display
        display_warnings = []
        for w in warnings:
            display_warnings.append(f"参数 '{PARAM_MAP.get(w['param'], w['param'])}' ({w['current']:.2f}) 超出黄金基线范围 {w['range']}")
        return jsonify({'status': 'abnormal', 'messages': display_warnings})
    else:
        return jsonify({'status': 'ok', 'messages': ['所有参数均在黄金基线范围内']})

@app.route('/api/recommend_params', methods=['POST'])
def recommend_params():
    if not model_cache.get('is_ready'):
        return jsonify({'error': '请先上传数据并训练模型'}), 400

    product_id = request.get_json().get('product_id')
    kb = model_cache['knowledge_base'] # This is the df of OK samples, includes 'product_id' if exists in original data
    
    # Filter knowledge base by product_id if 'product_id' column exists in kb
    if 'product_id' in kb.columns:
        cases = kb[kb['product_id'] == product_id]
    else:
        cases = pd.DataFrame() # No product_id column to filter by

    # List of base features that are actual physical parameters
    physical_features = [
        'F_cut_act', 'v_cut_act', 'F_break_peak', 'v_wheel_act', 
        'F_wheel_act', 'P_cool_act', 't_glass_meas'
    ]
    
    recommended_params_data = {}
    message = ""

    if not cases.empty:
        # Calculate mean of physical parameters for this specific product_id
        for feature in physical_features:
            if feature in cases.columns:
                recommended_params_data[feature] = cases[feature].mean()
        message = f"为产品'{product_id}'生成了基于历史合格数据的推荐参数。"
    else:
        # Fallback: calculate mean of all OK physical parameters from the entire dataset
        for feature in physical_features:
            if feature in kb.columns: # kb is the full set of OK samples
                recommended_params_data[feature] = kb[feature].mean()
        message = f"知识库无产品'{product_id}'的专属案例，返回通用推荐参数 (所有合格产品的平均值)。"

    return jsonify({
        'product_id': product_id,
        'recommended_params': {k: float(v) for k, v in recommended_params_data.items()}, # Ensure float for JSON
        'message': message,
        'param_map': PARAM_MAP
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model_cache.get('is_ready'):
        return jsonify({'error': '请先上传数据并训练模型'}), 400

    data_raw = request.get_json() # Raw input from frontend

    try:
        features = model_cache['features'] # Features the model expects (base + engineered)
        
        # Pre-process input data (add defaults, calculate engineered features)
        input_data_for_processing = model_cache['feature_defaults'].copy()
        input_data_for_processing.update(data_raw)
        input_vector_processed = calculate_engineered_features(input_data_for_processing)
        
        # Create DataFrame for prediction, aligned to model's feature order
        input_vector_df = pd.DataFrame([input_vector_processed], columns=features)
        
        # Handle any NaNs that might still be present after engineering (e.g. from missing input)
        input_vector_df = input_vector_df.fillna(model_cache['feature_defaults'])


        # Check against golden baseline for warnings (only for base features provided in raw input)
        baseline_warnings = []
        gb = model_cache['golden_baseline']
        
        # Check only the original input features against baseline
        base_feature_names_from_raw = [f for f in data_raw.keys() if f in gb['mean']]

        for feature in base_feature_names_from_raw:
            val = input_vector_df[feature].iloc[0] # Get the value from the processed DF
            m, s = gb['mean'][feature], gb['std'][feature]
            if s is None or s < 1e-6: # Handle zero std
                if not (abs(val - m) < 1e-3):
                    baseline_warnings.append(f"参数 '{PARAM_MAP.get(feature, feature)}' ({val:.2f}) 超出黄金工艺基线范围 [{m:.2f} (± very small)]")
            else:
                lower_bound = m - 3 * s
                upper_bound = m + 3 * s
                if not (lower_bound <= val <= upper_bound):
                    baseline_warnings.append(f"参数 '{PARAM_MAP.get(feature, feature)}' ({val:.2f}) 超出黄金工艺基线范围 [{lower_bound:.2f}, {upper_bound:.2f}]")

        model = model_cache['model']
        explainer = shap.TreeExplainer(model)
        
        # SHAP values for the processed input_vector_df
        shap_values_obj = explainer(input_vector_df)
        shap_values_list = shap_values_obj.values[0].tolist() # Convert to list for JSON

        prob_ok = model.predict_proba(input_vector_df)[0, 1]
        model_says_ng = bool(prob_ok < model_cache['best_threshold'])

        final_status = "ok"
        verdict_reason = []
        show_suggestion_button = False # Flag for frontend

        if model_says_ng:
            final_status = "ng"
            verdict_reason.append(f"AI模型预测合格概率仅为 {prob_ok:.2%}，低于阈值 {model_cache['best_threshold']:.2%}。")
            verdict_reason.extend(baseline_warnings) # Add any baseline warnings
            show_suggestion_button = True # Enable suggestion for NG
        elif baseline_warnings:
            final_status = "warning"
            verdict_reason.extend(baseline_warnings)
            verdict_reason.append("虽然AI预测合格，但存在过程异常，有潜在质量风险。")
            show_suggestion_button = False # No suggestion for warning, just alert
        else:
            final_status = "ok"
            verdict_reason.append(f"所有参数均在基线内，且AI模型预测合格 (合格概率: {prob_ok:.2%})。")
            show_suggestion_button = False


        # Generate SHAP waterfall plot
        fig = plt.figure(figsize=(10, 6))
        # shap.plots.waterfall expects Explanation object
        shap.plots.waterfall(shap.Explanation(values=shap_values_obj.values[0], 
                                               base_values=explainer.expected_value, 
                                               data=input_vector_df.iloc[0].values, # Pass original data for SHAP plot
                                               feature_names=features), 
                             show=False, max_display=10)
        plt.title('缺陷根因分析 (SHAP 瀑布图)', fontsize=16)
        plt.tight_layout()
        waterfall_plot_base64 = fig_to_base64(fig)


        return jsonify({
            'status': final_status,
            'prob': float(prob_ok),
            'threshold': float(model_cache['best_threshold']),
            'waterfall_plot': waterfall_plot_base64,
            'input_data': data_raw, # Send raw input back for adjustment calculations
            'shap_values': shap_values_list, # Send SHAP values list for adjustment calculations
            'verdict_reason': verdict_reason,
            'show_suggestion_button': show_suggestion_button
        })

    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/adjust', methods=['POST'])
def adjust():
    if not model_cache.get('is_ready'):
        return jsonify({'error': '请先上传数据并训练模型'}), 400

    data = request.get_json()
    try:
        clf = model_cache['model']
        features = model_cache['features']
        threshold = model_cache['best_threshold']
        gb = model_cache['golden_baseline']

        # Get raw input data and SHAP values from the request
        input_data_raw = data.get('input_data') # This is the raw dict that came from predict_api
        shap_values_list = data.get('shap_values') # This is the list from predict_api

        if not input_data_raw or not shap_values_list:
            raise ValueError("Input data or SHAP values missing for adjustment.")

        # Call the helper function for adjustment logic
        adjustments, final_prob, message = calculate_adjustment_suggestions_v2(
            clf, input_data_raw, features, shap_values_list, threshold, gb
        )
        
        # Convert adjustments to a JSON-friendly format, ensure float types
        json_adjustments = {}
        for k, v in adjustments.items():
            json_adjustments[k] = {inner_k: float(inner_v) if isinstance(inner_v, (int, float, np.number)) else inner_v for inner_k, inner_v in v.items()}


        return jsonify({
            'adjustments': json_adjustments,
            'final_prob_after_adjustment': float(final_prob),
            'message': message,
            'param_map': PARAM_MAP
        })

    except Exception as e:
        logging.error(f"参数调整时出错: {e}", exc_info=True)
        return jsonify({'error': f'参数调整失败: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) # Default to 5000 for local development
    app.run(host='0.0.0.0', port=port, debug=True) # Run in debug mode locally

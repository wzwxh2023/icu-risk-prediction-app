import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import requests
import json

# --- Configuration Area ---
# These are used for local testing. When deploying to Streamlit Community Cloud,
# st.secrets should be used to manage keys.
# You need to replace these with your actual API information.
SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/embeddings") # SiliconFlow API Endpoint
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "YOUR_SILICONFLOW_API_KEY_HERE") # Your SiliconFlow API Key

TRANSLATE_API_URL = os.getenv("TRANSLATE_API_URL", "YOUR_TRANSLATE_API_ENDPOINT_HERE") # e.g., DeepL, Google Translate
TRANSLATE_API_KEY = os.getenv("TRANSLATE_API_KEY", "YOUR_TRANSLATE_API_KEY_HERE")
BGE_MODEL_NAME = "bge-large-zh-v1.5" # Ensure this is the correct model name supported by SiliconFlow

# --- Model and File Paths ---
# Assume model files are in the same directory as streamlit_app.py or in a subdirectory model_files/
# On Streamlit Community Cloud, you need to upload these files to your GitHub repository.
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_files') # Or just '.' if files are in the root directory
MODEL_PATH = os.path.join(MODEL_DIR, 'lightgbm_icu_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, 'feature_list.txt')

# --- Global Loading of Model and Auxiliary Files (executes once at app startup) ---
@st.cache_resource # Use Streamlit's caching mechanism to avoid reloading on every interaction
def load_resources():
    """Load model, scaler, and feature list."""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Error: Model file not found at {MODEL_PATH}. Please ensure the file is uploaded to the correct location in your GitHub repository.")
            return None, None, []
        if not os.path.exists(SCALER_PATH):
            st.error(f"Error: Scaler file not found at {SCALER_PATH}.")
            return None, None, []
        if not os.path.exists(FEATURE_LIST_PATH):
            st.error(f"Error: Feature list file not found at {FEATURE_LIST_PATH}.")
            return None, None, []

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURE_LIST_PATH, 'r', encoding='utf-8') as f: # Specify encoding
            feature_list = [line.strip() for line in f.readlines() if line.strip()]
        
        st.success("Model and auxiliary files loaded successfully!")
        return model, scaler, feature_list
    except FileNotFoundError as fnf_error:
        st.error(f"File loading error: {fnf_error}. Please check file paths and GitHub repository.")
        return None, None, []
    except Exception as e:
        st.error(f"An unknown error occurred while loading resources: {str(e)}")
        return None, None, []

model, scaler, feature_list_from_file = load_resources()

# --- API Call Functions ---
def translate_text_if_needed(text: str, target_lang: str = "zh") -> str:
    """ (Optional) Detect text language and call translation API if not the target language (Chinese). """
    # For simplification, assume you will manually decide if translation is needed or always attempt it.
    # In a real application, you might want to integrate a library like langdetect.
    # from langdetect import detect, LangDetectException
    # try:
    #     detected_lang = detect(text)
    #     if detected_lang in ['zh-cn', 'zh-tw']: return text # If already Chinese, return
    # except LangDetectException:
    #     st.warning(f"Could not detect language for text '{text[:30]}...', will proceed without translation.")
    #     return text

    if not TRANSLATE_API_URL or not TRANSLATE_API_KEY or TRANSLATE_API_KEY == "YOUR_TRANSLATE_API_KEY_HERE":
        st.warning("Translation API not configured. Proceeding with original text.")
        return text

    headers = {
        # Adjust authentication based on your chosen translation API
        "Authorization": f"Bearer {TRANSLATE_API_KEY}",
        "Content-Type": "application/json",
    }
    # Payload needs to be constructed according to your chosen translation API's documentation
    # Example payload (assuming a Google Translate-like API)
    payload = {"q": text, "target": target_lang} # "source": "auto" usually auto-detects

    try:
        response = requests.post(TRANSLATE_API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        # Parsing the response also depends on the API documentation
        translated_text = response.json().get("data", {}).get("translations", [{}])[0].get("translatedText", text)
        if translated_text != text:
            st.info(f"Text translated: '{text[:30]}...' -> '{translated_text[:30]}...'")
        return translated_text
    except requests.exceptions.RequestException as e:
        st.error(f"Translation API call failed: {e}")
        return text # Return original text if translation fails

def get_bge_embedding(text: str, api_key: str, api_url: str, model_name: str) -> list[float]:
    """Call SiliconFlow BGE API to get text embedding."""
    if not api_url or not api_key or api_key == "YOUR_SILICONFLOW_API_KEY_HERE":
        st.error("SiliconFlow Embedding API not configured. Please set it in the sidebar or via Secrets.")
        return [] # Return empty list or zero vector of appropriate length

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"input": [text], "model": model_name}

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        embedding_data = response.json().get("data")
        if not embedding_data or not isinstance(embedding_data, list) or not embedding_data[0].get("embedding"):
            st.error(f"Valid embedding not found in SiliconFlow API response for text: {text[:30]}...")
            return [] # Or return a zero vector of a specific length, e.g., [0.0] * 1024
        return embedding_data[0]["embedding"]
    except requests.exceptions.Timeout:
        st.error(f"SiliconFlow API request timed out for text: {text[:30]}...")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"SiliconFlow API request failed: {e} for text: {text[:30]}...")
        return []
    except (KeyError, IndexError, TypeError, ValueError) as e:
        st.error(f"Failed to parse SiliconFlow API response: {e}, for text: {text[:30]}...")
        return []

# --- Feature Processing Function (adapted from your predict.py) ---
def process_streamlit_features(data_input: dict, text_inputs: dict, sf_api_key: str, sf_api_url: str, sf_model_name: str):
    """Extract and process features from Streamlit input, including calling Embedding API."""
    if not scaler or not feature_list_from_file:
        st.error("Scaler or feature list not loaded, cannot process features.")
        return None

    # 1. Process text features: Translation (optional) and get Embeddings
    embeddings_map = {}
    text_feature_keys = {
        "diagnosis": "diagnosis_text", # key in text_inputs : prefix for embedding columns
        "history": "history_text",
        "exam_critical_value": "exam_critical_value_text",
        "lab_critical_value": "lab_critical_value_text"
    }

    for emb_name_prefix, text_key in text_feature_keys.items():
        raw_text = text_inputs.get(text_key, "")
        if raw_text and raw_text.strip() and raw_text.lower() != "n/a" and raw_text.lower() != "none" and raw_text.lower() != "not applicable" and raw_text.lower() != "未见危急值": # More robust check for empty/default
            # translated_text = translate_text_if_needed(raw_text) # If translation is enabled
            translated_text = raw_text # Temporarily disable translation, focus on Embedding
            embedding_vector = get_bge_embedding(translated_text, sf_api_key, sf_api_url, sf_model_name)
            if embedding_vector: # Ensure embedding is obtained
                 embeddings_map[emb_name_prefix] = embedding_vector
            else: # If fetching fails, use zero vector or show error
                st.warning(f"Failed to get text embedding for {emb_name_prefix}, using zero vector.")
                # Assume your model expects embedding dimension of 1024 (BGE-large is typically 1024)
                # You need to adjust this based on the actual embedding dimension
                embeddings_map[emb_name_prefix] = [0.0] * 1024 # Confirm this dimension
        else:
            embeddings_map[emb_name_prefix] = [0.0] * 1024 # For empty text or "Not applicable"

    # 2. Process structured features (based on your predict.py logic)
    raw_physiological_keys = ['age', 'bmi', 'pulse', 'tempreture', 'sbp', 'res']
    mews_total_keys = ['mews_total']
    
    categorical_feature_keys = [
        'gender', 'admission_unit', 'surgey', 'intervention',
        'exam_critical_flag', 'lab_critical_flag',
        'o2', 'mews_aware'
    ]
    
    numeric_feature_keys = raw_physiological_keys + mews_total_keys
    
    numeric_data_dict = {}
    for feature in numeric_feature_keys:
        numeric_data_dict[feature] = data_input.get(feature, 0) 
    
    numeric_df = pd.DataFrame([numeric_data_dict])
    try:
        numeric_scaled_array = scaler.transform(numeric_df)
        numeric_scaled_df = pd.DataFrame(numeric_scaled_array, columns=numeric_feature_keys)
    except Exception as e:
        st.error(f"Numerical feature scaling error: {e}. Please check input data and Scaler.")
        st.error(f"Expected Scaler features: {scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else 'N/A'}")
        st.error(f"Provided numerical features: {list(numeric_df.columns)}")
        return None

    categorical_data_dict = {}
    for feature in categorical_feature_keys:
        categorical_data_dict[feature] = data_input.get(feature, '') # Assume empty string is a reasonable default
        
    categorical_df_raw = pd.DataFrame([categorical_data_dict])
    try:
        # One-hot encode - Note: This needs to be strictly consistent with training time
        # A more robust method is to use a OneHotEncoder saved during training
        categorical_df_dummies = pd.get_dummies(categorical_df_raw, columns=categorical_feature_keys, dummy_na=False)
    except Exception as e:
        st.error(f"Categorical feature one-hot encoding error: {e}")
        return None

    # 3. Combine all features
    # Add Embeddings to numeric_scaled_df (as they are numerical)
    # Your feature_list.txt needs to contain column names for these embedding features
    # e.g., diagnosis_emb_0, diagnosis_emb_1, ..., history_emb_0, ...
    for emb_name_prefix, emb_values in embeddings_map.items():
        for i, val in enumerate(emb_values):
            column_name = f"{emb_name_prefix}_emb_{i}" # Ensure this matches feature_list.txt
            numeric_scaled_df[column_name] = val # Add as new column

    # Combine numerical (with embeddings) and categorical features
    # Reset index to ensure correct concatenation
    all_features_df = pd.concat([numeric_scaled_df.reset_index(drop=True), 
                                 categorical_df_dummies.reset_index(drop=True)], 
                                 axis=1)
    
    # 4. Ensure feature order and completeness match training time (using feature_list_from_file)
    final_features_df = pd.DataFrame(columns=feature_list_from_file)
    for col in feature_list_from_file:
        if col in all_features_df.columns:
            final_features_df[col] = all_features_df[col]
        else:
            # If a feature is in the list but not generated (e.g., a one-hot encoded category didn't appear, or embedding failed)
            # fill with 0. This is standard for missing categories in one-hot encoding.
            final_features_df[col] = 0 
    
    # Ensure all columns are numerical, LightGBM might need float32
    try:
        final_features_df = final_features_df.astype(np.float32)
    except Exception as e:
        st.error(f"Error converting final features to float32: {e}")
        for col in final_features_df.columns:
            if final_features_df[col].dtype != np.float32: # Check which column causes issue
                st.write(f"Column '{col}' has dtype {final_features_df[col].dtype}, failed to convert.")
        return None

    return final_features_df


# --- Streamlit UI Construction ---
st.set_page_config(page_title="ICU Risk Prediction System", layout="wide")
st.title("48-Hour ICU Transfer Risk Prediction for General Ward Patients")

# Load API keys from Secrets (recommended for deployment)
# Or allow user input in sidebar (for local testing)
# Note: When deploying to Streamlit Community Cloud, use st.secrets["SILICONFLOW_API_KEY"]
st.sidebar.header("API Key Configuration (Optional)")
st.sidebar.info("When deploying to Streamlit Cloud, please use the platform's Secrets management feature to configure API keys.")
input_sf_api_key = st.sidebar.text_input("SiliconFlow API Key", value=SILICONFLOW_API_KEY, type="password")
input_sf_api_url = st.sidebar.text_input("SiliconFlow API URL", value=SILICONFLOW_API_URL)
input_sf_model_name = st.sidebar.text_input("SiliconFlow Model Name", value=BGE_MODEL_NAME)

# input_translate_api_key = st.sidebar.text_input("Translation API Key (Optional)", value=TRANSLATE_API_KEY, type="password")
# input_translate_api_url = st.sidebar.text_input("Translation API URL (Optional)", value=TRANSLATE_API_URL)


if not model or not scaler or not feature_list_from_file:
    st.error("Core resources failed to load. The application cannot run. Please check backend logs and file paths.")
else:
    st.markdown("Please enter the patient's relevant information for prediction.")

    with st.form("prediction_form"):
        st.subheader("Physiological Indicators and Basic Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=65, step=1)
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=24.0, step=0.1, format="%.1f")
            gender_options = {"Male": 1, "Female": 0} # Assuming 1 Male, 0 Female, or based on your encoding
            gender_display = st.selectbox("Gender", options=list(gender_options.keys()))
            gender = gender_options[gender_display]

        with col2:
            pulse = st.number_input("Pulse (beats/min)", min_value=30, max_value=250, value=80, step=1)
            tempreture = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1, format="%.1f")
            admission_unit_options = {"Emergency": "Emergency", "General Ward": "GeneralWard", "Other": "Other"} # Example
            admission_unit_display = st.selectbox("Admission Unit", options=list(admission_unit_options.keys()))
            admission_unit = admission_unit_options[admission_unit_display] # This will pass the value e.g. "Emergency"

        with col3:
            sbp = st.number_input("Systolic Blood Pressure (SBP, mmHg)", min_value=50, max_value=300, value=120, step=1)
            res = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=60, value=20, step=1)
            o2_options = {"Yes": 1, "No": 0} # Oxygen therapy
            o2_display = st.selectbox("Oxygen Therapy (O2)", options=list(o2_options.keys()))
            o2 = o2_options[o2_display]

        st.subheader("MEWS Score Related")
        col_mews1, col_mews2 = st.columns(2)
        with col_mews1:
            mews_total = st.number_input("MEWS Total Score", min_value=0, max_value=20, value=2, step=1) # MEWS score usually has an upper limit
        with col_mews2:
            # Ensure these values ('Alert', 'Voice', etc.) match what your model expects after one-hot encoding
            mews_aware_options = {"Alert": "Alert", "Responds to Voice": "Voice", "Responds to Pain": "Pain", "Unresponsive": "Unresponsive"} # MEWS awareness level
            mews_aware_display = st.selectbox("MEWS Awareness Level", options=list(mews_aware_options.keys()))
            mews_aware = mews_aware_options[mews_aware_display]

        st.subheader("Other Categorical Information")
        col_cat1, col_cat2, col_cat3, col_cat4 = st.columns(4)
        with col_cat1:
            surgey_options = {"Yes": 1, "No": 0}
            surgey_display = st.selectbox("Surgery", options=list(surgey_options.keys()))
            surgey = surgey_options[surgey_display]
        with col_cat2:
            intervention_options = {"Yes": 1, "No": 0}
            intervention_display = st.selectbox("Intervention", options=list(intervention_options.keys()))
            intervention = intervention_options[intervention_display]
        with col_cat3:
            exam_critical_flag_options = {"Yes": 1, "No": 0}
            exam_critical_flag_display = st.selectbox("Exam Critical Value Flag", options=list(exam_critical_flag_options.keys()))
            exam_critical_flag = exam_critical_flag_options[exam_critical_flag_display]
        with col_cat4:
            lab_critical_flag_options = {"Yes": 1, "No": 0}
            lab_critical_flag_display = st.selectbox("Lab Critical Value Flag", options=list(lab_critical_flag_options.keys()))
            lab_critical_flag = lab_critical_flag_options[lab_critical_flag_display]

        st.subheader("Text Information (Please fill in detail)")
        diagnosis_text = st.text_area("Main Diagnosis", placeholder="e.g., Acute myocardial infarction, Community-acquired pneumonia", height=100)
        history_text = st.text_area("Past Medical History", placeholder="e.g., Hypertension for 10 years, Diabetes for 5 years", height=100)
        exam_critical_value_text = st.text_area("Specific Exam Critical Values", placeholder="e.g., Cardiac tamponade (If none, enter 'N/A' or 'Not applicable')", height=100)
        lab_critical_value_text = st.text_area("Specific Lab Critical Values", placeholder="e.g., Troponin 2.5 (If none, enter 'N/A' or 'Not applicable')", height=100)

        submitted = st.form_submit_button("Start Prediction", type="primary")

        if submitted:
            if not model or not scaler or not feature_list_from_file:
                st.error("Model or necessary files not loaded. Prediction cannot be performed.")
            elif not input_sf_api_key or input_sf_api_key == "YOUR_SILICONFLOW_API_KEY_HERE":
                st.error("Please enter a valid SiliconFlow API Key.")
            else:
                with st.spinner("Processing data and making prediction..."):
                    # Prepare structured data payload
                    structured_data_payload = {
                        'age': age, 'bmi': bmi, 'pulse': pulse, 'tempreture': tempreture, 
                        'sbp': sbp, 'res': res, 'mews_total': mews_total,
                        'gender': gender, 'admission_unit': admission_unit, 'surgey': surgey, 
                        'intervention': intervention, 'exam_critical_flag': exam_critical_flag, 
                        'lab_critical_flag': lab_critical_flag, 'o2': o2, 'mews_aware': mews_aware
                    }
                    # Prepare text data payload
                    text_data_payload = {
                        "diagnosis_text": diagnosis_text,
                        "history_text": history_text,
                        "exam_critical_value_text": exam_critical_value_text,
                        "lab_critical_value_text": lab_critical_value_text
                    }

                    # Process features
                    current_sf_api_key = input_sf_api_key
                    current_sf_api_url = input_sf_api_url
                    current_sf_model_name = input_sf_model_name
                    
                    # When deployed to Streamlit Cloud, get from secrets
                    # Check if st.secrets is available (it is on Streamlit Cloud)
                    if hasattr(st, 'secrets') and "SILICONFLOW_API_KEY" in st.secrets:
                        current_sf_api_key = st.secrets.get("SILICONFLOW_API_KEY", input_sf_api_key)
                        current_sf_api_url = st.secrets.get("SILICONFLOW_API_URL", input_sf_api_url)
                        # TRANSLATE_API_KEY = st.secrets.get("TRANSLATE_API_KEY", input_translate_api_key)
                        # TRANSLATE_API_URL = st.secrets.get("TRANSLATE_API_URL", input_translate_api_url)

                    final_feature_vector_df = process_streamlit_features(
                        structured_data_payload, 
                        text_data_payload,
                        current_sf_api_key,
                        current_sf_api_url,
                        current_sf_model_name
                    )

                    if final_feature_vector_df is not None and not final_feature_vector_df.empty:
                        st.write("---")
                        st.subheader("Prediction Result")
                        try:
                            # Predict probability
                            prediction_proba_array = model.predict_proba(final_feature_vector_df)
                            # We are usually interested in the probability of class 1 (transfer to ICU)
                            icu_transfer_probability = prediction_proba_array[0][1] 

                            # Determine if ICU transfer is needed based on threshold (0.5 is common, adjust as needed)
                            threshold = 0.5 
                            icu_needed = icu_transfer_probability >= threshold

                            if icu_needed:
                                st.error(f"High Risk: ICU transfer recommended. Predicted Probability: {icu_transfer_probability:.2%}")
                            else:
                                st.success(f"Low Risk: ICU transfer not immediately indicated. Predicted Probability: {icu_transfer_probability:.2%}")
                            
                            st.caption(f"Prediction Model Used: LightGBM, Number of Features: {len(final_feature_vector_df.columns)}")
                            
                            # Optional: Display some processed features for debugging
                            # with st.expander("View Processed Features (Partial)"):
                            #    st.dataframe(final_feature_vector_df.head())

                        except Exception as e:
                            st.error(f"An error occurred during prediction: {str(e)}")
                            st.error("Please check if the input data is reasonable and if the model is compatible with the current features.")
                    else:
                        st.error("Feature processing failed. Prediction cannot be made. Please check error messages.")

st.sidebar.markdown("---")
st.sidebar.markdown("This application uses a LightGBM model to predict ICU transfer risk.")
st.sidebar.markdown("Text features are embedded via the SiliconFlow API.")
st.sidebar.markdown("**Important Note:** This prediction result is for clinical reference only and cannot replace the judgment of a professional physician.")

# --- Running Instructions ---
# 1. Ensure your model file (lightgbm_icu_model.pkl), scaler (scaler.pkl), 
#    and feature list (feature_list.txt) are in the `model_files` folder (or your specified path).
# 2. Create a `requirements.txt` file with all dependencies.
# 3. (Local run) Set environment variables SILICONFLOW_API_KEY etc., or input via sidebar.
#    `streamlit run streamlit_app.py`
# 4. (Deploy to Streamlit Community Cloud)
#    - Upload this script, `model_files` folder, and `requirements.txt` to a GitHub repository.
#    - Connect your repository in Streamlit Cloud.
#    - Configure `SILICONFLOW_API_KEY`, `SILICONFLOW_API_URL`, etc., in "Settings" -> "Secrets".

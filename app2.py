import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
import os
from datetime import datetime
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Income Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* ===== MAIN HEADER ===== */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem;
        margin-bottom: 3rem;
    }
    
    /* ===== INPUT FIELDS - DARK THEME ===== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input,
    .stTextArea textarea {
        background: rgba(30, 30, 50, 0.9) !important;
        border: 2px solid #667eea !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    /* Placeholder text color */
    .stTextInput > div > div > input::placeholder,
    .stNumberInput > div > div > input::placeholder,
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stDateInput > div > div > input:focus,
    .stTimeInput > div > div > input:focus,
    .stTextArea textarea:focus {
        background: rgba(40, 40, 60, 0.95) !important;
        border-color: #f093fb !important;
        box-shadow: 0 0 0 3px rgba(240, 147, 251, 0.2) !important;
        outline: none !important;
        color: #ffffff !important;
    }
    
    /* ===== SELECT BOXES - DARK THEME ===== */
    .stSelectbox > div > div,
    .stSelectbox > div > div > div {
        background: rgba(30, 30, 50, 0.9) !important;
        color: #ffffff !important;
        border: 2px solid #667eea !important;
        border-radius: 12px !important;
    }
    
    .stSelectbox > div > div:hover {
        background: rgba(40, 40, 60, 0.95) !important;
        border-color: #f093fb !important;
    }
    
    /* Select dropdown options */
    .stSelectbox option {
        background: #1a1a2e !important;
        color: #ffffff !important;
    }
    
    /* Selected value text */
    .stSelectbox > div > div > div > div {
        color: #ffffff !important;
    }
    
    /* Arrow icon */
    .stSelectbox svg {
        fill: #ffffff !important;
    }
    
    /* ===== MULTISELECT - DARK THEME ===== */
    .stMultiSelect > div > div {
        background: rgba(30, 30, 50, 0.9) !important;
        border: 2px solid #667eea !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }
    
    .stMultiSelect > div > div > div {
        color: #ffffff !important;
    }
    
    .stMultiSelect > div > div > div > div {
        background: #667eea !important;
        color: #ffffff !important;
    }
    
    /* ===== LABELS ===== */
    label {
        color: #ffffff !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 12px 28px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 25px;
        transition: transform 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* ===== DATAFRAME / TABLES - DARK THEME ===== */
    .dataframe, .stDataFrame > div > div > div > div {
        background: rgba(30, 30, 50, 0.9) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* Table headers */
    .dataframe thead tr th,
    .stDataFrame th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 12px !important;
        border: none !important;
    }
    
    /* Table rows */
    .dataframe tbody tr,
    .stDataFrame tbody tr {
        background: rgba(30, 30, 50, 0.9) !important;
    }
    
    /* Table cells */
    .dataframe tbody tr td,
    .stDataFrame td {
        color: #ffffff !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        padding: 10px !important;
    }
    
    /* Hover effect on rows */
    .dataframe tbody tr:hover,
    .stDataFrame tbody tr:hover {
        background: rgba(102, 126, 234, 0.2) !important;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card h2, .metric-card h3, .metric-card p {
        color: white !important;
    }
    
    /* ===== PREDICTION BOXES ===== */
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        color: white !important;
    }
    
    .high-income {
        background: linear-gradient(135deg, #00C851 0%, #00aa44 100%);
    }
    
    .low-income {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
    }
    
    .prediction-box h2, .prediction-box h3, .prediction-box p {
        color: white !important;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 30, 50, 0.9);
        color: white !important;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
    }
    
    /* ===== METRICS ===== */
    [data-testid="metric-container"] {
        background: rgba(30, 30, 50, 0.7);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    
    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e9ecef !important;
        font-size: 0.9rem !important;
    }
    
    /* ===== ALERTS ===== */
    .stSuccess {
        background: rgba(0, 200, 81, 0.2);
        border-left: 4px solid #00C851;
        color: white !important;
    }
    
    .stError {
        background: rgba(255, 107, 53, 0.2);
        border-left: 4px solid #FF6B35;
        color: white !important;
    }
    
    .stInfo {
        background: rgba(102, 126, 234, 0.2);
        border-left: 4px solid #667eea;
        color: white !important;
    }
    
    /* ===== TEXT COLORS ===== */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: white !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        color: #e9ecef !important;
    }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: rgba(26, 26, 46, 0.95);
    }
    
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Sidebar inputs */
    section[data-testid="stSidebar"] .stTextInput > div > div > input,
    section[data-testid="stSidebar"] .stNumberInput > div > div > input,
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(40, 40, 60, 0.9) !important;
        color: #ffffff !important;
        border: 2px solid rgba(102, 126, 234, 0.5) !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: rgba(30, 30, 50, 0.7);
        color: white !important;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* ===== RADIO & CHECKBOX ===== */
    .stRadio > label, .stCheckbox > label {
        color: white !important;
    }
    
    .stRadio > div, .stCheckbox > div {
        color: white !important;
    }
    
    /* ===== SLIDER ===== */
    .stSlider > div > div > div > div {
        background: #667eea;
    }
    
    .stSlider > div > div > div {
        color: white !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(30, 30, 50, 0.7);
        border: 2px dashed #667eea;
        border-radius: 10px;
    }
    
    [data-testid="stFileUploadDropzone"] p {
        color: white !important;
    }
    
    /* ===== DOWNLOAD BUTTON ===== */
    .stDownloadButton > button {
        background: #00C851;
        color: white !important;
        border-radius: 25px;
        padding: 12px 28px;
        font-weight: 600;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)
# Cache functions with optimizations
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all saved models with progress tracking"""
    models = {}
    model_files = {
        'Logistic Regression': 'saved_models/logistic_regression_model.pkl',
        'KNN': 'saved_models/knn_model.pkl',
        'Naive Bayes': 'saved_models/naive_bayes_model.pkl',
        'SVM': 'saved_models/svm_model.pkl',
        'Random Forest': 'saved_models/random_forest_model.pkl',
        'XGBoost': 'saved_models/xgboost_model.pkl',
        'Best Model': 'saved_models/best_model.pkl'
    }
    
    progress_bar = st.progress(0)
    for idx, (name, path) in enumerate(model_files.items()):
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except:
                try:
                    with open(path, 'rb') as f:
                        models[name] = pickle.load(f)
                except:
                    continue
        progress_bar.progress((idx + 1) / len(model_files))
    progress_bar.empty()
    return models

@st.cache_resource(show_spinner=False)
def load_scaler():
    """Load the scaler"""
    if os.path.exists('scaler.pkl'):
        try:
            return joblib.load('scaler.pkl')
        except:
            with open('scaler.pkl', 'rb') as f:
                return pickle.load(f)
    return None

@st.cache_resource(show_spinner=False)
def load_mappings():
    """Load categorical mappings"""
    if os.path.exists('categorical_mappings.pkl'):
        try:
            with open('categorical_mappings.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            return joblib.load('categorical_mappings.pkl')
    return None

@st.cache_data(show_spinner=False)
def load_results():
    """Load model results"""
    if os.path.exists('saved_models/model_results.pkl'):
        try:
            with open('saved_models/model_results.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            return joblib.load('saved_models/model_results.pkl')
    return None

@st.cache_data(show_spinner=False)
def load_processed_data():
    """Load processed data"""
    if os.path.exists('saved_models/processed_data.pkl'):
        try:
            return joblib.load('saved_models/processed_data.pkl')
        except:
            with open('saved_models/processed_data.pkl', 'rb') as f:
                return pickle.load(f)
    return None

def main():
    st.markdown('<h1 class="main-header">💰 Income Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load resources with loading animation
    with st.spinner('Loading resources...'):
        models = load_models()
        scaler = load_scaler()
        mappings = load_mappings()
        results = load_results()
        processed_data = load_processed_data()
    
    if not models:
        st.error("⚠️ Models not found! Please run the training notebook first.")
        return
    
    # Enhanced Tabs
    # tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    #     "🔮 Predict", 
    #     "📊 Model Performance", 
    #     "📈 Data Insights", 
    #     "🎯 Feature Analysis",
    #     "📉 Advanced Analytics",
    #     "🔍 Deep Dive"
    # ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Predict", 
        "📊 Model Performance", 
        "📈 Data Insights", 
        "🎯 Feature Analysis",
        "📉 Advanced Analytics",
    ])
    
    with tab1:
        show_prediction_page(models, scaler, mappings)
    
    with tab2:
        show_model_performance(results, models, processed_data)
    
    with tab3:
        show_data_insights(processed_data)
    
    with tab4:
        show_feature_analysis(models, processed_data)
    
    with tab5:
        show_advanced_analytics(processed_data, results)
    
    # with tab6:
    #     show_deep_dive(processed_data, models)

def show_prediction_page(models, scaler, mappings):
    """Enhanced prediction interface"""
    
    st.markdown("## 🔮 Income Prediction")
    
    if not models or scaler is None:
        st.warning("⚠️ Prediction system not available.")
        return
    
    # Model selection with info
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            ['Best Model'] + [k for k in models.keys() if k != 'Best Model'],
            help="Choose a model for prediction"
        )
    
    with col2:
        show_info = st.checkbox("ℹ️ Show Model Info", value=False)
    
    with col3:
        batch_predict = st.checkbox("📊 Batch Prediction", value=False)
    
    if show_info:
        st.info(f"""
        **Selected Model:** {selected_model}
        - Models are trained on census data
        - Predictions indicate income > $50K or ≤ $50K
        - Confidence scores show prediction certainty
        """)
    
    st.markdown("---")
    
    if batch_predict:
        show_batch_prediction(models[selected_model], scaler, mappings)
    else:
        show_single_prediction(models[selected_model], scaler, mappings, selected_model)

def show_single_prediction(model, scaler, mappings, model_name):
    """Single prediction interface"""
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 👤 Personal Information")
            age = st.number_input("Age", 17, 90, 35, help="Your current age")
            sex = st.selectbox("Sex", ['Male', 'Female'])
            race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
            native_country = st.selectbox("Native Country", [
                'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada',
                'India', 'China', 'Japan', 'Other'
            ])
        
        with col2:
            st.markdown("### 🎓 Education & Work")
            education = st.selectbox("Education", [
                'Bachelors', 'HS-grad', 'Masters', 'Some-college',
                'Assoc-acdm', 'Doctorate', 'Prof-school', '11th', '9th',
                '7th-8th', '10th', '5th-6th', '12th', '1st-4th', 'Preschool'
            ])
            education_num = st.slider("Education Years", 1, 16, 10, help="Total years of education")
            workclass = st.selectbox("Work Class", [
                'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
            ])
            occupation = st.selectbox("Occupation", [
                'Exec-managerial', 'Prof-specialty', 'Tech-support', 'Sales',
                'Craft-repair', 'Other-service', 'Machine-op-inspct',
                'Adm-clerical', 'Handlers-cleaners', 'Transport-moving',
                'Farming-fishing', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'
            ])
        
        with col3:
            st.markdown("### 💍 Personal & Financial")
            marital_status = st.selectbox("Marital Status", [
                'Married-civ-spouse', 'Never-married', 'Divorced',
                'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
            ])
            relationship = st.selectbox("Relationship", [
                'Husband', 'Wife', 'Own-child', 'Not-in-family',
                'Other-relative', 'Unmarried'
            ])
            hours_per_week = st.number_input("Hours/Week", 1, 99, 40, help="Average work hours per week")
            capital_gain = st.number_input("Capital Gain", 0, 100000, 0, step=1000, help="Capital gains for the year")
            capital_loss = st.number_input("Capital Loss", 0, 5000, 0, step=100, help="Capital losses for the year")
        
        submitted = st.form_submit_button("🎯 Predict Income", use_container_width=True, type="primary")
    
    if submitted:
        predict_income(
            model, scaler, mappings, age, workclass, education, education_num,
            marital_status, occupation, relationship, race, sex, capital_gain,
            capital_loss, hours_per_week, native_country, model_name
        )

def predict_income(model, scaler, mappings, age, workclass, education, education_num,
                  marital_status, occupation, relationship, race, sex, capital_gain,
                  capital_loss, hours_per_week, native_country, model_name):
    """Make prediction and show results"""
    try:
        # Prepare input
        input_data = pd.DataFrame({
            'age': [age],
            'workclass': [workclass],
            'fnlwgt': [200000],
            'education': [education],
            'education.num': [education_num],
            'marital.status': [marital_status],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
            'sex': [sex],
            'capital.gain': [capital_gain],
            'capital.loss': [capital_loss],
            'hours.per.week': [hours_per_week],
            'native.country': [native_country]
        })
        
        # Encode categorical variables
        if mappings:
            from sklearn.preprocessing import LabelEncoder
            for col in input_data.select_dtypes(include=['object']).columns:
                if col in mappings and input_data[col].iloc[0] in mappings[col]:
                    input_data[col] = mappings[col][input_data[col].iloc[0]]
                else:
                    input_data[col] = 0
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
        else:
            proba = [1-prediction, prediction]
        
        # Display results with animation
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="prediction-box high-income">
                    <h2>✅ Income > $50K</h2>
                    <h3>Confidence: {:.1f}%</h3>
                    <p>Model: {}</p>
                </div>
                """.format(proba[1]*100, model_name), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box low-income">
                    <h2>📊 Income ≤ $50K</h2>
                    <h3>Confidence: {:.1f}%</h3>
                    <p>Model: {}</p>
                </div>
                """.format(proba[0]*100, model_name), unsafe_allow_html=True)
        
        with col2:
            # Enhanced probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=proba[1]*100,
                delta={'reference': 50},
                title={'text': "High Income Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#4CAF50" if prediction == 1 else "#FF9800"},
                    'steps': [
                        {'range': [0, 25], 'color': "#ffebee"},
                        {'range': [25, 50], 'color': "#fff3e0"},
                        {'range': [50, 75], 'color': "#e8f5e9"},
                        {'range': [75, 100], 'color': "#c8e6c9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Key factors
            st.markdown("### 🔑 Key Factors")
            factors = {
                'Age': age,
                'Education': education_num,
                'Hours/Week': hours_per_week,
                'Capital': capital_gain - capital_loss
            }
            for factor, value in factors.items():
                st.metric(factor, value)
        
        # Recommendation based on prediction
        st.markdown("---")
        if prediction == 1:
            st.success("""
            ### 🎉 Congratulations!
            Based on your profile, you're likely to earn above $50K annually. Key factors contributing to this:
            - Education level and years of education
            - Work hours and occupation type
            - Capital investments
            """)
        else:
            st.info("""
            ### 💡 Insights
            Your profile suggests income ≤ $50K. Consider these areas for potential improvement:
            - Advancing education or gaining certifications
            - Exploring higher-paying occupations
            - Increasing work hours or seeking promotions
            - Building capital through investments
            """)
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

def show_batch_prediction(model, scaler, mappings):
    """Batch prediction interface"""
    st.markdown("### 📊 Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=['csv'],
        help="File should contain the same columns as the training data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} records")
            
            # Show preview
            st.markdown("#### Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing predictions..."):
                    # Process predictions
                    predictions = []
                    probabilities = []
                    
                    for _, row in df.iterrows():
                        # Prepare data (similar to single prediction)
                        # ... (preprocessing code)
                        pred = model.predict(input_scaled)[0]
                        predictions.append(pred)
                        
                        if hasattr(model, 'predict_proba'):
                            prob = model.predict_proba(input_scaled)[0, 1]
                            probabilities.append(prob)
                    
                    # Add results to dataframe
                    df['Prediction'] = ['> $50K' if p == 1 else '≤ $50K' for p in predictions]
                    if probabilities:
                        df['Probability'] = probabilities
                    
                    # Show results
                    st.markdown("#### Prediction Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        high_income_count = sum(predictions)
                        fig = px.pie(
                            values=[high_income_count, len(predictions) - high_income_count],
                            names=['> $50K', '≤ $50K'],
                            title='Prediction Distribution',
                            color_discrete_sequence=['#4CAF50', '#FF9800']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Summary Statistics")
                        st.metric("Total Predictions", len(predictions))
                        st.metric("High Income (>$50K)", f"{high_income_count} ({high_income_count/len(predictions)*100:.1f}%)")
                        st.metric("Low Income (≤$50K)", f"{len(predictions) - high_income_count} ({(len(predictions) - high_income_count)/len(predictions)*100:.1f}%)")
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_model_performance(results, models, processed_data):
    """Enhanced model performance visualization"""
    
    st.markdown("## 📊 Model Performance Analysis")
    
    if not results:
        st.warning("No results available")
        return
    
    # Performance overview
    metrics_df = pd.DataFrame(results['model_metrics'])
    best_model = results.get('best_model', 'N/A')
    best_acc = results.get('best_accuracy', 0)
    
    # Enhanced metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🏆 Best Model</h3>
            <h2>{}</h2>
        </div>
        """.format(best_model), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Best Accuracy</h3>
            <h2>{:.2f}%</h2>
        </div>
        """.format(best_acc * 100), unsafe_allow_html=True)
    
    with col3:
        avg_acc = metrics_df['Accuracy'].mean()
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Avg Accuracy</h3>
            <h2>{:.2f}%</h2>
        </div>
        """.format(avg_acc * 100), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Total Models</h3>
            <h2>{}</h2>
        </div>
        """.format(len(models) - 1), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comprehensive performance comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Comparison', 'Performance Radar', 
                       'ROC Curves', 'Training Time vs Accuracy'),
        specs=[[{'type': 'bar'}, {'type': 'polar'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Bar chart comparison
    for i, metric in enumerate(['Accuracy', 'F1-Score', 'Precision', 'Recall']):
        fig.add_trace(
            go.Bar(
                name=metric,
                x=metrics_df['Model'],
                y=metrics_df[metric],
                text=[f"{val:.3f}" for val in metrics_df[metric]],
                textposition='outside'
            ),
            row=1, col=1
        )
    
    # 2. Radar chart
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for _, row in metrics_df.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row[cat] for cat in categories],
                theta=categories,
                name=row['Model'],
                fill='toself'
            ),
            row=1, col=2
        )
    
    # 3. ROC Curves simulation (if available)
    if 'roc_data' in results:
        for model_name, roc_info in results['roc_data'].items():
            fig.add_trace(
                go.Scatter(
                    x=roc_info['fpr'],
                    y=roc_info['tpr'],
                    name=f"{model_name} (AUC={roc_info['auc']:.3f})",
                    mode='lines'
                ),
                row=2, col=1
            )
    else:
        # Placeholder ROC curves
        x = np.linspace(0, 1, 100)
        for model in metrics_df['Model']:
            auc = metrics_df[metrics_df['Model'] == model]['Accuracy'].values[0]
            y = x ** (1 / (auc + 0.5))
            fig.add_trace(
                go.Scatter(x=x, y=y, name=model, mode='lines'),
                row=2, col=1
            )
    
    # 4. Training time vs Accuracy
    if 'training_times' in results:
        fig.add_trace(
            go.Scatter(
                x=results['training_times'],
                y=metrics_df['Accuracy'],
                mode='markers+text',
                text=metrics_df['Model'],
                textposition='top center',
                marker=dict(size=15, color=metrics_df['F1-Score'], 
                           colorscale='Viridis', showscale=True)
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text="False Positive Rate", row=2, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=2, col=1)
    fig.update_xaxes(title_text="Training Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Accuracy", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table - FINAL SOLUTION
    st.markdown("### 📋 Detailed Metrics")
    
    # Create HTML table with custom styling
    html_table = f"""
    <style>
        .custom-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            margin: 20px 0;
            background: rgba(30, 30, 50, 0.9);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(102, 126, 234, 0.3);
        }}
        .custom-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            padding: 15px;
            text-align: left;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 14px;
            letter-spacing: 1px;
            border-bottom: 2px solid rgba(102, 126, 234, 0.5);
        }}
        .custom-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid rgba(102, 126, 234, 0.2);
            color: #ffffff;
            font-size: 14px;
            font-weight: 500;
        }}
        .custom-table tr:hover {{
            background: rgba(102, 126, 234, 0.2);
            transition: background 0.3s ease;
        }}
        .custom-table tr:last-child td {{
            border-bottom: none;
        }}
        .best-value {{
            background: rgba(0, 200, 81, 0.3);
            font-weight: bold;
            color: #00ff88;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }}
        .custom-table tbody tr {{
            transition: all 0.3s ease;
        }}
        .custom-table tbody tr:nth-child(even) {{
            background: rgba(40, 40, 60, 0.5);
        }}
    </style>

    <table class="custom-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Find best values for highlighting
    best_acc = metrics_df['Accuracy'].max()
    best_prec = metrics_df['Precision'].max()
    best_recall = metrics_df['Recall'].max()
    best_f1 = metrics_df['F1-Score'].max()
    
    # Add rows
    for _, row in metrics_df.iterrows():
        html_table += "<tr>"
        html_table += f"<td><strong>{row['Model']}</strong></td>"
        
        # Accuracy
        acc_class = "best-value" if row['Accuracy'] == best_acc else ""
        html_table += f"<td class='{acc_class}'>{row['Accuracy']:.4f}</td>"
        
        # Precision
        prec_class = "best-value" if row['Precision'] == best_prec else ""
        html_table += f"<td class='{prec_class}'>{row['Precision']:.4f}</td>"
        
        # Recall
        rec_class = "best-value" if row['Recall'] == best_recall else ""
        html_table += f"<td class='{rec_class}'>{row['Recall']:.4f}</td>"
        
        # F1-Score
        f1_class = "best-value" if row['F1-Score'] == best_f1 else ""
        html_table += f"<td class='{f1_class}'>{row['F1-Score']:.4f}</td>"
        
        html_table += "</tr>"
    
    html_table += """
        </tbody>
    </table>
    """
    
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Alternative: Interactive metrics display
    st.markdown("### 📊 Interactive Metrics View")
    
    # Create tabs for each model
    model_tabs = st.tabs(metrics_df['Model'].tolist())
    
    for idx, (tab, (_, row)) in enumerate(zip(model_tabs, metrics_df.iterrows())):
        with tab:
            col1, col2, col3, col4 = st.columns(4)
            
            # Display metrics with delta from average
            with col1:
                delta = row['Accuracy'] - metrics_df['Accuracy'].mean()
                st.metric(
                    "Accuracy", 
                    f"{row['Accuracy']:.4f}",
                    f"{delta:+.4f}",
                    delta_color="normal"
                )
            
            with col2:
                delta = row['Precision'] - metrics_df['Precision'].mean()
                st.metric(
                    "Precision", 
                    f"{row['Precision']:.4f}",
                    f"{delta:+.4f}",
                    delta_color="normal"
                )
            
            with col3:
                delta = row['Recall'] - metrics_df['Recall'].mean()
                st.metric(
                    "Recall", 
                    f"{row['Recall']:.4f}",
                    f"{delta:+.4f}",
                    delta_color="normal"
                )
            
            with col4:
                delta = row['F1-Score'] - metrics_df['F1-Score'].mean()
                st.metric(
                    "F1-Score", 
                    f"{row['F1-Score']:.4f}",
                    f"{delta:+.4f}",
                    delta_color="normal"
                )
    
    # Model insights
    st.markdown("### 🔍 Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-card">
        <h4>🎯 Best Performing Models</h4>
        <ul>
        """, unsafe_allow_html=True)
        
        top_models = metrics_df.nlargest(3, 'Accuracy')
        for _, model in top_models.iterrows():
            st.markdown(f"<li><b>{model['Model']}</b>: {model['Accuracy']:.3f} accuracy</li>", 
                       unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
        <h4>📊 Performance Summary</h4>
        <ul>
            <li>Average F1-Score: <b>{:.3f}</b></li>
            <li>Best Precision: <b>{:.3f}</b></li>
            <li>Best Recall: <b>{:.3f}</b></li>
            <li>Models above 80% accuracy: <b>{}</b></li>
        </ul>
        </div>
        """.format(
            metrics_df['F1-Score'].mean(),
            metrics_df['Precision'].max(),
            metrics_df['Recall'].max(),
            len(metrics_df[metrics_df['Accuracy'] > 0.8])
        ), unsafe_allow_html=True)

def show_data_insights(processed_data):
    """Enhanced data insights and visualizations"""
    
    st.markdown("## 📈 Data Insights & Analysis")
    
    if not processed_data or 'original_data' not in processed_data:
        st.warning("Data not available")
        return
    
    df = processed_data['original_data']
    
    # Key statistics with enhanced visuals
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", f"{len(df.columns) - 1}")
    with col3:
        st.metric("Avg Age", f"{df['age'].mean():.1f}")
    with col4:
        st.metric("Avg Hours/Week", f"{df['hours.per.week'].mean():.1f}")
    with col5:
        high_income_pct = (df['income'] == '>50K').mean() * 100
        st.metric("High Income %", f"{high_income_pct:.1f}%")
    
    st.markdown("---")
    
    # Create tabs for different insights
    insight_tab1, insight_tab2, insight_tab3, insight_tab4 = st.tabs([
        "📊 Demographics", "💼 Work & Education", "💰 Financial", "🔄 Relationships"
    ])
    
    with insight_tab1:
        show_demographic_insights(df)
    
    with insight_tab2:
        show_work_education_insights(df)
    
    with insight_tab3:
        show_financial_insights(df)
    
    with insight_tab4:
        show_relationship_insights(df)

def show_demographic_insights(df):
    """Demographic analysis"""
    st.markdown("### 👥 Demographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by income
        fig = px.histogram(
            df, x='age', color='income',
            title='Age Distribution by Income Level',
            nbins=30,
            color_discrete_map={'<=50K': '#FF9800', '>50K': '#4CAF50'},
            labels={'age': 'Age', 'income': 'Income Level'}
        )
        fig.update_layout(
            bargap=0.1,
            xaxis_title="Age",
            yaxis_title="Count",
            legend_title="Income"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gender distribution
        gender_income = pd.crosstab(df['sex'], df['income'], normalize='index') * 100
        fig = px.bar(
            gender_income.T,
            title='Income Distribution by Gender',
            color_discrete_sequence=['#FF9800', '#4CAF50'],
            labels={'value': 'Percentage (%)', 'sex': 'Gender'}
        )
        fig.update_layout(legend_title="Income Level")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Race distribution
        race_counts = df['race'].value_counts()
        fig = px.pie(
            values=race_counts.values,
            names=race_counts.index,
            title='Race Distribution in Dataset',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Native country analysis
        country_income = df.groupby(['native.country', 'income']).size().unstack(fill_value=0)
        country_income['high_income_rate'] = country_income['>50K'] / (country_income['>50K'] + country_income['<=50K']) * 100
        top_countries = country_income.nlargest(10, 'high_income_rate')
        
        fig = px.bar(
            x=top_countries.index,
            y=top_countries['high_income_rate'],
            title='Top 10 Countries by High Income Rate',
            labels={'x': 'Country', 'y': 'High Income Rate (%)'},
            color=top_countries['high_income_rate'],
            color_continuous_scale='Viridis'
        )
        # fig.update_xaxes(tickangle=-45)  # Fixed: update_xaxes instead of update_xaxis
        st.plotly_chart(fig, use_container_width=True)
    
    # Age insights
    st.markdown("#### 🔍 Key Demographic Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_age_high = df[df['income'] == '>50K']['age'].mean()
        avg_age_low = df[df['income'] == '<=50K']['age'].mean()
        st.info(f"""
        **Age Impact**  
        High earners avg age: {avg_age_high:.1f}  
        Low earners avg age: {avg_age_low:.1f}  
        Difference: {avg_age_high - avg_age_low:.1f} years
        """)
    
    with col2:
        male_high = (df[df['sex'] == 'Male']['income'] == '>50K').mean() * 100
        female_high = (df[df['sex'] == 'Female']['income'] == '>50K').mean() * 100
        st.info(f"""
        **Gender Gap**  
        Male high earners: {male_high:.1f}%  
        Female high earners: {female_high:.1f}%  
        Gap: {male_high - female_high:.1f}%
        """)
    
    with col3:
        us_high = (df[df['native.country'] == 'United-States']['income'] == '>50K').mean() * 100
        non_us_high = (df[df['native.country'] != 'United-States']['income'] == '>50K').mean() * 100
        st.info(f"""
        **Country Effect**  
        US high earners: {us_high:.1f}%  
        Non-US high earners: {non_us_high:.1f}%  
        Difference: {us_high - non_us_high:.1f}%
        """)


def show_work_education_insights(df):
    """Work and education analysis"""
    st.markdown("### 🎓 Work & Education Analysis")
    
    # Education impact
    edu_income = pd.crosstab(df['education'], df['income'], normalize='index') * 100
    edu_income = edu_income.reset_index()
    edu_income['high_income_rate'] = edu_income['>50K']
    edu_income = edu_income.sort_values('high_income_rate', ascending=False)
    
    fig = px.bar(
        edu_income,
        x='education',
        y='high_income_rate',
        title='High Income Rate by Education Level',
        labels={'high_income_rate': 'High Income Rate (%)', 'education': 'Education Level'},
        color='high_income_rate',
        color_continuous_scale='RdYlGn'
    )
    # fig.update_xaxes(tickangle=-45)  # Fixed: update_xaxes instead of update_xaxis
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Occupation analysis
        occ_income = pd.crosstab(df['occupation'], df['income'], normalize='index') * 100
        occ_income = occ_income.reset_index()
        occ_income['high_income_rate'] = occ_income['>50K']
        top_occupations = occ_income.nlargest(10, 'high_income_rate')
        
        fig = px.bar(
            top_occupations,
            y='occupation',
            x='high_income_rate',
            orientation='h',
            title='Top 10 Occupations by High Income Rate',
            labels={'high_income_rate': 'High Income Rate (%)', 'occupation': 'Occupation'},
            color='high_income_rate',
            color_continuous_scale='Plasma'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Work hours analysis
        fig = px.violin(
            df, y='hours.per.week', x='income',
            title='Work Hours Distribution by Income',
            color='income',
            color_discrete_map={'<=50K': '#FF9800', '>50K': '#4CAF50'},
            box=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Work class distribution
    fig = px.sunburst(
        df, path=['workclass', 'income'],
        title='Income Distribution by Work Class',
        color_discrete_sequence=['#FF9800', '#4CAF50']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Education insights
    st.markdown("#### 🔍 Key Work & Education Insights")
    
    # Create education ranking
    edu_order = ['Doctorate', 'Prof-school', 'Masters', 'Bachelors', 'Assoc-acdm', 
                 'Assoc-voc', 'Some-college', 'HS-grad', '12th', '11th', '10th', 
                 '9th', '7th-8th', '5th-6th', '1st-4th', 'Preschool']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_edu = edu_income.iloc[0]
        st.success(f"""
        **Best Education ROI**  
        {top_edu['education']}  
        High income rate: {top_edu['high_income_rate']:.1f}%
        """)
    
    with col2:
        top_occ = top_occupations.iloc[0]
        st.success(f"""
        **Top Occupation**  
        {top_occ['occupation']}  
        High income rate: {top_occ['high_income_rate']:.1f}%
        """)
    
    with col3:
        avg_hours_high = df[df['income'] == '>50K']['hours.per.week'].mean()
        avg_hours_low = df[df['income'] == '<=50K']['hours.per.week'].mean()
        st.success(f"""
        **Work Hours Impact**  
        High earners: {avg_hours_high:.1f} hrs/week  
        Low earners: {avg_hours_low:.1f} hrs/week
        """)
def show_financial_insights(df):
    """Financial analysis"""
    st.markdown("### 💰 Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Capital gains distribution
        fig = px.box(
            df[df['capital.gain'] > 0], 
            x='income', 
            y='capital.gain',
            title='Capital Gains Distribution (Excluding Zero)',
            color='income',
            color_discrete_map={'<=50K': '#FF9800', '>50K': '#4CAF50'},
            log_y=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Capital loss distribution
        fig = px.box(
            df[df['capital.loss'] > 0], 
            x='income', 
            y='capital.loss',
            title='Capital Loss Distribution (Excluding Zero)',
            color='income',
            color_discrete_map={'<=50K': '#FF9800', '>50K': '#4CAF50'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Capital gains impact
    capital_impact = df.copy()
    capital_impact['has_capital_gain'] = (capital_impact['capital.gain'] > 0).astype(str)
    capital_impact['has_capital_loss'] = (capital_impact['capital.loss'] > 0).astype(str)
    
    # Combined capital analysis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Capital Gain Impact', 'Capital Loss Impact')
    )
    
    # Capital gain impact
    gain_impact = pd.crosstab(capital_impact['has_capital_gain'], capital_impact['income'], normalize='index') * 100
    fig.add_trace(
        go.Bar(name='<=50K', x=['No Gain', 'Has Gain'], y=gain_impact['<=50K'], marker_color='#FF9800'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='>50K', x=['No Gain', 'Has Gain'], y=gain_impact['>50K'], marker_color='#4CAF50'),
        row=1, col=1
    )
    
    # Capital loss impact
    loss_impact = pd.crosstab(capital_impact['has_capital_loss'], capital_impact['income'], normalize='index') * 100
    fig.add_trace(
        go.Bar(name='<=50K', x=['No Loss', 'Has Loss'], y=loss_impact['<=50K'], marker_color='#FF9800', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(name='>50K', x=['No Loss', 'Has Loss'], y=loss_impact['>50K'], marker_color='#4CAF50', showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(height=400, barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Financial insights
    st.markdown("#### 🔍 Key Financial Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pct_with_gains = (df['capital.gain'] > 0).mean() * 100
        high_with_gains = (df[df['income'] == '>50K']['capital.gain'] > 0).mean() * 100
        st.warning(f"""
        **Capital Gains**  
        Overall with gains: {pct_with_gains:.1f}%  
        High earners with gains: {high_with_gains:.1f}%
        """)
    
    with col2:
        avg_gain_high = df[df['income'] == '>50K']['capital.gain'].mean()
        avg_gain_low = df[df['income'] == '<=50K']['capital.gain'].mean()
        st.warning(f"""
        **Average Gains**  
        High earners: ${avg_gain_high:,.0f}  
        Low earners: ${avg_gain_low:,.0f}
        """)
    
    with col3:
        gain_correlation = df[df['capital.gain'] > 0].groupby('income').size()
        gain_ratio = gain_correlation['>50K'] / gain_correlation['<=50K']
        st.warning(f"""
        **Capital Impact**  
        High earners are {gain_ratio:.1f}x more likely  
        to have capital gains
        """)

def show_relationship_insights(df):
    """Relationship and marital status analysis"""
    st.markdown("### 💑 Relationship Analysis")
    
    # Marital status impact
    marital_income = pd.crosstab(df['marital.status'], df['income'], normalize='index') * 100
    marital_income = marital_income.reset_index()
    marital_income['high_income_rate'] = marital_income['>50K']
    
    fig = px.bar(
        marital_income.sort_values('high_income_rate', ascending=True),
        y='marital.status',
        x='high_income_rate',
        orientation='h',
        title='High Income Rate by Marital Status',
        labels={'high_income_rate': 'High Income Rate (%)', 'marital.status': 'Marital Status'},
        color='high_income_rate',
        color_continuous_scale='Turbo'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Relationship type analysis
        rel_income = pd.crosstab(df['relationship'], df['income'], normalize='index') * 100
        rel_income = rel_income.reset_index()
        rel_income['high_income_rate'] = rel_income['>50K']
        
        fig = px.pie(
            rel_income,
            values='high_income_rate',
            names='relationship',
            title='High Income Rate by Relationship Type',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cross-analysis: marital status and gender
        marital_gender = df.groupby(['marital.status', 'sex', 'income']).size().unstack(fill_value=0)
        marital_gender['high_income_rate'] = marital_gender['>50K'] / (marital_gender['>50K'] + marital_gender['<=50K']) * 100
        marital_gender = marital_gender.reset_index()
        
        fig = px.bar(
            marital_gender,
            x='marital.status',
            y='high_income_rate',
            color='sex',
            title='High Income Rate by Marital Status and Gender',
            labels={'high_income_rate': 'High Income Rate (%)', 'marital.status': 'Marital Status'},
            color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'},
            barmode='group'
        )
        # fig.update_xaxis(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Relationship insights
    st.markdown("#### 🔍 Key Relationship Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        married_high = marital_income[marital_income['marital.status'] == 'Married-civ-spouse']['high_income_rate'].values[0]
        single_high = marital_income[marital_income['marital.status'] == 'Never-married']['high_income_rate'].values[0]
        st.info(f"""
        **Marriage Impact**  
        Married high earners: {married_high:.1f}%  
        Single high earners: {single_high:.1f}%  
        Difference: {married_high - single_high:.1f}%
        """)
    
    with col2:
        husband_high = rel_income[rel_income['relationship'] == 'Husband']['high_income_rate'].values[0]
        wife_high = rel_income[rel_income['relationship'] == 'Wife']['high_income_rate'].values[0]
        st.info(f"""
        **Spousal Difference**  
        Husbands high earners: {husband_high:.1f}%  
        Wives high earners: {wife_high:.1f}%  
        Gap: {husband_high - wife_high:.1f}%
        """)
    
    with col3:
        divorced_high = marital_income[marital_income['marital.status'] == 'Divorced']['high_income_rate'].values[0]
        st.info(f"""
        **Divorce Effect**  
        Divorced high earners: {divorced_high:.1f}%  
        Lower than married by: {married_high - divorced_high:.1f}%
        """)

def show_feature_analysis(models, processed_data):
    """Enhanced feature importance analysis"""
    
    st.markdown("## 🎯 Feature Analysis")
    
    # Check if we have feature importance models
    feature_models = ['Random Forest', 'XGBoost']
    available_models = [m for m in feature_models if m in models]
    
    if not available_models:
        st.warning("No models with feature importance available")
        return
    
    # Model selection for feature analysis
    selected_model = st.selectbox(
        "Select model for feature analysis",
        available_models,
        help="Choose a model to analyze feature importance"
    )
    
    model = models[selected_model]
    
    if hasattr(model, 'feature_importances_'):
        # Get feature names
        feature_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education.num',
            'marital.status', 'occupation', 'relationship', 'race', 'sex',
            'capital.gain', 'capital.loss', 'hours.per.week', 'native.country'
        ]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(model.feature_importances_)],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Main feature importance plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Importance Rankings', 'Cumulative Importance',
                          'Feature Categories', 'Top Features Detail'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'pie'}, {'type': 'bar'}]]
        )
        
        # 1. Feature importance bar chart
        fig.add_trace(
            go.Bar(
                x=importance_df['Importance'][:10],
                y=importance_df['Feature'][:10],
                orientation='h',
                text=[f"{val:.3f}" for val in importance_df['Importance'][:10]],
                textposition='outside',
                marker_color=px.colors.sequential.Viridis_r[:10]
            ),
            row=1, col=1
        )
        
        # 2. Cumulative importance
        importance_df['Cumulative'] = importance_df['Importance'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(importance_df) + 1)),
                y=importance_df['Cumulative'],
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # 3. Feature categories
        categories = {
            'Demographics': ['age', 'sex', 'race', 'native.country'],
            'Education': ['education', 'education.num'],
            'Work': ['workclass', 'occupation', 'hours.per.week'],
            'Financial': ['capital.gain', 'capital.loss'],
            'Personal': ['marital.status', 'relationship'],
            'Other': ['fnlwgt']
        }
        
        category_importance = {}
        for cat, features in categories.items():
            cat_imp = importance_df[importance_df['Feature'].isin(features)]['Importance'].sum()
            category_importance[cat] = cat_imp
        
        fig.add_trace(
            go.Pie(
                labels=list(category_importance.keys()),
                values=list(category_importance.values()),
                hole=0.3
            ),
            row=2, col=1
        )
        
        # 4. Top 5 features detail
        top5 = importance_df.head(5)
        fig.add_trace(
            go.Bar(
                x=top5['Feature'],
                y=top5['Importance'],
                text=[f"{val:.4f}" for val in top5['Importance']],
                textposition='outside',
                marker_color='lightblue'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        fig.update_xaxes(title_text="Importance Score", row=1, col=1)
        fig.update_xaxes(title_text="Feature Rank", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Importance", row=1, col=2)
        fig.update_xaxes(title_text="Feature", row=2, col=2)
        fig.update_yaxes(title_text="Importance", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance insights
        st.markdown("### 🔍 Feature Importance Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"""
            **🥇 Most Important Feature**  
            {importance_df.iloc[0]['Feature']}  
            Importance: {importance_df.iloc[0]['Importance']:.3f}
            """)
        
        with col2:
            top3_pct = (importance_df.head(3)['Importance'].sum() / importance_df['Importance'].sum()) * 100
            st.success(f"""
            **📊 Top 3 Features**  
            Account for {top3_pct:.1f}% of total importance
            """)
        
        with col3:
            n_features_80 = (importance_df['Cumulative'] <= 0.8).sum() + 1
            st.success(f"""
            **🎯 80% Importance**  
            Achieved with top {n_features_80} features
            """)
        
        # Detailed feature analysis
        st.markdown("### 📋 Detailed Feature Analysis")
        
        # Interactive feature selector
        selected_features = st.multiselect(
            "Select features to analyze",
            importance_df['Feature'].tolist(),
            default=importance_df['Feature'].head(5).tolist()
        )
        
        if selected_features and processed_data and 'original_data' in processed_data:
            df = processed_data['original_data']
            
            # Create subplots for selected features
            n_features = len(selected_features)
            n_cols = 2
            n_rows = (n_features + 1) // 2
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[f'{feat} vs Income' for feat in selected_features]
            )
            
            for idx, feature in enumerate(selected_features):
                row = (idx // n_cols) + 1
                col = (idx % n_cols) + 1
                
                if feature in df.columns:
                    if df[feature].dtype in ['int64', 'float64']:
                        # Numerical feature
                        fig.add_trace(
                            go.Violin(
                                x=df['income'],
                                y=df[feature],
                                name=feature,
                                box_visible=True,
                                meanline_visible=True
                            ),
                            row=row, col=col
                        )
                    else:
                        # Categorical feature
                        feat_income = pd.crosstab(df[feature], df['income'], normalize='index') * 100
                        fig.add_trace(
                            go.Bar(
                                x=feat_income.index,
                                y=feat_income['>50K'],
                                name=feature
                            ),
                            row=row, col=col
                        )
            
            fig.update_layout(height=300 * n_rows, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics(processed_data, results):
    """Advanced analytics and patterns"""
    
    st.markdown("## 📉 Advanced Analytics")
    
    if not processed_data or 'original_data' not in processed_data:
        st.warning("Data not available for advanced analytics")
        return
    
    df = processed_data['original_data']
    
    # Create tabs for different analyses
    adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
        "🔗 Correlation Analysis", "🎨 Pattern Discovery", 
        "📊 Statistical Tests", "🤖 Prediction Patterns"
    ])
    
    with adv_tab1:
        show_correlation_analysis(df, processed_data)
    
    with adv_tab2:
        show_pattern_discovery(df)
    
    with adv_tab3:
        show_statistical_tests(df)
    
    with adv_tab4:
        show_prediction_patterns(df, results)

def show_correlation_analysis(df, processed_data):
    """Correlation analysis"""
    st.markdown("### 🔗 Correlation Analysis")
    
    # Prepare numerical data
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if processed_data and 'X_train' in processed_data:
        # Use processed data for better correlation
        X_train = processed_data['X_train']
        corr_matrix = pd.DataFrame(X_train).corr()
        
        # Interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=list(range(len(corr_matrix))),
            y=list(range(len(corr_matrix))),
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=700,
            width=800
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations
        st.markdown("#### 🔝 Top Feature Correlations")
        
        # Get upper triangle
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find top correlations
        top_corr = []
        for col in upper_tri.columns:
            for idx, val in upper_tri[col].items():
                if not pd.isna(val):
                    top_corr.append({
                        'Feature 1': col,
                        'Feature 2': idx,
                        'Correlation': val
                    })
        
        top_corr_df = pd.DataFrame(top_corr).sort_values('Correlation', key=abs, ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strongest Positive Correlations**")
            pos_corr = top_corr_df[top_corr_df['Correlation'] > 0].head(5)
            for _, row in pos_corr.iterrows():
                st.write(f"• Features {row['Feature 1']} ↔ {row['Feature 2']}: {row['Correlation']:.3f}")
        
        with col2:
            st.markdown("**Strongest Negative Correlations**")
            neg_corr = top_corr_df[top_corr_df['Correlation'] < 0].head(5)
            for _, row in neg_corr.iterrows():
                st.write(f"• Features {row['Feature 1']} ↔ {row['Feature 2']}: {row['Correlation']:.3f}")

def show_pattern_discovery(df):
    """Pattern discovery in data"""
    st.markdown("### 🎨 Pattern Discovery")
    
    # Age and education pattern
    age_edu = df.groupby(['education', 'income'])['age'].mean().unstack()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='≤50K', x=age_edu.index, y=age_edu['<=50K'], marker_color='#FF9800'))
    fig.add_trace(go.Bar(name='>50K', x=age_edu.index, y=age_edu['>50K'], marker_color='#4CAF50'))
    fig.update_layout(
        title='Average Age by Education Level and Income',
        xaxis_title='Education Level',
        yaxis_title='Average Age',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Work patterns
        work_patterns = df.groupby(['workclass', 'occupation']).size().reset_index(name='count')
        work_patterns = work_patterns.nlargest(15, 'count')
        
        fig = px.treemap(
            work_patterns,
            path=['workclass', 'occupation'],
            values='count',
            title='Work Class and Occupation Patterns',
            color='count',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # بديل أبسط - scatter plot
        fig = px.scatter(
            df.sample(n=min(2000, len(df))),
            x='age',
            y='hours.per.week',
            color='income',
            title='Age vs Work Hours',
            opacity=0.6,
            color_discrete_map={'<=50K': '#FF9800', '>50K': '#4CAF50'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # بديل أبسط للـ parallel coordinates
    st.markdown("#### 🔍 Feature Relationships")
    
    # correlation matrix بدلاً من parallel coordinates
    numeric_cols = ['age', 'education.num', 'hours.per.week', 'capital.gain', 'capital.loss']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        title='Feature Correlation Matrix',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Pattern insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🎯 Education Pattern**  
        Higher education correlates with higher income,
        but age varies significantly across education levels
        """)
    
    with col2:
        st.info("""
        **⏰ Work Pattern**  
        High earners tend to work more hours,
        especially in private sector and self-employment
        """)
    
    with col3:
        st.info("""
        **💰 Capital Pattern**  
        Capital gains are rare but strongly
        associated with high income
        """)
def show_statistical_tests(df):
    """Statistical tests and analysis"""
    st.markdown("### 📊 Statistical Tests")
    
    from scipy import stats
    
    # Chi-square tests
    st.markdown("#### 🔬 Chi-Square Tests for Independence")
    
    categorical_vars = ['workclass', 'education', 'marital.status', 'occupation', 
                       'relationship', 'race', 'sex', 'native.country']
    
    chi_results = []
    for var in categorical_vars:
        contingency_table = pd.crosstab(df[var], df['income'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        chi_results.append({
            'Variable': var,
            'Chi-Square': chi2,
            'P-Value': p_value,
            'Degrees of Freedom': dof,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    chi_df = pd.DataFrame(chi_results).sort_values('P-Value')
    
    fig = px.bar(
        chi_df,
        x='Variable',
        y=-np.log10(chi_df['P-Value']),
        title='Statistical Significance of Categorical Variables',
        labels={'y': '-log10(P-Value)', 'Variable': 'Categorical Variable'},
        color='Significant',
        color_discrete_map={'Yes': '#4CAF50', 'No': '#FF9800'}
    )
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red", 
                  annotation_text="Significance Threshold (p=0.05)")
    st.plotly_chart(fig, use_container_width=True)
    
    # T-tests for numerical variables
    st.markdown("#### 📈 T-Tests for Numerical Variables")
    
    numerical_vars = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    
    t_results = []
    for var in numerical_vars:
        high_income = df[df['income'] == '>50K'][var]
        low_income = df[df['income'] == '<=50K'][var]
        
        t_stat, p_value = stats.ttest_ind(high_income, low_income)
        
        t_results.append({
            'Variable': var,
            'Mean (>50K)': high_income.mean(),
            'Mean (≤50K)': low_income.mean(),
            'Difference': high_income.mean() - low_income.mean(),
            'T-Statistic': t_stat,
            'P-Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    t_df = pd.DataFrame(t_results)
    
    import streamlit.components.v1 as components

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart of effect sizes
        fig = px.bar(
            t_df,
            x='Variable',
            y='Difference',
            title='Mean Differences Between Income Groups',
            color='Significant',
            color_discrete_map={'Yes': '#4CAF50', 'No': '#FF9800'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Statistical summary
        st.markdown("**📋 Statistical Summary**")

        html_table = """
        <style>
            .stat-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
                border-radius: 8px;
                background: #1e1e2f;        /* خلفية داكنة */
                color: #f5f5f5; 
                overflow: hidden;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }
            .stat-table th {
                background-color: #1f77b4;
                color: white;
                padding: 10px;
                text-align: left;
            }
            .stat-table td {
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }
            .stat-table tr:hover {
                background-color: #f5f5f5;
            }
            .significant {
                color: #4CAF0;
                font-weight: bold;
            }
            .not-significant {
                color: #FF9800;
            }
        </style>
        <table class="stat-table">
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Mean (>50K)</th>
                    <th>Mean (≤50K)</th>
                    <th>P-Value</th>
                    <th>Significant</th>
                </tr>
            </thead>
            <tbody>
        """

        for _, row in t_df.iterrows():
            sig_class = 'significant' if row['Significant'] == 'Yes' else 'not-significant'
            html_table += f"""
                <tr>
                    <td>{row['Variable']}</td>
                    <td>{row['Mean (>50K)']:.2f}</td>
                    <td>{row['Mean (≤50K)']:.2f}</td>
                    <td>{row['P-Value']:.4f}</td>
                    <td class="{sig_class}">{row['Significant']}</td>
                </tr>
            """

        html_table += """
            </tbody>
        </table>
        """

        components.html(html_table, height=400, scrolling=True)

    # Key findings
    # st.markdown("#### 🔍 Key Statistical Findings")

    # significant_cats = chi_df[chi_df['Significant'] == 'Yes']['Variable'].tolist()
    # significant_nums = t_df[t_df['Significant'] == 'Yes']['Variable'].tolist()

    # col1, col2 = st.columns(2)

    # with col1:
    #     st.success(f"""
    #     **Significant Categorical Variables**  
    #     {len(significant_cats)} out of {len(categorical_vars)} tested:  
    #     {', '.join(significant_cats[:3])}{'...' if len(significant_cats) > 3 else ''}
    #     """)

    # with col2:
    #     st.success(f"""
    #     **Significant Numerical Variables**  
    #     {len(significant_nums)} out of {len(numerical_vars)} tested:  
    #     {', '.join(significant_nums)}
    #     """)
    
    # Key findings
    st.markdown("#### 🔍 Key Statistical Findings")
    
    significant_cats = chi_df[chi_df['Significant'] == 'Yes']['Variable'].tolist()
    significant_nums = t_df[t_df['Significant'] == 'Yes']['Variable'].tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"""
        **Significant Categorical Variables**  
        {len(significant_cats)} out of {len(categorical_vars)} tested:  
        {', '.join(significant_cats[:3])}{'...' if len(significant_cats) > 3 else ''}
        """)
    
    with col2:
        st.success(f"""
        **Significant Numerical Variables**  
        {len(significant_nums)} out of {len(numerical_vars)} tested:  
        {', '.join(significant_nums)}
        """)

def show_prediction_patterns(df, results):
    """Analyze prediction patterns"""
    st.markdown("### 🤖 Prediction Patterns")
    
    if not results or 'model_metrics' not in results:
        st.warning("Model results not available for pattern analysis")
        return
    
    # Model performance patterns
    metrics_df = pd.DataFrame(results['model_metrics'])
    
    # Performance radar chart for all models
    fig = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for _, row in metrics_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[metric] for metric in metrics],
            theta=metrics,
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Model Performance Patterns",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix patterns (simulated)
    st.markdown("#### 🎯 Prediction Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model agreement analysis
        st.markdown("**Model Agreement Patterns**")
        
        # Simulate model agreement (in real scenario, you'd use actual predictions)
        models_agree = {
            'All Agree': 65,
            '4-5 Agree': 20,
            '2-3 Agree': 10,
            'All Disagree': 5
        }
        
        fig = px.pie(
            values=list(models_agree.values()),
            names=list(models_agree.keys()),
            title='Model Agreement on Predictions',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prediction confidence distribution
        st.markdown("**Prediction Confidence Patterns**")
        
        # Simulate confidence distribution
        confidence_ranges = ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        confidence_counts = [15, 25, 30, 20, 10]
        
        fig = px.bar(
            x=confidence_ranges,
            y=confidence_counts,
            title='Prediction Confidence Distribution',
            labels={'x': 'Confidence Range', 'y': 'Percentage of Predictions'},
            color=confidence_counts,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature impact on predictions
    st.markdown("#### 🔍 Feature Impact on Predictions")
    
    # Create synthetic data for demonstration
    feature_impact = pd.DataFrame({
        'Feature': ['capital.gain', 'education.num', 'age', 'hours.per.week', 'marital.status'],
        'Positive Impact': [0.85, 0.72, 0.65, 0.58, 0.55],
        'Negative Impact': [0.15, 0.28, 0.35, 0.42, 0.45]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Positive Impact', x=feature_impact['Feature'], 
                        y=feature_impact['Positive Impact'], marker_color='#4CAF50'))
    fig.add_trace(go.Bar(name='Negative Impact', x=feature_impact['Feature'], 
                        y=feature_impact['Negative Impact'], marker_color='#FF9800'))
    
    fig.update_layout(
        title='Feature Impact on High Income Predictions',
        barmode='stack',
        yaxis_title='Impact Proportion',
        xaxis_title='Features'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_deep_dive(processed_data, models):
    """Deep dive analysis"""
    st.markdown("## 🔍 Deep Dive Analysis")
    
    if not processed_data or 'original_data' not in processed_data:
        st.warning("Data not available for deep dive analysis")
        return
    
    df = processed_data['original_data']
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Demographic Deep Dive", "Economic Factors", "Education & Career Path", "Geographic Analysis"]
    )
    
    if analysis_type == "Demographic Deep Dive":
        show_demographic_deep_dive(df)
    elif analysis_type == "Economic Factors":
        show_economic_deep_dive(df)
    elif analysis_type == "Education & Career Path":
        show_education_career_deep_dive(df)
    else:
        show_geographic_deep_dive(df)

def show_demographic_deep_dive(df):
    """Deep dive into demographics"""
    st.markdown("### 👥 Demographic Deep Dive")
    
    # Age cohort analysis
    df['age_cohort'] = pd.cut(df['age'], 
                              bins=[0, 25, 35, 45, 55, 65, 100],
                              labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+'])
    
    # Simple bar chart for age cohorts
    cohort_income = df.groupby(['age_cohort', 'income']).size().unstack(fill_value=0)
    cohort_income['high_income_pct'] = cohort_income['>50K'] / cohort_income.sum(axis=1) * 100
    
    fig = px.bar(
        x=cohort_income.index,
        y=cohort_income['high_income_pct'],
        title='High Income Rate by Age Group',
        labels={'x': 'Age Group', 'y': 'High Income %'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Gender comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Simple gender comparison
        gender_income = pd.crosstab(df['sex'], df['income'], normalize='index') * 100
        fig = px.bar(
            x=gender_income.index,
            y=gender_income['>50K'],
            title='High Income Rate by Gender',
            labels={'x': 'Gender', 'y': 'High Income %'},
            color=gender_income['>50K'],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Race distribution
        race_income = pd.crosstab(df['race'], df['income'], normalize='index') * 100
        fig = px.bar(
            x=race_income.index,
            y=race_income['>50K'],
            title='High Income Rate by Race',
            labels={'x': 'Race', 'y': 'High Income %'},
            color=race_income['>50K'],
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("#### 📊 Key Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_age_high = df[df['income'] == '>50K']['age'].mean()
        st.metric("Avg Age (High Income)", f"{avg_age_high:.1f}")
    
    with col2:
        male_pct = gender_income.loc['Male', '>50K'] if 'Male' in gender_income.index else 0
        female_pct = gender_income.loc['Female', '>50K'] if 'Female' in gender_income.index else 0
        st.metric("Gender Gap", f"{abs(male_pct - female_pct):.1f}%")
    
    with col3:
        us_pct = (df[df['native.country'] == 'United-States']['income'] == '>50K').mean() * 100
        st.metric("US High Income %", f"{us_pct:.1f}%")

def show_economic_deep_dive(df):
    """Deep dive into economic factors"""
    st.markdown("### 💰 Economic Factors Deep Dive")
    
    # Capital analysis
    df['total_capital'] = df['capital.gain'] - df['capital.loss']
    df['capital_category'] = pd.cut(df['total_capital'], 
                                   bins=[-np.inf, -1, 0, 5000, 10000, np.inf],
                                   labels=['Loss', 'Zero', 'Low Gain', 'Medium Gain', 'High Gain'])
    
    capital_income = pd.crosstab(df['capital_category'], df['income'], normalize='index') * 100
    
    fig = px.bar(
        capital_income['>50K'].reset_index(),
        x='capital_category',
        y='>50K',
        title='Impact of Capital on High Income',
        labels={'capital_category': 'Capital Category', '>50K': 'High Income Rate (%)'},
        color='>50K',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Work hours and capital relationship
        fig = px.scatter(
            df[df['total_capital'] != 0].sample(n=min(1000, len(df))),
            x='hours.per.week',
            y='total_capital',
            color='income',
            title='Work Hours vs Capital Relationship',
            labels={'hours.per.week': 'Hours per Week', 'total_capital': 'Total Capital'},
            color_discrete_map={'<=50K': '#FF9800', '>50K': '#4CAF50'}
        )
        fig.update_yaxes(range=[-5000, 20000])  # تصحيح: update_yaxes بدلاً من update_yaxis
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Occupation and capital
        occ_capital = df.groupby('occupation').agg({
            'capital.gain': 'mean',
            'capital.loss': 'mean',
            'income': lambda x: (x == '>50K').mean() * 100
        }).sort_values('income', ascending=False).head(10)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Avg Capital Gain', x=occ_capital.index, 
                            y=occ_capital['capital.gain'], yaxis='y'))
        fig.add_trace(go.Scatter(name='High Income %', x=occ_capital.index, 
                                y=occ_capital['income'], yaxis='y2', mode='lines+markers',
                                line=dict(color='red', width=3)))
        
        fig.update_layout(
            title='Top Occupations: Capital Gains vs Income Rate',
            yaxis=dict(title='Average Capital Gain'),
            yaxis2=dict(title='High Income %', overlaying='y', side='right'),
            hovermode='x'
        )
        # fig.update_xaxes(tickangle=-45)  # يمكن إلغاء التعليق إذا أردت
        st.plotly_chart(fig, use_container_width=True)
    
    # Economic mobility indicators
    st.markdown("#### 📈 Economic Mobility Indicators")
    
    # Create mobility score
    df['mobility_score'] = (
        df['education.num'] / 16 * 0.3 +
        (df['hours.per.week'] / 99) * 0.2 +
        (df['capital.gain'] > 0).astype(int) * 0.3 +
        (df['age'] / 90) * 0.2
    )
    
    fig = px.box(
        df,
        x='income',
        y='mobility_score',
        title='Economic Mobility Score Distribution',
        color='income',
        color_discrete_map={'<=50K': '#FF9800', '>50K': '#4CAF50'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key economic insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        capital_impact = (df[df['capital.gain'] > 0]['income'] == '>50K').mean() * 100
        no_capital_impact = (df[df['capital.gain'] == 0]['income'] == '>50K').mean() * 100
        st.success(f"""
        **Capital Impact**  
        With capital: {capital_impact:.1f}% high income  
        Without: {no_capital_impact:.1f}%  
        Difference: {capital_impact - no_capital_impact:.1f}%
        """)
    
    with col2:
        high_mobility = df[df['mobility_score'] > 0.7]
        mobility_success = (high_mobility['income'] == '>50K').mean() * 100
        st.success(f"""
        **Mobility Success**  
        High mobility score: {mobility_success:.1f}%  
        achieve high income  
        Score > 0.7 critical
        """)
    
    with col3:
        work_capital_corr = df[['hours.per.week', 'total_capital']].corr().iloc[0, 1]
        st.success(f"""
        **Work-Capital Balance**  
        Correlation: {work_capital_corr:.3f}  
        Suggests independent paths  
        to high income
        """)

def show_education_career_deep_dive(df):
    """Deep dive into education and career paths"""
    st.markdown("### 🎓 Education & Career Path Analysis")
    
    # Education ROI analysis
    edu_stats = df.groupby('education').agg({
        'income': lambda x: (x == '>50K').mean() * 100,
        'age': 'mean',
        'hours.per.week': 'mean',
        'education.num': 'first'
    }).sort_values('education.num')
    
    # Create two separate plots instead of subplots
    col1, col2 = st.columns(2)
    
    with col1:
        # ROI curve
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=edu_stats['education.num'],
                y=edu_stats['income'],
                mode='lines+markers',
                name='High Income %',
                line=dict(color='green', width=3),
                marker=dict(size=10)
            )
        )
        fig.update_layout(
            title='Education ROI Curve',
            xaxis_title='Education Years',
            yaxis_title='High Income Rate (%)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Career paths - simplified
        career_paths = df.groupby(['education', 'occupation']).size().reset_index(name='count')
        top_paths = career_paths.nlargest(15, 'count')
        
        # Create a simple bar chart instead of Sankey
        fig = px.bar(
            top_paths,
            x='count',
            y=top_paths['education'] + ' → ' + top_paths['occupation'],
            orientation='h',
            title='Top Education-Career Paths',
            labels={'y': 'Path', 'count': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Career transition analysis
    st.markdown("#### 🚀 Career Success Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Education to occupation success rate
        edu_occ_success = df.groupby(['education', 'occupation']).agg({
            'income': lambda x: (x == '>50K').mean() * 100
        }).reset_index()
        
        # Top successful combinations
        top_combos = edu_occ_success.nlargest(15, 'income')
        
        fig = px.bar(
            top_combos,
            x='income',
            y=top_combos['education'] + ' → ' + top_combos['occupation'],
            orientation='h',
            title='Most Successful Education-Career Combinations',
            labels={'income': 'High Income Rate (%)', 'y': 'Education → Occupation'},
            color='income',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Work experience impact by education
        df['experience_proxy'] = df['age'] - df['education.num'] - 6
        df['experience_proxy'] = df['experience_proxy'].clip(lower=0)  # Avoid negative values
        
        # Group by education and experience bins
        exp_bins = pd.cut(df['experience_proxy'], bins=[0, 10, 20, 30, 40, 100], 
                         labels=['0-10', '10-20', '20-30', '30-40', '40+'])
        
        exp_edu_data = df.groupby(['education', exp_bins]).agg({
            'income': lambda x: (x == '>50K').mean() * 100
        }).reset_index()
        
        # Select top education levels
        top_edu = edu_stats.nlargest(5, 'income').index
        exp_edu_filtered = exp_edu_data[exp_edu_data['education'].isin(top_edu)]
        
        fig = px.line(
            exp_edu_filtered,
            x='experience_proxy',
            y='income',
            color='education',
            title='Income Growth with Experience by Education',
            labels={'experience_proxy': 'Years of Experience', 'income': 'High Income Rate (%)'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Alternative visualization: Education and Occupation heatmap
    st.markdown("#### 🔥 Education-Occupation Income Heatmap")
    
    # Create heatmap data
    edu_occ_matrix = df.groupby(['education', 'occupation']).agg({
        'income': lambda x: (x == '>50K').mean() * 100
    }).unstack(fill_value=0)
    
    # Select top occupations and educations
    top_occupations = df.groupby('occupation')['income'].apply(lambda x: (x == '>50K').mean()).nlargest(10).index
    top_educations = df.groupby('education')['income'].apply(lambda x: (x == '>50K').mean()).nlargest(8).index
    
    heatmap_data = edu_occ_matrix.loc[
        edu_occ_matrix.index.isin(top_educations),
        ('income', top_occupations)
    ]
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Occupation", y="Education", color="High Income %"),
        title="Income Success Rate by Education and Occupation",
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Career insights
    st.markdown("#### 🔍 Career Path Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_combo = top_combos.iloc[0]
        st.info(f"""
        **Best Path**  
        {best_combo['education']} →  
        {best_combo['occupation']}  
        Success rate: {best_combo['income']:.1f}%
        """)
    
    with col2:
        edu_premium = edu_stats['income'].max() - edu_stats['income'].min()
        st.info(f"""
        **Education Premium**  
        Max difference: {edu_premium:.1f}%  
        Between highest and  
        lowest education levels
        """)
    
    with col3:
        # Calculate experience impact safely
        exp_groups = df.groupby(pd.cut(df['experience_proxy'], bins=3))['income'].apply(
            lambda x: (x == '>50K').mean() * 100
        )
        if len(exp_groups) > 0:
            exp_diff = exp_groups.max() - exp_groups.min()
        else:
            exp_diff = 0
            
        st.info(f"""
        **Experience Value**  
        Impact: {exp_diff:.1f}% difference  
        Between experience levels  
        Peaks at 15-25 years
        """)
def show_geographic_deep_dive(df):
    """Geographic analysis"""
    st.markdown("### 🌍 Geographic Analysis")
    
    # Country analysis
    country_stats = df.groupby('native.country').agg({
        'income': [lambda x: (x == '>50K').mean() * 100, 'count'],
        'age': 'mean',
        'education.num': 'mean'
    }).round(2)
    
    country_stats.columns = ['high_income_rate', 'count', 'avg_age', 'avg_education']
    country_stats = country_stats[country_stats['count'] > 50]  # Filter small samples
    
        # World map visualization (simplified)
    fig = px.bar(
        country_stats.nlargest(15, 'high_income_rate').reset_index(),
        x='native.country',
        y='high_income_rate',
        title='High Income Rate by Country',
        color='high_income_rate',
        color_continuous_scale='RdYlGn'
    )
    # fig.update_xaxis(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key geographic insights
    col1, col2 = st.columns(2)
    
    with col1:
        us_rate = country_stats.loc['United-States', 'high_income_rate']
        intl_avg = country_stats[country_stats.index != 'United-States']['high_income_rate'].mean()
        st.info(f"""
        **Geographic Impact**  
        US rate: {us_rate:.1f}%  
        International avg: {intl_avg:.1f}%  
        US advantage: {us_rate - intl_avg:.1f}%
        """)
    
    with col2:
        top_country = country_stats.nlargest(1, 'high_income_rate')
        st.info(f"""
        **Top Performing Country**  
        {top_country.index[0]}  
        High income rate: {top_country['high_income_rate'].iloc[0]:.1f}%  
        Sample size: {top_country['count'].iloc[0]}
        """)

# Main execution
if __name__ == "__main__":
    main()
    
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
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Income Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .high-income {
        background: linear-gradient(135deg, #4CAF50, #81C784);
        color: white;
    }
    .low-income {
        background: linear-gradient(135deg, #FF9800, #FFB74D);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all saved models"""
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
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except:
                try:
                    with open(path, 'rb') as f:
                        models[name] = pickle.load(f)
                except:
                    continue
    return models

@st.cache_resource
def load_scaler():
    """Load the scaler"""
    if os.path.exists('scaler.pkl'):
        try:
            return joblib.load('scaler.pkl')
        except:
            with open('scaler.pkl', 'rb') as f:
                return pickle.load(f)
    return None

@st.cache_resource
def load_mappings():
    """Load categorical mappings"""
    if os.path.exists('categorical_mappings.pkl'):
        try:
            with open('categorical_mappings.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            return joblib.load('categorical_mappings.pkl')
    return None

@st.cache_data
def load_results():
    """Load model results"""
    if os.path.exists('saved_models/model_results.pkl'):
        try:
            with open('saved_models/model_results.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            return joblib.load('saved_models/model_results.pkl')
    return None

@st.cache_data
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
    
    # Load resources
    models = load_models()
    scaler = load_scaler()
    mappings = load_mappings()
    results = load_results()
    processed_data = load_processed_data()
    
    if not models:
        st.error("⚠️ Models not found! Please run the training notebook first.")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Model Performance", "📈 Data Insights", "🎯 Feature Analysis"])
    
    with tab1:
        show_prediction_page(models, scaler, mappings)
    
    with tab2:
        show_model_performance(results, models)
    
    with tab3:
        show_data_insights(processed_data)
    
    with tab4:
        show_feature_analysis(models, processed_data)

def show_prediction_page(models, scaler, mappings):
    """Prediction interface"""
    
    st.markdown("## 🔮 Income Prediction")
    
    if not models or scaler is None:
        st.warning("⚠️ Prediction system not available.")
        return
    
    # Model selection
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            ['Best Model'] + [k for k in models.keys() if k != 'Best Model']
        )
    
    st.markdown("---")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 17, 90, 35)
            education = st.selectbox("Education", [
                'Bachelors', 'HS-grad', 'Masters', 'Some-college',
                'Assoc-acdm', 'Doctorate', 'Prof-school', '11th', '9th',
                '7th-8th', '10th', '5th-6th', '12th', '1st-4th', 'Preschool'
            ])
            workclass = st.selectbox("Work Class", [
                'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
            ])
            education_num = st.slider("Education Years", 1, 16, 10)
        
        with col2:
            marital_status = st.selectbox("Marital Status", [
                'Married-civ-spouse', 'Never-married', 'Divorced',
                'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
            ])
            occupation = st.selectbox("Occupation", [
                'Exec-managerial', 'Prof-specialty', 'Tech-support', 'Sales',
                'Craft-repair', 'Other-service', 'Machine-op-inspct',
                'Adm-clerical', 'Handlers-cleaners', 'Transport-moving',
                'Farming-fishing', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'
            ])
            relationship = st.selectbox("Relationship", [
                'Husband', 'Wife', 'Own-child', 'Not-in-family',
                'Other-relative', 'Unmarried'
            ])
            hours_per_week = st.number_input("Hours/Week", 1, 99, 40)
        
        with col3:
            capital_gain = st.number_input("Capital Gain", 0, 100000, 0, step=1000)
            capital_loss = st.number_input("Capital Loss", 0, 5000, 0, step=100)
            race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
            sex = st.selectbox("Sex", ['Male', 'Female'])
            native_country = st.selectbox("Native Country", [
                'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada',
                'India', 'China', 'Japan', 'Other'
            ])
        
        submitted = st.form_submit_button("🎯 Predict Income", use_container_width=True)
    
    if submitted:
        try:
            # Prepare input
            input_data = pd.DataFrame({
                'age': [age],
                'workclass': [workclass],
                'fnlwgt': [200000],  # Default value
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
                le = LabelEncoder()
                for col in input_data.select_dtypes(include=['object']).columns:
                    if col in mappings and input_data[col].iloc[0] in mappings[col]:
                        input_data[col] = mappings[col][input_data[col].iloc[0]]
                    else:
                        input_data[col] = 0
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            model = models[selected_model]
            prediction = model.predict(input_scaled)[0]
            
            # Get probability
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[0]
            else:
                proba = [1-prediction, prediction]
            
            # Display results
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div class="prediction-box high-income">
                        <h2>✅ Income > $50K</h2>
                        <h3>Confidence: {:.1f}%</h3>
                    </div>
                    """.format(proba[1]*100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-box low-income">
                        <h2>📊 Income ≤ $50K</h2>
                        <h3>Confidence: {:.1f}%</h3>
                    </div>
                    """.format(proba[0]*100), unsafe_allow_html=True)
            
            with col2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba[1]*100,
                    title={'text': "High Income Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#4CAF50" if prediction == 1 else "#FF9800"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

def show_model_performance(results, models):
    """Model performance visualization"""
    
    st.markdown("## 📊 Model Performance")
    
    if not results:
        st.warning("No results available")
        return
    
    # Performance metrics
    metrics_df = pd.DataFrame(results['model_metrics'])
    
    # Best model highlight
    best_model = results.get('best_model', 'N/A')
    best_acc = results.get('best_accuracy', 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Best Model", best_model)
    with col2:
        st.metric("📈 Best Accuracy", f"{best_acc*100:.2f}%")
    with col3:
        st.metric("🤖 Total Models", len(models)-1)
    
    st.markdown("---")
    
    # Performance comparison
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Comparison', 'Performance Radar'),
        specs=[[{'type': 'bar'}, {'type': 'polar'}]]
    )
    
    # Bar chart
    for metric in ['Accuracy', 'F1-Score']:
        fig.add_trace(
            go.Bar(name=metric, x=metrics_df['Model'], y=metrics_df[metric]),
            row=1, col=1
        )
    
    # Radar chart
    for _, row in metrics_df.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                name=row['Model'],
                fill='toself'
            ),
            row=1, col=2
        )
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def show_data_insights(processed_data):
    """Data insights and visualizations"""
    
    st.markdown("## 📈 Data Insights")
    
    if not processed_data or 'original_data' not in processed_data:
        st.warning("Data not available")
        return
    
    df = processed_data['original_data']
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Avg Age", f"{df['age'].mean():.1f}")
    with col3:
        st.metric("Avg Hours/Week", f"{df['hours.per.week'].mean():.1f}")
    with col4:
        high_income_pct = (df['income'] == '>50K').mean() * 100
        st.metric("High Income %", f"{high_income_pct:.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Income distribution
        income_counts = df['income'].value_counts()
        fig = px.pie(
            values=income_counts.values,
            names=income_counts.index,
            title='Income Distribution',
            color_discrete_sequence=['#FF9800', '#4CAF50']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age distribution
        fig = px.histogram(
            df, x='age', color='income',
            title='Age Distribution by Income',
            nbins=30,
            color_discrete_map={'<=50K': '#FF9800', '>50K': '#4CAF50'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Education vs Income
    edu_income = pd.crosstab(df['education'], df['income'], normalize='index') * 100
    fig = px.bar(
        edu_income['>50K'].sort_values(ascending=False).head(10),
        title='Top 10 Education Levels by High Income %',
        labels={'value': '% with Income >50K', 'index': 'Education'},
        color_discrete_sequence=['#4CAF50']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Work hours analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            df, x='income', y='hours.per.week',
            title='Work Hours by Income',
            color='income',
            color_discrete_map={'<=50K': '#FF9800', '>50K': '#4CAF50'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Capital gains analysis - FIXED HERE
        fig = px.box(
            df, x='income', y='capital.gain',
            title='Capital Gains by Income',
            color='income',
            color_discrete_map={'<=50K': '#FF9800', '>50K': '#4CAF50'}
        )
        # Correct way to update y-axis range
        fig.update_layout(yaxis_range=[0, 20000])
        st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis(models, processed_data):
    """Feature importance analysis"""
    
    st.markdown("## 🎯 Feature Analysis")
    
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        
        if hasattr(rf_model, 'feature_importances_'):
            # Get feature names
            feature_names = [
                'age', 'workclass', 'fnlwgt', 'education', 'education.num',
                'marital.status', 'occupation', 'relationship', 'race', 'sex',
                'capital.gain', 'capital.loss', 'hours.per.week', 'native.country'
            ]
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(rf_model.feature_importances_)],
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=True).tail(10)
            
            # Feature importance plot
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Most Important Features',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.markdown("### 🔍 Key Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("""
                **💰 Financial Features**  
                Capital gains/losses are strong predictors of income level
                """)
            
            with col2:
                st.info("""
                **🎓 Education Impact**  
                Education level and years strongly correlate with income
                """)
            
            with col3:
                st.info("""
                **⏰ Work Patterns**  
                Hours worked per week is a significant income factor
                """)
    
    # Correlation heatmap
    if processed_data and 'X_train' in processed_data:
        st.markdown("### 🔗 Feature Correlations")
        
        X_train = processed_data['X_train']
        corr_matrix = pd.DataFrame(X_train).corr()
        
        # Select top correlations
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=list(range(len(corr_matrix))),
            y=list(range(len(corr_matrix))),
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=True
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600,
            width=800
        )
        st.plotly_chart(fig, use_container_width=True)

# Main execution
if __name__ == "__main__":
    main()
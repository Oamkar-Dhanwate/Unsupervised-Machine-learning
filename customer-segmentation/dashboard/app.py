import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .cluster-0 { border-left-color: #ff6b6b !important; }
    .cluster-1 { border-left-color: #4ecdc4 !important; }
    .cluster-2 { border-left-color: #45b7d1 !important; }
    .cluster-3 { border-left-color: #96ceb4 !important; }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# --- Caching: Load Models & Preprocessors ---
@st.cache_resource
def load_models_and_preprocessor():
    """
    Loads the saved models and replicates the data preparation 
    (including fitting the PCA) from the notebook.
    """
    
    # Load Original Data for PCA fitting
    data_path = 'data/customer_segmentation.csv'
    if not os.path.exists(data_path):
        st.error(f"Error: `data/customer_segmentation.csv` not found.")
        st.error("Please create a `data` folder in the same directory as `app.py` and place the CSV file inside it.")
        return None, None, None, None

    df = pd.read_csv(data_path)

    # Replicate Feature Engineering & Cleaning
    df.dropna(inplace=True)
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    df['Age'] = 2025 - df['Year_Birth']
    
    # Calculate total_amount
    df['totat_amount'] = df[['MntWines', 'MntFruits',
                           'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                           'MntGoldProds']].sum(axis=1)
    
    # Define Feature Set & Fit PCA
    feature_cols = ['Age', 'Income', 'totat_amount', 'NumWebPurchases', 
                    'NumStorePurchases', 'NumWebVisitsMonth', 'Recency']
    
    X_train_full = df[feature_cols]
    
    # Initialize and fit the PCA object
    pca = PCA()
    pca.fit(X_train_full)
    
    # Load the trained models
    xgb_path = 'model/xgboost_pipeline.pkl'
    kmeans_path = 'model/kmeans_clustering.pkl'

    if not os.path.exists(xgb_path) or not os.path.exists(kmeans_path):
        st.error("Error: Model files not found.")
        st.error("Please create a `model` folder and ensure `xgboost_pipeline.pkl` and `kmeans_clustering.pkl` are inside.")
        return None, None, None, None
        
    xgb_model = joblib.load(xgb_path)
    kmeans_model = joblib.load(kmeans_path)

    return pca, xgb_model, kmeans_model, df

# Load models and data
pca, xgb_model, kmeans_model, original_df = load_models_and_preprocessor()

# --- App Header ---
st.markdown('<h1 class="main-header">👥 Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Analyze customer profiles, predict campaign responses, and uncover valuable insights for targeted marketing strategies.")

# --- Sidebar for User Inputs ---
st.sidebar.header("🎯 Customer Profile Input")

# Create tabs in sidebar for better organization
tab1, tab2 = st.sidebar.tabs(["👤 Personal Details", "🛒 Behavioral Data"])

with tab1:
    st.subheader("Demographic Information")
    year_birth = st.slider("Year of Birth", 1940, 2000, 1980)
    income = st.slider("Annual Income ", 0, 200000, 50000, 1000)
    
    # Calculate and display age automatically
    age = 2025 - year_birth
    st.metric("Calculated Age", f"{age} years")

with tab2:
    st.subheader("Spending Habits (Last 2 Years)")
    
    col1, col2 = st.columns(2)
    with col1:
        mnt_wines = st.number_input("Wine Spending", 0, 2000, 200)
        mnt_meat = st.number_input("Meat Spending", 0, 2000, 100)
        mnt_sweet = st.number_input("Sweets Spending", 0, 1000, 50)
    with col2:
        mnt_fruits = st.number_input("Fruits Spending", 0, 500, 20)
        mnt_fish = st.number_input("Fish Spending", 0, 500, 30)
        mnt_gold = st.number_input("Gold Spending", 0, 1000, 40)
    
    st.subheader("Purchase Behavior")
    num_web_purchases = st.slider("Web Purchases", 0, 30, 4)
    num_store_purchases = st.slider("Store Purchases", 0, 30, 6)
    num_web_visits_month = st.slider("Monthly Web Visits", 0, 20, 5)
    recency = st.slider("Days Since Last Purchase", 0, 100, 30)

# Quick stats in sidebar
st.sidebar.markdown("---")
total_amount = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweet + mnt_gold
st.sidebar.metric("Total Spending", f"Rs {total_amount:,.0f}")

# --- Prediction Logic ---
if st.sidebar.button("🚀 Analyze Customer Profile", type="primary", use_container_width=True):
    
    # Apply Feature Engineering
    age = 2025 - year_birth 
    total_amount = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweet + mnt_gold
    
    # Create Input DataFrame for PCA
    input_data = {
        'Age': [age],
        'Income': [income],
        'totat_amount': [total_amount],
        'NumWebPurchases': [num_web_purchases],
        'NumStorePurchases': [num_store_purchases],
        'NumWebVisitsMonth': [num_web_visits_month],
        'Recency': [recency]
    }
    input_df = pd.DataFrame(input_data)
    
    # Apply PCA Transformation
    input_scaled = pca.transform(input_df)
    
    # Make Predictions
    cluster_pred = kmeans_model.predict(input_scaled)[0]
    response_pred = xgb_model.predict(input_scaled)[0]
    response_proba = xgb_model.predict_proba(input_scaled)[0]
    
    # --- Display Results ---
    st.markdown("---")
    
    # Header with results
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("📊 Customer Analysis Results")
    
    with col2:
        cluster_color = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4"][cluster_pred]
        st.markdown(f'<div class="metric-card cluster-{cluster_pred}">', unsafe_allow_html=True)
        st.metric("Customer Segment", f"Cluster {cluster_pred}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        response_label = "LIKELY" if response_pred == 1 else "UNLIKELY"
        response_color = "#2ecc71" if response_pred == 1 else "#e74c3c"
        st.markdown(f'<div class="metric-card" style="border-left-color: {response_color}">', unsafe_allow_html=True)
        st.metric("Campaign Response", response_label)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Detailed Analysis Section ---
    st.markdown("---")
    
    # Cluster Description
    cluster_descriptions = {
        0: {
            "name": "Budget-Conscious Shoppers",
            "description": "Lower income customers with minimal spending across categories. Prefer value-based purchases.",
            "strategy": "Target with discount campaigns, loyalty programs, and budget-friendly offers.",
            "color": "#ff6b6b"
        },
        1: {
            "name": "Premium Lifestyle",
            "description": "High-income customers with substantial spending, particularly on wines and premium products.",
            "strategy": "Engage with exclusive offers, premium product launches, and personalized service.",
            "color": "#4ecdc4"
        },
        2: {
            "name": "Balanced Spenders",
            "description": "Middle-income customers with moderate, balanced spending across all categories.",
            "strategy": "Use cross-selling opportunities and moderate-value promotions.",
            "color": "#45b7d1"
        },
        3: {
            "name": "Elite Customers",
            "description": "Very high-income customers with exceptional spending patterns. Often early adopters.",
            "strategy": "Provide VIP treatment, early access to new products, and personalized concierge service.",
            "color": "#96ceb4"
        }
    }
    
    current_cluster = cluster_descriptions[cluster_pred]
    
    # Create columns for cluster info and response probability
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🎯 Segment Analysis: {current_cluster['name']}")
        st.info(f"**Profile:** {current_cluster['description']}")
        st.success(f"**Recommended Strategy:** {current_cluster['strategy']}")
        
        # Spending breakdown
        st.subheader("💰 Spending Distribution")
        spending_data = {
            'Category': ['Wine', 'Meat', 'Fruits', 'Fish', 'Sweets', 'Gold'],
            'Amount': [mnt_wines, mnt_meat, mnt_fruits, mnt_fish, mnt_sweet, mnt_gold]
        }
        spending_df = pd.DataFrame(spending_data)
        
        fig_spending = px.pie(spending_df, values='Amount', names='Category', 
                             color_discrete_sequence=px.colors.sequential.RdBu)
        fig_spending.update_layout(height=300)
        st.plotly_chart(fig_spending, use_container_width=True)
    
    with col2:
        st.subheader("📈 Response Probability")
        
        # Probability gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = response_proba[1] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Response Likelihood"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2ecc71" if response_pred == 1 else "#e74c3c"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Probability breakdown
        st.metric("Probability to Respond", f"{response_proba[1]*100:.1f}%")
        st.metric("Probability to Decline", f"{response_proba[0]*100:.1f}%")
    
    # --- Customer Comparison Analysis ---
    st.markdown("---")
    st.subheader("📊 Comparison with Customer Base")
    
    # Calculate percentiles for the current customer
    if original_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        income_percentile = (original_df['Income'] < income).mean() * 100
        total_spend_percentile = (original_df['totat_amount'] < total_amount).mean() * 100
        age_percentile = (original_df['Age'] < age).mean() * 100
        web_visits_percentile = (original_df['NumWebVisitsMonth'] < num_web_visits_month).mean() * 100
        
        with col1:
            st.metric("Income Percentile", f"{income_percentile:.1f}%")
        with col2:
            st.metric("Spending Percentile", f"{total_spend_percentile:.1f}%")
        with col3:
            st.metric("Age Percentile", f"{age_percentile:.1f}%")
        with col4:
            st.metric("Web Engagement Percentile", f"{web_visits_percentile:.1f}%")
    
    # --- Feature Importance Visualization ---
    st.markdown("---")
    st.subheader("🔍 Feature Impact Analysis")
    
    # Create a radar chart for customer profile
    categories = ['Income', 'Total Spend', 'Web Purchases', 'Store Purchases', 'Web Visits', 'Recency']
    
    # Normalize values for radar chart (0-1 scale)
    max_values = [200000, 5000, 30, 30, 20, 100]  # Approximate max values
    normalized_values = [
        income / max_values[0],
        total_amount / max_values[1],
        num_web_purchases / max_values[2],
        num_store_purchases / max_values[3],
        num_web_visits_month / max_values[4],
        1 - (recency / max_values[5])  # Invert recency (lower is better)
    ]
    
    # Complete the circle
    categories = categories + [categories[0]]
    normalized_values = normalized_values + [normalized_values[0]]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        name='Customer Profile',
        line_color=current_cluster['color']
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # --- Recommendations Section ---
    st.markdown("---")
    st.subheader("🎯 Marketing Recommendations")
    
    recommendations = []
    
    # Based on cluster
    recommendations.append(f"**Segment Strategy:** {current_cluster['strategy']}")
    
    # Based on spending patterns
    if mnt_wines > total_amount * 0.4:
        recommendations.append("**Wine Enthusiast:** Target with premium wine offers and wine-tasting events")
    if mnt_meat + mnt_fish > total_amount * 0.3:
        recommendations.append("**Protein Focused:** Promote high-quality meat and fish products")
    if mnt_gold > total_amount * 0.2:
        recommendations.append("**Luxury Buyer:** Introduce premium and luxury product lines")
    
    # Based on behavior
    if num_web_purchases > num_store_purchases:
        recommendations.append("**Online Preferred:** Focus on digital marketing and online-exclusive offers")
    else:
        recommendations.append("**Store Preferred:** Use in-store promotions and loyalty programs")
    
    if recency < 30:
        recommendations.append("**Recent Customer:** Engage with follow-up offers and referral programs")
    else:
        recommendations.append("**At-Risk Customer:** Reactivate with special comeback offers")
    
    # Based on response probability
    if response_proba[1] > 0.7:
        recommendations.append("**High-Value Prospect:** Prioritize for premium campaign outreach")
    elif response_proba[1] < 0.3:
        recommendations.append("**Low Probability:** Test different messaging or offer types")
    
    # Display recommendations
    for i, recommendation in enumerate(recommendations, 1):
        st.write(f"{i}. {recommendation}")
    
    # --- Raw Data Section ---
    with st.expander("📋 View Technical Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Processed Input Features")
            st.dataframe(input_df.style.format({"Income": "Rs {:,.0f}", "totat_amount": "Rs {:,.0f}"}))
        
        with col2:
            st.subheader("PCA Transformed Data")
            pca_df = pd.DataFrame(input_scaled, columns=[f'PC{i+1}' for i in range(input_scaled.shape[1])])
            st.dataframe(pca_df)

else:
    # Default state - show overview when no prediction is made
    st.markdown("""
    ## 🎯 Welcome to Customer Intelligence Dashboard
    
    This tool helps you:
    
    - **Segment customers** into meaningful groups based on their behavior and demographics
    - **Predict campaign responses** using machine learning models
    - **Generate actionable insights** for targeted marketing strategies
    - **Compare customer profiles** against your existing customer base
    
    ### 🚀 Getting Started:
    1. Fill in the customer details in the sidebar
    2. Click **"Analyze Customer Profile"** to generate insights
    3. Explore the comprehensive analysis and recommendations
    
    ### 📊 What You'll Discover:
    - Customer segmentation and profile analysis
    - Campaign response probability
    - Spending pattern insights
    - Personalized marketing recommendations
    - Comparative analysis with your customer base
    """)
    
    # Show sample statistics if data is loaded
    if original_df is not None:
        st.markdown("---")
        st.subheader("📈 Customer Base Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(original_df):,}")
        with col2:
            st.metric("Average Income", f"Rs {original_df['Income'].mean():,.0f}")
        with col3:
            st.metric("Average Age", f"{original_df['Age'].mean():.0f} years")
        with col4:
            st.metric("Avg Total Spending", f"Rs {original_df['totat_amount'].mean():,.0f}")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Customer Intelligence Dashboard • Powered by Machine Learning • 
        <a href='https://streamlit.io/' target='_blank'>Built with Streamlit</a>
    </div>
    """, 
    unsafe_allow_html=True
)
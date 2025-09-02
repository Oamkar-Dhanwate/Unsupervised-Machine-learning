import streamlit as st
import pandas as pd
import joblib
import os
import pycountry
import plotly.express as px 
from pathlib import Path # ✅ NEW: Import the Path library

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Supply Chain Dashboard",
    page_icon="🚚",
    layout="wide"
)

# --- BUILD ABSOLUTE PATHS ---
# ✅ NEW: Get the path to the root of your project
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_data_with_segments.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "delivery_risk_pipeline.joblib"


# --- DATA AND MODEL LOADING ---
@st.cache_data
def load_data():
    """Loads the final dataset with features and segments."""
    # ✅ CHANGED: Use the absolute path
    df = pd.read_csv(DATA_PATH, parse_dates=['order_date_dateorders'])
    return df

@st.cache_resource
def load_prediction_pipeline():
    """Loads the saved prediction pipeline."""
    # ✅ CHANGED: Use the absolute path
    pipeline = joblib.load(MODEL_PATH)
    return pipeline

@st.cache_resource
def load_forecast_models():
    """Loads all regional demand forecast models."""
    models = {}
    models_path = PROJECT_ROOT / "models"
    for filename in os.listdir(models_path):
        if filename.startswith("demand_forecaster_"):
            region = filename.replace("demand_forecaster_", "").replace(".pkl", "").replace("_", " ")
            models[region] = joblib.load(models_path / filename)
    return models

# ... inside the loading section ...
forecast_models = load_forecast_models()
# Load all assets
df = load_data()
prediction_pipeline = load_prediction_pipeline()

# At the top of your app, after loading the data
min_date = df['order_date_dateorders'].min().date()
max_date = df['order_date_dateorders'].max().date()
st.info(f"Full dataset ranges from {min_date} to {max_date}")

# ... (after df = load_data()) ...

# --- SIDEBAR FILTERS ---
st.sidebar.header("Dashboard Filters")

# Filter by Region
selected_region = st.sidebar.multiselect(
    "Filter by Region",
    options=df['order_region'].unique(),
    default=df['order_region'].unique()
)

# Filter by Shipping Mode
selected_shipping = st.sidebar.multiselect(
    "Filter by Shipping Mode",
    options=df['shipping_mode'].unique(),
    default=df['shipping_mode'].unique()
)

# In the SIDEBAR FILTERS section
start_date = st.sidebar.date_input("Start Date", df['order_date_dateorders'].min())
end_date = st.sidebar.date_input("End Date", df['order_date_dateorders'].max())

# Then, add this to your df_filtered logic
df_filtered = df[
    (df['order_date_dateorders'] >= pd.to_datetime(start_date)) &
    (df['order_date_dateorders'] <= pd.to_datetime(end_date)) &
    df['order_region'].isin(selected_region) &
    df['shipping_mode'].isin(selected_shipping)
]

# # Apply filters to the dataframe
# df_filtered = df[
#     df['order_region'].isin(selected_region) &
#     df['shipping_mode'].isin(selected_shipping)
# ]

# --- HEADER ---
st.title("Supply Chain Optimization & Risk Management Dashboard 🚚")

st.write("This dashboard provides an overview of supply chain performance, customer segments, and demand forecasts, along with a tool to predict late delivery risk for new orders.")

# --- Add a separator
st.markdown("---")

# The rest of the dashboard code will go here...

# --- KPIs ---
st.header("Key Performance Indicators")

# Calculate KPIs
otif_rate = 1 - df_filtered['late_delivery_risk'].mean()
perfect_order_rate = df_filtered['is_perfect_order'].mean()
avg_real_days = df_filtered['days_for_shipping_real'].mean()
avg_scheduled_days = df_filtered['days_for_shipment_scheduled'].mean()

# Display KPIs in columns
col1, col2, col3, col4 = st.columns(4)
col1.metric("On-Time-In-Full (OTIF) Rate", f"{otif_rate:.2%}")
col2.metric("Perfect Order Rate", f"{perfect_order_rate:.2%}")
col3.metric("Avg. Real Shipping Days", f"{avg_real_days:.2f}")
col4.metric("Avg. Scheduled Shipping Days", f"{avg_scheduled_days:.2f}")

st.markdown("---")

# --- DATA EXPORTER ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string."""
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtered)

st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_supply_chain_data.csv',
    mime='text/csv',
)

# --- ANALYTICS CHARTS ---
st.header("Supply Chain Analytics")

# Create two columns for charts
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.subheader("Late Delivery Risk by Shipping Mode")
    
    # Calculate late risk by shipping mode
    risk_by_shipping_mode = df_filtered.groupby('shipping_mode')['late_delivery_risk'].mean().sort_values(ascending=False)
    
    # Create a bar chart with Plotly Express
    fig = px.bar(
        risk_by_shipping_mode,
        x=risk_by_shipping_mode.index,
        y='late_delivery_risk',
        title="Late Delivery Risk per Shipping Mode",
        labels={'late_delivery_risk': 'Late Risk %', 'shipping_mode': 'Shipping Mode'},
        color='late_delivery_risk',
        color_continuous_scale=px.colors.sequential.YlOrRd
    )
    fig.update_layout(yaxis_tickformat='.2%')
    st.plotly_chart(fig, use_container_width=True)

# The code for the second chart will go in fig_col2...



# # --- ANALYTICS CHARTS ---
# st.header("Supply Chain Analytics")
# # ... (code for fig_col1) ...

import pycountry # ✅ NEW IMPORT

# ... (inside your app.py) ...

with fig_col2:
    st.subheader("Geographical Late Delivery Risk")

    # Group data by country
    risk_by_country = df_filtered.groupby('customer_country')['late_delivery_risk'].mean().reset_index()

    # ✅ NEW: Function to get ISO alpha-3 codes
    def get_iso_alpha(country_name):
        try:
            return pycountry.countries.get(name=country_name).alpha_3
        except AttributeError:
            # Handle special cases or names not found in the library
            if country_name == "EE. UU.": # Spanish for USA
                return "USA"
            return None # Return None for countries not found

    # Apply the conversion
    risk_by_country['iso_alpha'] = risk_by_country['customer_country'].apply(get_iso_alpha)

    # Create the choropleth map using ISO codes
    fig = px.choropleth(
        risk_by_country.dropna(subset=['iso_alpha']), # Drop countries we couldn't find
        locations="iso_alpha", # ✅ CHANGED: Use the new ISO code column
        locationmode='ISO-3',  # ✅ CHANGED: Tell Plotly we're using ISO-3 codes
        color="late_delivery_risk",
        hover_name="customer_country",
        color_continuous_scale=px.colors.sequential.YlOrRd,
        title="Late Delivery Risk % by Country"
    )
    fig.update_layout(geo=dict(showcoastlines=True))
    st.plotly_chart(fig, use_container_width=True)

# --- CATEGORY PROFITABILITY VS RISK ---
st.markdown("---")
st.header("Category Performance")
st.subheader("Profitability vs. Risk Scatter Plot")

# Group data by category and calculate total profit and average risk
category_analysis = df_filtered.groupby('category_name').agg(
    total_profit=('benefit_per_order', 'sum'),
    avg_late_risk=('late_delivery_risk', 'mean')
).reset_index()

# Create the scatter plot
fig_scatter = px.scatter(
    category_analysis,
    x="total_profit",
    y="avg_late_risk",
    size="total_profit", # Optional: make bubble size proportional to profit
    color="avg_late_risk",
    hover_name="category_name",
    color_continuous_scale=px.colors.sequential.YlOrRd,
    labels={
        "total_profit": "Total Profit ($)",
        "avg_late_risk": "Average Late Delivery Risk"
    },
    title="Product Category Performance: Profit vs. Risk"
)
fig_scatter.update_layout(yaxis_tickformat='.2%')
st.plotly_chart(fig_scatter, use_container_width=True)


# --- DEMAND FORECAST ---
st.markdown("---")
# ... (The rest of your demand forecast code follows) ...

    # --- ML PREDICTION TOOL ---
st.sidebar.markdown("---") # Adds a visual separator
st.sidebar.header("📦 Predict Late Delivery Risk")
st.sidebar.write("Enter new order details to get a risk prediction.")

# Create input widgets for model features
order_date = st.sidebar.date_input("Order Date")
days_scheduled = st.sidebar.slider("Days for Shipment (Scheduled)", min_value=0, max_value=10, value=4)
sales = st.sidebar.number_input("Sales per Order ($)", min_value=0.0, value=250.0)
benefit = st.sidebar.number_input("Benefit per Order ($)", value=50.0)

# Dropdowns with unique values from the dataframe
shipping_mode = st.sidebar.selectbox("Shipping Mode", options=df['shipping_mode'].unique())
customer_segment = st.sidebar.selectbox("Customer Segment", options=df['customer_segment'].unique())
market = st.sidebar.selectbox("Market", options=df['market'].unique())
category = st.sidebar.selectbox("Product Category", options=df['category_name'].unique())
order_region = st.sidebar.selectbox("Order Region", options=df['order_region'].unique())

# Prediction button
if st.sidebar.button("Predict Risk"):
    
    # Create a single-row DataFrame from the inputs
    input_df = pd.DataFrame({
        'days_for_shipment_scheduled': [days_scheduled],
        'benefit_per_order': [benefit],
        'sales_per_customer': [sales],
        'category_name': [category],
        'customer_segment': [customer_segment],
        'market': [market],
        'order_region': [order_region],
        'shipping_mode': [shipping_mode],
        'order_month': [order_date.month],
        'order_weekday': [order_date.weekday()]
    })

    # Get the prediction probability
    prediction_proba = prediction_pipeline.predict_proba(input_df)[0][1] # Probability of class 1 (late)

    # Display the result
    st.sidebar.subheader("Prediction Result")
    risk_percent = prediction_proba * 100
    
    # Use a metric with a color-coded delta to show risk level
    if risk_percent >= 50:
        st.sidebar.metric(
            label="Risk of Late Delivery",
            value=f"{risk_percent:.2f}%",
            delta="High Risk",
            delta_color="inverse"
        )
    else:
        st.sidebar.metric(
            label="Risk of Late Delivery",
            value=f"{risk_percent:.2f}%",
            delta="Low Risk",
            delta_color="normal"
        )

 # --- ✅ NEW: Add Prediction Reasons ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Top Factors Influencing Prediction:")

    # 1. Get feature importances from the model inside the pipeline
    feature_importances = prediction_pipeline.named_steps['classifier'].feature_importances_

    # 2. Get feature names from the preprocessor step in the pipeline
    feature_names = prediction_pipeline.named_steps['preprocessor'].get_feature_names_out()

    # 3. Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)

    # 4. Display the top 5 factors
    st.sidebar.dataframe(
        importance_df.head(5),
        column_config={
            "feature": "Factor",
            "importance": st.column_config.ProgressColumn(
                "Importance",
                format="%.3f",
                min_value=0,
                max_value=importance_df['importance'].max(),
            ),
        },
        hide_index=True
    )

# --- (This code goes inside the ML PREDICTION TOOL section in the sidebar) ---

# Recommendation button
if st.sidebar.button("Recommend Optimal Shipping Mode"):
    
    # Define a simple cost/benefit proxy for each mode
    shipping_mode_profitability = df.groupby('shipping_mode')['benefit_per_order'].mean()

    recommendations = []
    
    # Loop through every available shipping mode
    for mode in df['shipping_mode'].unique():
        
        # Create an input DataFrame for the current mode
        input_df = pd.DataFrame({
            'days_for_shipment_scheduled': [days_scheduled],
            'benefit_per_order': [benefit],
            'sales_per_customer': [sales],
            'category_name': [category],
            'customer_segment': [customer_segment],
            'market': [market],
            'order_region': [order_region],
            'shipping_mode': [mode],
            'order_month': [order_date.month],
            'order_weekday': [order_date.weekday()]
        })
        
        # Get the prediction probability
        prediction_proba = prediction_pipeline.predict_proba(input_df)[0][1]
        
        # Store the mode, its risk, and its profitability
        recommendations.append({
            "mode": mode,
            "risk": prediction_proba,
            "profitability": shipping_mode_profitability.get(mode, 0)
        })

    # Convert to a DataFrame for easy filtering
    reco_df = pd.DataFrame(recommendations)
    
    # ✅ 1. DEBUG: Display the predicted risks for all modes (you can remove this later)
    st.sidebar.write("Debug: Predicted Risks")
    st.sidebar.dataframe(reco_df)
    
    # --- Apply the Feasibility & Optimality Logic ---
    st.sidebar.subheader("Recommendation Result")
    
    # ✅ 2. ADJUST: Find all modes with risk below a more realistic threshold
    feasible_options = reco_df[reco_df['risk'] < 0.35] # Increased threshold from 0.25 to 0.35
    
    if not feasible_options.empty:
        # Optimality: Find the one with the highest profitability
        best_option = feasible_options.loc[feasible_options['profitability'].idxmax()]
        
        st.sidebar.success(f"**Recommended Mode:** {best_option['mode']}")
        st.sidebar.write(f"This option has a low late delivery risk of **{best_option['risk']:.2%}**.")
    else:
        # If no option is safe, recommend the one with the lowest risk
        lowest_risk_option = reco_df.loc[reco_df['risk'].idxmin()]
        st.sidebar.warning(f"**No low-risk option found.**")
        st.sidebar.info(f"The safest available option is **{lowest_risk_option['mode']}** with a late delivery risk of **{lowest_risk_option['risk']:.2%}**.")


        # --- DEMAND FORECAST ---
st.markdown("---")
st.header("Demand Forecast by Region")

# Allow user to select a region
selected_forecast_region = st.selectbox(
    "Select a Region to Forecast",
    options=list(forecast_models.keys())
)

if selected_forecast_region:
    # Load the correct model
    model = forecast_models[selected_forecast_region]

    # Make future predictions
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    # Display the forecast plot
    st.subheader(f"90-Day Demand Forecast for {selected_forecast_region}")
    fig = model.plot(forecast)
    st.pyplot(fig)
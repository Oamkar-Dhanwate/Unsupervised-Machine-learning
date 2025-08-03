from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np

# Initialize Flask app
app = Flask(__name__, template_folder='.')

# Load the pre-trained models
pipeline = joblib.load('pipeline.pkl')
kmeans = joblib.load('kmeans_clustering.pkl')

# Load and preprocess data
def load_and_preprocess_data():
    """Loads and preprocesses the customer segmentation data."""
    df = pd.read_csv('customer_segmentation.csv')
    df.dropna(inplace=True)
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    df['Age'] = datetime.now().year - df['Year_Birth']
    df['total_amount'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
    return df

df = load_and_preprocess_data()

# --- Helper functions to generate data for endpoints ---

def get_overview_data(data_frame):
    """Provides overview data for the dashboard."""
    if data_frame.empty:
        return {
            'total_customers': 0,
            'avg_income': "₹0",
            'avg_age': 0,
            'avg_spend': "₹0",
            'income_distribution': {'labels': [], 'data': []},
            'age_distribution': {'labels': [], 'data': []}
        }

    total_customers = len(data_frame)
    avg_income = data_frame['Income'].mean()
    avg_age = data_frame['Age'].mean()
    avg_spend = data_frame['total_amount'].mean()

    income_bins = [0, 30000, 50000, 70000, 90000, float('inf')]
    income_labels = ['0-30k', '30k-50k', '50k-70k', '70k-90k', '90k+']
    income_dist = pd.cut(data_frame['Income'], bins=income_bins, labels=income_labels).value_counts().sort_index().to_dict()

    age_bins = [18, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    age_labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100-109', '110+']
    age_dist = pd.cut(data_frame['Age'], bins=age_bins, labels=age_labels).value_counts().sort_index().to_dict()

    return {
        'total_customers': total_customers,
        'avg_income': f"₹{avg_income:,.0f}",
        'avg_age': int(avg_age),
        'avg_spend': f"₹{avg_spend:,.0f}",
        'income_distribution': {'labels': list(income_dist.keys()), 'data': list(income_dist.values())},
        'age_distribution': {'labels': list(age_dist.keys()), 'data': list(age_dist.values())}
    }

def get_demographics_data(data_frame):
    """Provides demographic data for the dashboard."""
    if data_frame.empty:
        return {
            'education_distribution': {'labels': [], 'data': []},
            'marital_distribution': {'labels': [], 'data': []},
            'kidhome_distribution': {'labels': [], 'data': []},
            'teenhome_distribution': {'labels': [], 'data': []},
            'enrollment_distribution': {'labels': [], 'data': []}
        }
        
    education_dist = data_frame['Education'].value_counts().to_dict()
    marital_dist = data_frame['Marital_Status'].value_counts().to_dict()
    kidhome_dist = data_frame['Kidhome'].value_counts().to_dict()
    teenhome_dist = data_frame['Teenhome'].value_counts().to_dict()
    enrollment_dist = data_frame['Dt_Customer'].dt.year.value_counts().sort_index().to_dict()

    return {
        'education_distribution': {'labels': list(education_dist.keys()), 'data': list(education_dist.values())},
        'marital_distribution': {'labels': list(marital_dist.keys()), 'data': list(marital_dist.values())},
        'kidhome_distribution': {'labels': list(kidhome_dist.keys()), 'data': list(kidhome_dist.values())},
        'teenhome_distribution': {'labels': list(teenhome_dist.keys()), 'data': list(teenhome_dist.values())},
        'enrollment_distribution': {'labels': list(enrollment_dist.keys()), 'data': list(enrollment_dist.values())}
    }

def get_purchasing_data(data_frame):
    """Provides purchasing behavior data."""
    if data_frame.empty:
        return {
            'purchase_channels': {'labels': ['Web', 'Catalog', 'Store', 'Deals'], 'data': [0,0,0,0]},
            'product_spending': {'labels': ['Wines', 'Meat', 'Fruits', 'Fish', 'Sweet', 'Gold'], 'data': [0,0,0,0,0,0]},
            'recency_spend': []
        }
        
    purchase_channels = {
        'Web': data_frame['NumWebPurchases'].mean(),
        'Catalog': data_frame['NumCatalogPurchases'].mean(),
        'Store': data_frame['NumStorePurchases'].mean(),
        'Deals': data_frame['NumDealsPurchases'].mean()
    }
    product_spending = {
        'Wines': data_frame['MntWines'].mean(),
        'Meat': data_frame['MntMeatProducts'].mean(),
        'Fruits': data_frame['MntFruits'].mean(),
        'Fish': data_frame['MntFishProducts'].mean(),
        'Sweet': data_frame['MntSweetProducts'].mean(),
        'Gold': data_frame['MntGoldProds'].mean()
    }
    recency_spend = data_frame[['Recency', 'total_amount']].sample(min(100, len(data_frame))).to_dict('records')

    return {
        'purchase_channels': {'labels': list(purchase_channels.keys()), 'data': list(purchase_channels.values())},
        'product_spending': {'labels': list(product_spending.keys()), 'data': list(product_spending.values())},
        'recency_spend': recency_spend
    }

def get_campaigns_data(data_frame):
    """Provides marketing campaign data."""
    if data_frame.empty:
        return {
            'campaign_acceptance': {'accepted': [0,0,0,0,0], 'rejected': [0,0,0,0,0]},
            'web_activity': {'visits': [], 'purchases': []}
        }
        
    campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    campaign_acceptance = {
        'accepted': data_frame[campaign_cols].sum().to_list(),
        'rejected': (len(data_frame) - data_frame[campaign_cols].sum()).to_list()
    }
    web_activity = {
        'visits': data_frame.groupby(data_frame['Dt_Customer'].dt.month)['NumWebVisitsMonth'].mean().to_list(),
        'purchases': data_frame.groupby(data_frame['Dt_Customer'].dt.month)['NumWebPurchases'].mean().to_list()
    }
    return {
        'campaign_acceptance': campaign_acceptance,
        'web_activity': web_activity
    }

def get_segments_data(data_frame):
    """Provides customer segmentation data."""
    if data_frame.empty:
        return {
            'cluster_data': [],
            'segment_descriptions': [
                {"title": "High Income, High Spend", "description": "Affluent customers who make frequent purchases across all categories. Highly responsive to premium product campaigns."},
                {"title": "Mid Income, Mid Spend", "description": "Value-conscious customers who respond well to deals and promotions. Make regular but modest purchases."},
                {"title": "Low Income, Low Spend", "description": "Budget-focused customers who primarily purchase discounted items and respond to price-based promotions."},
                {"title": "Loyalists", "description": "Customers with high recency and frequent purchases, often across multiple channels."}
            ]
        }
        
    features = data_frame[['Age', 'Income', 'total_amount', 'NumWebPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Recency']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    data_frame['Cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_data = []
    for cluster in sorted(data_frame['Cluster'].unique()):
        cluster_df = data_frame[data_frame['Cluster'] == cluster]
        cluster_data.append({
            'name': f'Segment {cluster}',
            'data': cluster_df[['Income', 'total_amount']].to_dict('records')
        })

    segment_descriptions = [
        {"title": "High Income, High Spend", "description": "Affluent customers who make frequent purchases across all categories. Highly responsive to premium product campaigns."},
        {"title": "Mid Income, Mid Spend", "description": "Value-conscious customers who respond well to deals and promotions. Make regular but modest purchases."},
        {"title": "Low Income, Low Spend", "description": "Budget-focused customers who primarily purchase discounted items and respond to price-based promotions."},
        {"title": "Loyalists", "description": "Customers with high recency and frequent purchases, often across multiple channels."}
    ]

    return {
        'cluster_data': cluster_data,
        'segment_descriptions': segment_descriptions
    }

# --- Flask Routes ---

@app.route('/')
def dashboard():
    """Renders the main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/data/overview')
def overview_data():
    """Provides overview data for the dashboard."""
    return jsonify(get_overview_data(df))

@app.route('/api/data/demographics')
def demographics_data():
    """Provides demographic data for the dashboard."""
    return jsonify(get_demographics_data(df))

@app.route('/api/data/purchasing')
def purchasing_data():
    """Provides purchasing behavior data."""
    return jsonify(get_purchasing_data(df))

@app.route('/api/data/campaigns')
def campaigns_data():
    """Provides marketing campaign data."""
    return jsonify(get_campaigns_data(df))

@app.route('/api/data/segments')
def segments_data():
    """Provides customer segmentation data."""
    return jsonify(get_segments_data(df.copy())) # Use copy to avoid modifying global df with cluster column

@app.route('/api/data/filtered', methods=['POST'])
def filtered_data():
    """Endpoint for filtering data."""
    filters = request.json
    filtered_df = df.copy()

    if filters.get('education') and filters['education'] != 'all':
        filtered_df = filtered_df[filtered_df['Education'] == filters['education']]
    
    if filters.get('marital_status') and filters['marital_status'] != 'all':
        filtered_df = filtered_df[filtered_df['Marital_Status'] == filters['marital_status']]
        
    if filters.get('income_range') and filters['income_range'] != 'all':
        if filters['income_range'] == 'low':
            filtered_df = filtered_df[filtered_df['Income'] <= 30000]
        elif filters['income_range'] == 'medium':
            filtered_df = filtered_df[(filtered_df['Income'] > 30000) & (filtered_df['Income'] <= 60000)]
        elif filters['income_range'] == 'high':
            filtered_df = filtered_df[filtered_df['Income'] > 60000]

    return jsonify({
        "overview": get_overview_data(filtered_df),
        "demographics": get_demographics_data(filtered_df),
        "purchasing": get_purchasing_data(filtered_df),
        "campaigns": get_campaigns_data(filtered_df),
        "segments": get_segments_data(filtered_df)
    })

if __name__ == '__main__':
    app.run(debug=True)

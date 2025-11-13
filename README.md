# Customer Segmentation & Response Prediction Dashboard

## 1. Overview

This project analyzes customer data to perform two key marketing tasks:
1.  **Customer Segmentation:** It uses unsupervised learning (K-Means Clustering) to group customers into distinct segments based on their demographics and purchasing behavior.
2.  **Campaign Response Prediction:** It uses supervised learning (XGBoost) to predict the likelihood of a customer responding to a marketing campaign.

The results are presented in an interactive web dashboard built with Streamlit, where you can input a new customer's profile and receive an instant analysis, segment assignment, and actionable marketing recommendations.

## 2. Features

The interactive dashboard (`dashboard/app.py`) provides the following features:

* **Interactive Customer Input:** A sidebar to input a new customer's profile (Age, Income, Spending Habits, Purchase Behavior).
* **Real-time Segmentation:** Automatically classifies the input customer into one of four distinct segments (e.g., "Budget-Conscious," "Premium Lifestyle") using a pre-trained K-Means model.
* **Response Prediction:** Predicts whether the customer is "LIKELY" or "UNLIKELY" to respond to a campaign, including a probability score.
* **Actionable Recommendations:** Provides tailored marketing strategies based on the customer's predicted segment.
* **Data Visualization:**
    * **Spending Pie Chart:** Shows the breakdown of the customer's spending.
    * **Probability Gauge:** A visual gauge of the campaign response likelihood.
    * **Profile Radar Chart:** Compares the customer's profile (Income, Spending, etc.) against normalized values.
* **Comparative Analysis:** Benchmarks the input customer against the existing database by showing their percentile rank for Income, Spending, and Age.

## 3. Methodology

The analysis and models were developed in the `notebook/EDA.ipynb` notebook.

1.  **Data Loading & Cleaning:**
    * Loaded the `customer_segmentation.csv` dataset.
    * Handled missing values by dropping rows where 'Income' was null.
    * Converted `Dt_Customer` to datetime objects.

2.  **Feature Engineering:**
    * Created an `Age` column by subtracting `Year_Birth` from 2025.
    * Created a `totat_amount` column by summing all spending-related columns (`MntWines`, `MntFruits`, etc.).
    * Created an `Age_grp` for categorical analysis.

3.  **Unsupervised Learning (Segmentation):**
    * A feature set (`Age`, `Income`, `totat_amount`, `NumWebPurchases`, etc.) was selected.
    * Principal Component Analysis (PCA) was applied to reduce dimensionality.
    * The Elbow Method was used on the PCA-transformed data to find the optimal number of clusters (k=4).
    * A K-Means model with 4 clusters was trained and saved as `model/kmeans_clustering.pkl`.

4.  **Supervised Learning (Prediction):**
    * The same PCA-transformed data was used as features (X) to predict the `Response` column (y).
    * An XGBoost Classifier was trained, as it provided good performance (88% accuracy in the notebook).
    * The trained XGBoost model was saved as `model/xgboost_pipeline.pkl`.

## 4. Tech Stack

* **Python 3.10+**
* **Data Analysis & ML:** Pandas, NumPy, Scikit-learn, XGBoost, Joblib
* **Dashboard:** Streamlit
* **Visualization:** Plotly, Matplotlib, Seaborn

## 5. Project Structure

```
customer-segmentation/
├── dashboard/
│   └── app.py
├── data/
│   └── customer_segmentation.csv
├── model/
│   ├── kmeans_clustering.pkl
│   └── xgboost_pipeline.pkl
├── notebook/
│   └── EDA.ipynb
└── requirements.txt
```

## 6. Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://your-repository-url.git](https://your-repository-url.git)
    cd customer-segmentation
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    The `requirements.txt` file should contain:
    ```
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    plotly
    streamlit
    xgboost
    joblib
    ```
    Install them using:
    ```bash
    pip install -r requirements.txt
    ```

## 7. How to Run the Dashboard

1.  Navigate to the `dashboard` directory:
    ```bash
    cd dashboard
    ```

2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

3.  Open your web browser and go to the local URL provided (e.g., `http://localhost:8501`).

4.  Use the sidebar to enter a customer's details and click "Analyze Customer Profile" to see the results.
````

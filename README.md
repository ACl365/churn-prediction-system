# TeleChurn Predictor: Advanced Customer Churn Prediction System

![TeleChurn Predictor Banner](https://github.com/ACl365/churn-prediction-system/raw/main/assets/banner.png)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8.0-yellow)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.10.0-red)](https://streamlit.io/)

## 📊 Project Overview

TeleChurn Predictor will be a sophisticated machine learning system designed to predict customer churn in the telecommunications industry with high accuracy. This end-to-end solution will combine advanced feature engineering, ensemble modelling techniques, and an interactive dashboard to help telecom companies proactively identify at-risk customers and implement targeted retention strategies.

## 🔍 Business Problem

Customer churn (the loss of clients or customers) presents a significant challenge for telecom companies:

- The average monthly churn rate in the telecom industry ranges from 2-3%
- Acquiring a new customer costs 5-25 times more than retaining an existing one
- A 5% increase in customer retention can increase profits by 25-95%

TeleChurn Predictor will address this challenge by:
1. Identifying customers at high risk of churning before they leave
2. Providing actionable insights into churn factors
3. Enabling targeted retention campaigns to maximise ROI
4. Monitoring churn metrics in real-time through an interactive dashboard

## 💻 Technical Approach

This project will implement a comprehensive machine learning pipeline:

1. **Data Preprocessing**:
   - Handling missing values with advanced imputation techniques
   - Feature scaling and normalisation
   - Encoding categorical variables
   - Time-based feature engineering

2. **Feature Engineering**:
   - Customer behaviour pattern extraction
   - Temporal feature creation (e.g., service usage trends)
   - Interaction feature development
   - Automated feature selection using SHAP values

3. **Model Development**:
   - Ensemble approach combining:
     - Gradient Boosting (XGBoost)
     - Random Forest
     - Deep Neural Networks
   - Hyperparameter optimisation using Bayesian techniques
   - Cross-validation with stratified k-fold

4. **Explainability**:
   - SHAP (SHapley Additive exPlanations) for model interpretation
   - Feature importance visualisation
   - Individual prediction explanations

5. **Deployment**:
   - RESTful API for real-time predictions
   - Streamlit dashboard for business users
   - Automated retraining pipeline

## ✨ Key Features

- **High-Performance Prediction**: Advanced algorithms to identify customers likely to churn
- **Real-Time Scoring**: API endpoint for integrating predictions into existing systems
- **Actionable Insights**: Identification of key factors driving churn
- **Customer Segmentation**: Clustering of at-risk customers for targeted interventions
- **Interactive Dashboard**: Visualisation of churn metrics and prediction explanations
- **Automated Monitoring**: Drift detection and model performance tracking
- **What-If Analysis**: Simulation tool to test retention strategies
- **Scheduled Retraining**: Automated model updates to maintain accuracy

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/ACl365/churn-prediction-system.git
cd churn-prediction-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialise the database
python scripts/init_db.py
```

### Prerequisites

- Python 3.8+
- PostgreSQL 13+ (for storing customer data and predictions)
- 8GB+ RAM (recommended for training models)

## 📝 Usage

### Training the Model

```python
from telechurn.pipeline import ChurnPipeline

# Initialise the pipeline
pipeline = ChurnPipeline(config_path="config/model_config.yaml")

# Train the model
pipeline.train(data_path="data/customer_data.csv")

# Evaluate performance
metrics = pipeline.evaluate(test_data_path="data/test_data.csv")
print(f"Model Accuracy: {metrics['accuracy']:.4f}")
print(f"Model F1-Score: {metrics['f1_score']:.4f}")

# Save the trained model
pipeline.save_model("models/churn_model_v1.pkl")
```

### Making Predictions

```python
from telechurn.predictor import ChurnPredictor

# Load the trained model
predictor = ChurnPredictor(model_path="models/churn_model_v1.pkl")

# Predict for a single customer
customer_data = {
    "tenure": 24,
    "monthly_charges": 65.5,
    "total_charges": 1570.0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    # Additional features...
}

prediction = predictor.predict(customer_data)
print(f"Churn Probability: {prediction['churn_probability']:.2f}")
print(f"Risk Level: {prediction['risk_level']}")
print(f"Top Factors: {prediction['top_factors']}")
```

### Running the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run dashboard/app.py

# For production deployment with authentication
streamlit run dashboard/app.py -- --prod --auth
```

## 📊 Dashboard

The TeleChurn Predictor will include an interactive Streamlit dashboard that provides:

- **Executive Summary**: High-level churn metrics and KPIs
- **Customer Risk Profiles**: Detailed view of at-risk customers
- **Prediction Explanations**: Visual breakdown of factors influencing churn
- **What-If Simulator**: Tool to test how changes in service offerings affect churn
- **Retention Campaign Tracker**: Monitor the effectiveness of retention efforts
- **Model Performance Monitor**: Track accuracy and other metrics over time

![Dashboard Screenshot](https://github.com/ACl365/churn-prediction-system/raw/main/assets/dashboard.png)

## 📁 Project Structure

```
churn-prediction-system/
├── config/                  # Configuration files
│   ├── model_config.yaml    # Model hyperparameters
│   └── feature_config.yaml  # Feature engineering settings
├── data/                    # Data directory
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed datasets
│   └── external/            # External data sources
├── models/                  # Trained model files
├── notebooks/               # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   ├── 02_feature_eng.ipynb # Feature engineering experiments
│   └── 03_modeling.ipynb    # Model development and evaluation
├── telechurn/               # Main package
│   ├── __init__.py
│   ├── data/                # Data processing modules
│   ├── features/            # Feature engineering
│   ├── models/              # Model implementations
│   ├── pipeline.py          # End-to-end pipeline
│   └── predictor.py         # Prediction interface
├── dashboard/               # Streamlit dashboard
│   ├── app.py               # Main dashboard application
│   └── components/          # Dashboard components
├── api/                     # API service
│   ├── app.py               # FastAPI application
│   └── routers/             # API endpoints
├── scripts/                 # Utility scripts
├── tests/                   # Unit and integration tests
├── .env.example             # Example environment variables
├── requirements.txt         # Project dependencies
├── setup.py                 # Package installation
└── README.md                # Project documentation
```

## 🔮 Future Work

The TeleChurn Predictor roadmap includes:

1. **Advanced ML Techniques**:
   - Implement deep learning models for sequence prediction
   - Explore reinforcement learning for retention strategy optimisation
   - Develop multi-objective models balancing churn risk and customer value

2. **Enhanced Features**:
   - Integrate NLP for customer sentiment analysis from support interactions
   - Incorporate network effects (social connections between customers)
   - Add macroeconomic indicators as external features

3. **System Improvements**:
   - Develop a recommendation engine for personalised retention offers
   - Implement A/B testing framework for retention strategies
   - Create automated customer journey mapping

4. **Expanded Deployment**:
   - Mobile application for field sales teams
   - Integration with CRM systems (Salesforce, etc.)
   - Edge deployment for real-time scoring in retail locations

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Alexander Clarke - [alexanderclarke365@gmail.com](mailto:alexanderclarke365@gmail.com)

Project Link: [https://github.com/ACl365/churn-prediction-system](https://github.com/ACl365/churn-prediction-system)

---

*This project is being developed as part of advanced machine learning research in customer retention optimisation.*
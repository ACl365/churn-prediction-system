# TeleChurn Predictor: Telecom Customer Churn Prediction System

![TeleChurn Predictor Banner](https://github.com/ACl365/churn-prediction-system/raw/main/assets/banner.png)

## 📊 Project Overview

TeleChurn Predictor is a sophisticated machine learning system designed to predict customer churn in the telecommunications industry with high accuracy. This end-to-end solution combines advanced feature engineering, ensemble modeling techniques, and an interactive dashboard to help telecom companies proactively identify at-risk customers and implement targeted retention strategies.

## 🔍 Business Problem

Customer churn (the loss of clients or customers) presents a significant challenge for telecom companies:

- The average monthly churn rate in the telecom industry ranges from 2-3%
- Acquiring a new customer costs 5-25 times more than retaining an existing one
- A 5% increase in customer retention can increase profits by 25-95%

TeleChurn Predictor addresses this challenge by:
1. Identifying customers at high risk of churning before they leave
2. Providing actionable insights into churn factors
3. Enabling targeted retention campaigns to maximize ROI
4. Monitoring churn metrics in real-time through an interactive dashboard

## 📁 Project Structure

```
project/
├── data/                    # Data directory
│   ├── raw/                 # Raw data files
│   └── processed/           # Processed datasets
├── notebooks/               # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   ├── 02_feature_eng.ipynb # Feature engineering experiments (to be created)
│   └── 03_modeling.ipynb    # Model development and evaluation (to be created)
├── telechurn/               # Main package
│   ├── data/                # Data processing modules
│   │   └── preprocessing.py # Data preprocessing module
│   ├── features/            # Feature engineering (to be created)
│   ├── models/              # Model implementations (to be created)
│   └── utils/               # Utility functions (to be created)
├── scripts/                 # Utility scripts
│   └── setup_data.py        # Script to set up data
└── README.md                # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ACl365/churn-prediction-system.git
   cd churn-prediction-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the data:
   ```bash
   python scripts/setup_data.py
   ```

## 📊 Data Exploration

The project includes a comprehensive exploratory data analysis notebook that examines:

- Data structure and quality
- Feature distributions and relationships
- Missing value analysis
- Correlation analysis
- Target variable distribution and relationships
- Outlier detection

To run the EDA notebook:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

## 🔧 Data Preprocessing

The `telechurn/data/preprocessing.py` module handles:

- Loading raw telecom data
- Cleaning and handling missing values
- Encoding categorical variables
- Scaling numerical features
- Creating feature groups
- Preparing data for modeling

Example usage:
```python
from telechurn.data.preprocessing import TelecomDataPreprocessor

# Initialize preprocessor
preprocessor = TelecomDataPreprocessor()

# Process and save data
preprocessor.process_and_save(
    train_path="data/raw/cell2celltrain.csv",
    output_train_path="data/processed/train_processed.csv",
    holdout_path="data/raw/cell2cellholdout.csv",
    output_holdout_path="data/processed/holdout_processed.csv"
)
```

## 📝 Next Steps

The project is currently in development with the following components planned:

1. **Feature Engineering Module**:
   - Create advanced features from raw telecom data
   - Implement feature selection techniques
   - Develop domain-specific transformations

2. **Model Development**:
   - Implement ensemble models (Random Forest, XGBoost, etc.)
   - Perform hyperparameter optimization
   - Evaluate model performance

3. **Deployment**:
   - Create a prediction API
   - Develop a dashboard for visualizing results
   - Implement model monitoring

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Alexander Clarke - [alexanderclarke365@gmail.com](mailto:alexanderclarke365@gmail.com)

Project Link: [https://github.com/ACl365/churn-prediction-system](https://github.com/ACl365/churn-prediction-system)
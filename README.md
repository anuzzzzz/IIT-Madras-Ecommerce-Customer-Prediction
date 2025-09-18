IIT-Madras E-commerce Customer Behavior Prediction
🏆 Competition Results
Kaggle Competition: E-commerce Shoppers Behaviour Understanding
Final Ranking: Top 10% (achieving F1-score of 0.623+ in final submission)
Team Performance: Ranked among 769+ participants
📋 Project Overview
This project focuses on predicting customer purchase behavior in e-commerce using machine learning techniques. The challenge involves analyzing user session data to determine whether a customer will make a purchase based on their browsing patterns, demographics, and technical attributes.
🎯 Problem Statement
An e-commerce company wants to understand customer behavior and predict purchase likelihood. The dataset contains user session data for a year, with each row representing a different user. The target variable "Made_Purchase" indicates whether the user made a purchase during that period.
Key Challenges:

Imbalanced dataset (more non-purchases than purchases)
Mixed data types (numerical and categorical)
Data quality issues (collected by non-experts)
Complex feature relationships

📊 Dataset Description
Features include:

Page Metrics: HomePage visits/duration, LandingPage visits/duration, ProductDescriptionPage visits/duration
Google Analytics: Bounce rates, Exit rates, Page values
Temporal: SeasonalPurchase indicators, WeekendPurchase, Month_SeasonalPurchase
Technical: OS, SearchEngine, Zone, Traffic Type
Demographics: CustomerType, Gender, Education, Marital Status
Settings: Cookies preferences

Target: Made_Purchase (Binary: True/False)
🔧 Technical Approach
Data Preprocessing

Missing Value Handling: KNNImputer (k=50) for numerical features, mode imputation for categorical
Data Validation: Logical consistency checks (e.g., page duration vs page visits)
Feature Engineering:

Ratio features (duration/visits)
Interaction features
Sanity filtering of impossible data combinations



Class Imbalance Handling

SMOTENC: Synthetic minority oversampling for mixed data types
Applied strategically within training pipeline to prevent data leakage

Model Architecture
Stacking Ensemble Approach:

Base Models:

AdaBoost Classifier
Extra Trees Classifier


Meta-learner: XGBoost (max_depth=8, learning_rate=0.01)
Feature Passing: Used passthrough=True for additional feature information

Feature Processing Pipeline
python# Numerical Pipeline
num_pipe = Pipeline([('imputer', KNNImputer(n_neighbors=50))])

# Categorical Pipeline  
cat_pipe = Pipeline([('cat_imputer', SimpleImputer(strategy="most_frequent"))])

# One-Hot Encoding for final features
OneHotEncoder() for categorical variables
🚀 Key Results

F1-Score: 0.623+ on test set
Cross-validation: Robust performance across different data splits
Feature Importance: Page values and user engagement metrics were most predictive
Model Interpretability: Ensemble approach provided stable predictions

📁 Project Structure
IIT-Madras-Ecommerce-Prediction/
│
├── notebook/
│   └── 21f1005327-notebook.ipynb    # Main analysis notebook
│
├── data/                             # Data files (not included due to size)
│   ├── train_data_v2.csv
│   ├── test_data_v2.csv
│   └── sample.csv
│
├── src/                              # Source code modules
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── model_training.py
│
├── submissions/                      # Competition submissions
│   └── final_submission.csv
│
├── requirements.txt                  # Dependencies
└── README.md                        # This file
🛠️ Installation & Usage
Prerequisites
bashpip install -r requirements.txt
Dependencies

pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn

Running the Code
bash# Open and run the Jupyter notebook
jupyter notebook notebook/21f1005327-notebook.ipynb
🔍 Model Performance Analysis
Confusion Matrix Results
[[3205  410]
 [1033 1193]]

True Negatives: 3205 (correctly predicted non-purchases)
False Positives: 410 (incorrectly predicted purchases)
False Negatives: 1033 (missed purchases)
True Positives: 1193 (correctly predicted purchases)

Key Insights

Page Values emerged as the strongest predictor
User Engagement (time spent, pages visited) highly correlated with purchase intent
Seasonal Patterns showed moderate predictive power
Technical attributes (OS, browser) had minimal impact

🏅 Competition Strategy

Robust Preprocessing: Handled missing values and data quality issues systematically
Feature Engineering: Created meaningful interaction features and ratios
Ensemble Methods: Combined multiple algorithms for better generalization
Cross-Validation: Ensured model stability across different data splits
Hyperparameter Tuning: Optimized key parameters for final performance

🔮 Future Improvements

Time-series analysis for seasonal patterns
Deep learning approaches for complex feature interactions
More sophisticated feature selection techniques
Advanced ensemble methods (e.g., multi-level stacking)

📞 Contact
Student ID: 21f1005327
Institution: Indian Institute of Technology Madras
Course: Machine Learning / Data Science Project
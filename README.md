# churned-prediction-r
This project is implemented to analyse which variables in the ecommerce dataset have strong effect on the customer's churn situation. Through the analysing procedures, there are some interesting dectections of attributes and statistical results after processing data and utilising three models Logistic Regression, Random Forest, and XGBoost in R Programming or R language.

## The structure of churned-prediction-using-r project
```
├── data/
│   ├── raw/                    
│   │   └── ecommerce_customer_churn.csv
│   └── processed/               
│       └── handle abnormal cases (age outliers, negative total purchases)
│       └── split train - test - cross validation data
│       └── tackle missing values (median method, kNN method, default values)
│       └── remove uneccesary values
│       └── transfer categorical data into factor values
├── models/                   
│   ├── 01. feature engineering: encoding, scaling data
│   ├── 02. multicollinearity check       # Before train Logistic Regression model
│   └── 03. use models to train & predict
│       └── Logistic Regression (ROC 78%)
│       └── Random Forest       (ROC 92.15%)
│       └── XGBoost             (ROC 92.48%)
├── plots/                      
│   ├── EDA. age_outlier, abnormal_total_purchase, churn_vs_country, churn_vs_membership_years, etc.
│   ├── correlation
│   └── confusion matrix
│   └── ROC plot
├── save_models/                      
└── README.md             
```

# churned-prediction-r
This project is implemented to analyse which variables in the ecommerce dataset have strong effect on the customer's churn situation. Through the analysing procedures, there are some interesting dectections of attributes and statistical results after processing data and utilising three models Logistic Regression, Random Forest, and XGBoost in R Programming or R language. The project aims to help businesses shift from a passive to a proactive approach to customer retention, in another word is Proactive Retention.

## The structure of churned-prediction-using-r project
```
├── data/
│   ├── raw/                    
│   │   └── ecommerce_customer_churn.csv
│   └── processed/               
│       └── handle abnormal cases (age outliers, negative total purchases)
│       └── split train - test - cross validation data
│       └── tackle missing values (median method, kNN method, default values)
│       └── remove an uneccesary variable
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
## Dataset
- This is [Customer Engagement and Churn Analytics Dataset](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset) from Kaggle, containing behavioral, demographic, and transactional data with 50.000 records and 25 columns. The data types of this dataset include numerical, categorical and object.
- There are some **missing values** in certain columns.
- The target variable is **_"Churned"_** with values are 1 - 0 (1 = Churned & 0 = Active).

## Project Objective
> Predict the ability of customer churn based on behavioral, demographic, transactional features to detect which factor has the most effect on this ability through using machine learning models such as Logistic Regression, Random Forest, and XGBoost on R language.

## Features
1. Core Libraries
   - tidyverse: data manipulation
   - caret: machine learning (ML) framework
   - pROC, VIM, car, ggcorrplot, RANN, scales
   - xgboost
2. Data Cleaning
   - Handled some age outliers which have the abnormal values Age > 122 (because the highest age is recognised is 122), transfer them into N/A values and then use median method to impute them as missing values.
   - Tackled the negative values of "Total_Purchase" by "**abs()**" to transfer the number from negative to positive values.
   - Used kNN or k-Nearest Neighbors, which finds the very similar to values and choose the average of those values to fill in the missing value position.
3. Exploratory Data Analysis - EDA
   - Utilised "**ggplot**" library to illustrate the relationship between some variables versus "Churned".
4. Machine Learning Models
   - Logistic Regression: the base model to predict and find which is the most important model effect on churn situation, and received the ROC is **78%**.
   - Random Forest: applied the bagging technique to reduce overfitting of data, and received the ROC is **92.15%**.
   - XGBoost: the superb model using boosting technique to optimise the accuracy and the imbalanced data, and received the ROC is** 92.48%**.
5. Business result
> The "Customer_Service_Calls", "Cart_Abandonment_Rate", "Lifetime_Value", "Discount_Usage_Rate" are the most important variables that business should monitor and prioritise for improvement. For along with, the business has to enhance the customer service through taking care them on the email platform to actively keep the customers.

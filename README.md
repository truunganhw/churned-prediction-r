# churned-prediction-r
This project is implemented to analyse which variables in the dataset, which the dataset has 50,000 records and 25 columns, have strong effect on the customer's churn situation. Through the analysing procedures, there are some interesting dectections of attributes and statistical results after processing data and utilising three models Logistic Regression, Random Forest, and XGBoost in R Programming or R language.

## The structure of churned-prediction-using-r project
├── library/
│   ├── tidyverse, caret, VIM, car, RANN, scales, ggcorrplot, pROC, xgboost                             
├── data/                     
│   ├── raw/ 
│       └──ecommerce_customer_churn_dataset.csv
│   ├── process/  
│       └──split train, test data
│       └──set trainControl, cross validation
│       └──handle missing value, abnormal dataa
├── models/                      
│   ├── final_churn_model.rds   
│   └── preprocessing_rules.rds  
├── plots/                   
│   ├── correlation_heatmap.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── report/                     
│   ├── CETM24_Assignment2_Report.pdf
│   └── references.bib           
├── .gitignore                   
├── customer_churn_project.Rproj 
└── README.md                    

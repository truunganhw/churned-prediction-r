# churned-prediction-r
This project is implemented to analyse which variables in the dataset, which the dataset has 50,000 records and 25 columns, have strong effect on the customer's churn situation. Through the analysing procedures, there are some interesting dectections of attributes and statistical results after processing data and utilising three models Logistic Regression, Random Forest, and XGBoost in R Programming or R language.

## The structure of churned-prediction-using-r project
> Raw data
> > Data processing 1: Abnormal values
> > > Exploratory Data Analysis (EDA)
> > > > Split train - test data - trControl - cross validation
> > > > > Data processing 2: Missing values (median, kNN method)
> > > > > > Remove unecessary column
> > > > > > > Transfer categorical data into factor
> > > > > > > > Feature Engineering: Encoding & Scaling
> > > > > > > > > Check multicollinearity
> > > > > > > > > > Model 1: Logistic Regression
> > > > > > > > > > > Model 2: Random Forest
> > > > > > > > > > > > Model 3: XGBoost
> > > > > > > > > > > > > Compare the ROC of the models
> > > > > > > > > > > > > > Save the models
                 

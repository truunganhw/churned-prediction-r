# Call the libraries
library(tidyverse)
library(caret)
library(VIM) # Visual & Impute missing values
library(car) # VIF, multicollinearity
library(RANN) # Nearest neighbor search
library(scales)
library(ggcorrplot)
library(pROC) 
library(xgboost)
library(corrplot)

# Data overview
customer_data <- read.csv("C:/Users/trung/Downloads/archive/ecommerce_customer_churn_dataset.csv")
View(customer_data)
glimpse(customer_data)
summary(customer_data)

# Counting total missing values
total_missing <- sum(is.na(customer_data))
cat("Total missing values is", total_missing)

# EDA - Session 1: For abnormal cases
abnormal_age <- sum(customer_data$Age > 122, na.rm = TRUE)
cat("Total number of customer's age > 122 is:", abnormal_age)

abnormal_age_info <- customer_data %>%
  filter(Age > 122) %>%
  select(Age, Gender, Churned)
abnormal_age_info

ggplot(customer_data, aes(y=Age)) +
  geom_boxplot(fill = "darkgreen", alpha = 0.7,
               outlier.color = "red",
               outlier.size = 3) +
  
  scale_y_continuous(limits = c(0, 200), breaks = seq(0, 200, by = 20)) +
  
  labs(title = "Detect the Abnormal Age",
       subtitle = "Outlier - Age > 122",
       y = "Age") +
  theme_minimal()

# Age outlier handling
# Assign from >122 to NA and handle with median
customer_data <- customer_data %>%
  mutate(Age = ifelse(Age > 122, NA, Age))

summary(customer_data)

negative_total_purchases <- sum(customer_data$Total_Purchases < 0, na.rm = TRUE)
cat("Total number of negative 'total_purchase' is:", negative_total_purchases)

abnormal_purchases_info <- customer_data %>%
  filter(Total_Purchases < 0) %>%
  select(Total_Purchases, Returns_Rate, Average_Order_Value, Membership_Years, Churned)

abnormal_purchases_info

customer_data <- customer_data %>%
  mutate(Total_Purchases = abs(Total_Purchases))

summary(customer_data)

# EDA - Session 2: For exploring data
plot_country <- customer_data %>%
  group_by(Country) %>%
  summarise(Total = n(),
            Churn_count = sum(Churned == 1),
            Churn_rate = Churn_count / Total
            ) %>%
  arrange(desc(Churn_rate))

ggplot(plot_country, aes(x = reorder(Country, -Churn_rate), y = Churn_rate, fill = Country)) +
  geom_col(show.legend = F) +
  geom_text(aes(label = percent(Churn_rate, accuracy = 0.1)), vjust = -0.5) +
  scale_y_continuous(labels = percent) +
  labs(
    title = "Percentage of churn is based on Country",
    x = "Country",
    y = "Churn_rate (%)"
  ) +
  theme_minimal()

ggplot(customer_data, aes(x = as.factor(Churned), y = Membership_Years, fill = as.factor(Churned))) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("darkgreen", "lightpink"), labels = c("Active", "Churned")) +
  labs(
    title = "The relationship between Membership_Years vs Churned status",
    x = "Churn status (0: Active, 1: Churned)",
    y = "Membership years",
    fill = "Active or Churned"
  ) +
  theme_light()

ggplot(customer_data, aes(x = as.factor(Churned), y = Days_Since_Last_Purchase, fill = as.factor(Churned))) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("lightblue", "orange"), labels = c("Active", "Churned")) +
  labs(
    title = "The effect of the days since the last purchase vs Churned situation",
    x = "Churn status (0: Active, 1:Churned)",
    y = "The days sice the last purchase",
    fill = "Active or Churned"
  ) +
  theme_light()

# Train - Test split
set.seed(123)
train_index <- createDataPartition(customer_data$Churned, p = 0.8, list = F)
train_data <- customer_data[train_index, ]
test_data <- customer_data[-train_index, ]

# Train data : Handling missing values
train_data <- train_data %>%
  mutate(
    Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age),
    
    Session_Duration_Avg = ifelse(is.na(Session_Duration_Avg),
                                  median(Session_Duration_Avg, na.rm = TRUE),
                                  Session_Duration_Avg),
    
    Pages_Per_Session = ifelse(is.na(Pages_Per_Session),
                               median(Pages_Per_Session, na.rm = TRUE),
                               Pages_Per_Session),
    
    # use median to avoid extreme bias
    Days_Since_Last_Purchase = ifelse(is.na(Days_Since_Last_Purchase),
                                      median(Days_Since_Last_Purchase, na.rm = TRUE),
                                      Days_Since_Last_Purchase),
    
    # rate is continuous behavior, shouldn't be 0
    Discount_Usage_Rate = ifelse(is.na(Discount_Usage_Rate),
                                 median(Discount_Usage_Rate, na.rm = TRUE),
                                 Discount_Usage_Rate),
    
    Returns_Rate = ifelse(is.na(Returns_Rate),
                          median(Returns_Rate, na.rm = TRUE),
                          Returns_Rate),
    
    Email_Open_Rate = ifelse(is.na(Email_Open_Rate),
                             median(Email_Open_Rate, na.rm = TRUE),
                             Email_Open_Rate),
    
    Mobile_App_Usage = ifelse(is.na(Mobile_App_Usage),
                              median(Mobile_App_Usage, na.rm = TRUE),
                              Mobile_App_Usage),
    
    # no wishlist, no review, no call = 0
    Wishlist_Items = ifelse(is.na(Wishlist_Items), 0, Wishlist_Items),
    
    Product_Reviews_Written = ifelse(is.na(Product_Reviews_Written), 0, Product_Reviews_Written),
    
    Customer_Service_Calls = ifelse(is.na(Customer_Service_Calls), 0, Customer_Service_Calls),
    
    # do not link social media account, score = 0
    Social_Media_Engagement_Score = ifelse(is.na(Social_Media_Engagement_Score), 0, Social_Media_Engagement_Score),
    
    # default missing method = 1
    Payment_Method_Diversity = ifelse(is.na(Payment_Method_Diversity), 1, Payment_Method_Diversity)
  )

# Test data: Handling missing value
test_data <- test_data %>%
  mutate(
    Age = ifelse(is.na(Age), median(train_data$Age), Age),
    
    Session_Duration_Avg = ifelse(is.na(Session_Duration_Avg),
                                  median(train_data$Session_Duration_Avg),
                                  Session_Duration_Avg),
    
    Pages_Per_Session = ifelse(is.na(Pages_Per_Session),
                               median(train_data$Pages_Per_Session),
                               Pages_Per_Session),
    
    # use median to avoid extreme bias
    Days_Since_Last_Purchase = ifelse(is.na(Days_Since_Last_Purchase),
                                      median(train_data$Days_Since_Last_Purchase),
                                      Days_Since_Last_Purchase),
    
    # rate is continuous behavior, shouldn't be 0
    Discount_Usage_Rate = ifelse(is.na(Discount_Usage_Rate),
                                 median(train_data$Discount_Usage_Rate),
                                 Discount_Usage_Rate),
    
    Returns_Rate = ifelse(is.na(Returns_Rate),
                          median(train_data$Returns_Rate),
                          Returns_Rate),
    
    Email_Open_Rate = ifelse(is.na(Email_Open_Rate),
                             median(train_data$Email_Open_Rate),
                             Email_Open_Rate),
    
    Mobile_App_Usage = ifelse(is.na(Mobile_App_Usage),
                              median(train_data$Mobile_App_Usage),
                              Mobile_App_Usage),
    
    # no wishlist, no review, no call = 0
    Wishlist_Items = ifelse(is.na(Wishlist_Items), 0, Wishlist_Items),
    
    Product_Reviews_Written = ifelse(is.na(Product_Reviews_Written), 0, Product_Reviews_Written),
    
    Customer_Service_Calls = ifelse(is.na(Customer_Service_Calls), 0, Customer_Service_Calls),
    
    # do not link social media account, score = 0
    Social_Media_Engagement_Score = ifelse(is.na(Social_Media_Engagement_Score), 0, Social_Media_Engagement_Score),
    
    # default missing method = 1
    Payment_Method_Diversity = ifelse(is.na(Payment_Method_Diversity), 1, Payment_Method_Diversity)
  )

# Using kNN to compute Credit_Balance
knn_variables <- c("Credit_Balance", "Lifetime_Value", "Total_Purchases", "Average_Order_Value")

datatrain_knn <- train_data[, knn_variables]
datatest_knn <- test_data[, knn_variables]

knn_model <- preProcess(datatrain_knn, method = "knnImpute", k = 5)

train_knn_model <- predict(knn_model, datatrain_knn)
test_knn_model <- predict(knn_model, datatest_knn)

train_data$Credit_Balance <- train_knn_model$Credit_Balance
test_data$Credit_Balance <- test_knn_model$Credit_Balance

View(train_data)
View(test_data)

# Remove the unecessary variable
train_data <- train_data %>% select(-City)
test_data <- test_data %>% select(-City)

# Transfer data as factor variables
factor_columns <- c("Gender", "Country", "Signup_Quarter", "Churned")
train_data[factor_columns] <- lapply(train_data[factor_columns], as.factor)
test_data[factor_columns] <- lapply(test_data[factor_columns], as.factor)

# Change the data of Churned column
levels(train_data$Churned) <- c("No", "Yes")
levels(test_data$Churned) <- c("No", "Yes")

# Encoding data
dummy_model <- dummyVars(~. -Churned, data = train_data, fullRank = T)

train_encoded <- predict(dummy_model, newdata = train_data)
test_encoded <- predict(dummy_model, newdata = test_data)

train_encoded_df <- as.data.frame(train_encoded)
test_encoded_df <- as.data.frame(test_encoded)

train_encoded_df$Churned <- train_data$Churned
test_encoded_df$Churned <- test_data$Churned

View(train_encoded_df)

# Scaling data
scale_model <- preProcess(train_encoded_df %>% select (-Churned), method = c("center", "scale"))

train_final <- predict(scale_model, train_encoded_df)
test_final <- predict(scale_model, test_encoded_df)

View(train_final)

# Check multicollinearity 
multicollinearity_model <- glm(Churned~., data = train_final, family = "binomial")
multicollinearity_values <- vif(multicollinearity_model)
multicollinearity_values

cor_matrix <- cor(train_final %>% select(-Churned))
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.cex = 0.6,
         title = "The correlation between variables")

# Machine Learning models
# Logistic Regression
fit_control_lg <- trainControl(
  method = "cv", number = 5, classProbs = T,
  summaryFunction = twoClassSummary, savePredictions = "final",
  verboseIter = T
)

set.seed(123)

# Train logistic regression model
logistic_model <- train(Churned~., data = train_final, 
                        method = "glm", family = "binomial",
                        trControl = fit_control_lg, metric = "ROC")

summary(logistic_model)
print(logistic_model)

# Test logistic regression model
pred_log_model <- predict(logistic_model, newdata = test_final)
pred_log_model_2 <- predict(logistic_model, newdata = test_final, type = "prob")

results_log_model <- data.frame( Actual = test_final$Churned,
                                 Predicted = pred_log_model,
                                 Prob_yes = round(pred_log_model_2$Yes, 4),
                                 Prob_no = round(pred_log_model_2$No, 4))

cm_logistic_model <- confusionMatrix(pred_log_model, test_final$Churned, mode = "everything", positive = "Yes")
cm_logistic_model

# Random Forest
fit_control_rf <- trainControl(
  method = "cv", number = 5, classProbs = T,
  summaryFunction = twoClassSummary,
  verboseIter = T, allowParallel = T
)

set.seed(123)

# Train random forest model
rf_model <- train(Churned ~., data = train_final, method = "ranger",
                  metric = "ROC", trControl = fit_control_rf, 
                  tuneLength = 5, importance = "impurity"
                  )

# Predict random forest model
pred_rf_model <- predict(rf_model, newdata = test_final)
pred_rf_model_2 <- predict(rf_model, newdata = test_final, type = "prob")

roc_rf <- roc(test_final$Churned, pred_rf_model_2$Yes, levels = c("No", "Yes"), direction = "<")
print(auc(roc_rf))

cm_rf_model <- confusionMatrix(pred_rf_model, test_final$Churned, mode = "everything", positive = "Yes")
cm_rf_model

cm_table_rf <- as.data.frame(cm_rf_model$table)

ggplot(cm_table_rf, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1, size = 6, color = "black") +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(
    title = "Confusion Matrix - Random Forest",
    x = "Actual",
    y = "Predicted"
  ) +
  theme_minimal()+
  theme(
    axis.text = element_text(size = 12, face = "bold"),
    plot.title = element_text(size = 16, face = "bold")
  )

plot(varImp(rf_model), top = 20, main = "Top 20 factors effect in using Random Forest")

# XGBoost
xgboost_grid <- expand.grid(
  nrounds = c(100, 200), max_depth = c(3,6), eta = c(0.01, 0.1, 0.3),
  gamma = 0, colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8
)

fit_control_xgb <- trainControl(
  method = "cv", number = 5, classProbs = T, summaryFunction = twoClassSummary,
  verboseIter = T, allowParallel = T
)

set.seed(123)

xgb_model <- train(Churned~., data = train_final, method = "xgbTree",
                   metric = "ROC", trControl = fit_control_xgb,
                   tuneGrid = xgboost_grid, verbosity = 0)

pred_xgb_model <- predict(xgb_model, newdata = test_final)

cm_xgboost <- confusionMatrix(pred_xgb_model, test_final$Churned, 
                              mode = "everything", positive = "Yes")
cm_xgboost

cm_table <- as.data.frame(cm_xgboost$table)

ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1, size = 6, color = "black") +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(
    title = "Confusion Matrix - XGBoost Model",
    x = "Actual",
    y = "Predicted"
  ) +
  theme_minimal()+
  theme(
    axis.text = element_text(size = 12, face = "bold"),
    plot.title = element_text(size = 16, face = "bold")
  )

plot(varImp(xgb_model), top = 20, main = "Top 20 factors effect in using XGBoost model")

# Compare models
compare_models <- resamples(list(
  Logistic = logistic_model, Random_Forest = rf_model, XGB = xgb_model))

summary(compare_models)

bwplot(compare_models, metric = "ROC", main = "The ROC index of each model")
bwplot(compare_models, metric = "Sens", main = "The Sens index of each model")

importance_model <- varImp(xgb_model)
pred_xgb_result <- predict(xgb_model, newdata = test_final, type = "prob")

ROC_plot_xgb <- roc(test_final$Churned, pred_xgb_result$Yes, levels = c("No", "Yes"), direction = "<")
print(auc(ROC_plot_xgb))

plot(ROC_plot_xgb, main = "ROC curve of XGBoost Model")

dir.create("models", showWarnings = FALSE)
saveRDS(xgb_model, "models/best_model_churn.rds")
saveRDS(knn_model, "models/knn_model.rds")
saveRDS(dummy_model, "models/dummy_rules.rds")
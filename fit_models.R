# -----------------------------------------------------------------------------
# Predictive Analytics Project - Fit Models
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Possible Methods
# -----------------------------------------------------------------------------
# Logistic regression (w/ stepwise selection)   Implemented
# LDA/QDA                                       Should implement
# KNN classifier                                Maybe?
# Naive Bayes                                   Maybe?
# Ridge and Lasso regression                    Implemented
# Linear, cubic, and smoothing splines          Maybe?
# Decision tree                                 Implemented
# Random forest                                 Implemented
# Bagging and boosting                          Implemented
# BART                                          Implemented
# SVMs                                          Should implement

# Import libraries
library(BART)           # for BART
library(caret)          # for train() function used to fit most of the models
library(tidyverse)      # for data cleaning

# Set the seed for reproducibility
set.seed(67)

# Import cleaned data
df_clean <- read.csv("df_clean.csv")
df_clean <- df_clean %>%
  mutate(across(where(is.character), as_factor))
df_clean$Churn <- factor(df_clean$Churn, levels = c("Yes", "No"))

# # Sanity check (the first factor level must be the positive event when using the `caret` package)
# levels(df_clean$Churn)[1] == "Yes"
# levels(df_clean$Churn)[2] == "No"

# Use 80/20 split for train/test data
train_index <- createDataPartition(df_clean$Churn, p = 0.8, list = FALSE)
train <- df_clean[train_index, ]
test <- df_clean[-train_index, ]

# -----------------------------------------------------------------------------
# Logistic Regression (Using `caret`)
# -----------------------------------------------------------------------------
tr_control_logistic <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

# Implement logistic regression
mod_logistic <- train(
  Churn ~ ., 
  data = train, 
  method = "glm", 
  family = "binomial", 
  trControl = tr_control_logistic, 
  metric = "ROC"
)
print(mod_logistic)
results_logistic <- mod_logistic$results
results_logistic

# Make predictions using test data
pred_logistic <- predict(
  object = mod_logistic, 
  newdata = test
)

# Create confusion matrix and show accuracy
cm_logistic <- confusionMatrix(
  data = pred_logistic, 
  reference = test$Churn, 
  positive = "Yes"
)
cm_logistic
acc_logistic <- cm_logistic$overall['Accuracy']
acc_logistic

# -----------------------------------------------------------------------------
# Logistic Regression with Stepwise Feature Selection (Using `caret`)
# -----------------------------------------------------------------------------
tr_control_stepwise <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

# Implement logistic regression with stepwise selection
mod_stepwise_both <- train(
  Churn ~ ., 
  data = train, 
  method = "glmStepAIC", 
  family = "binomial", 
  direction = "both", 
  trControl = tr_control_stepwise, 
  metric = "ROC"
)
print(mod_stepwise_both)
results_stepwise_both <- mod_stepwise_both$results
results_stepwise_both

# Make predictions using test data
pred_stepwise_both <- predict(
  object = mod_stepwise_both, 
  newdata = test
)

# Create confusion matrix and show accuracy
cm_stepwise_both <- confusionMatrix(
  data = pred_stepwise_both, 
  reference = test$Churn, 
  positive = "Yes"
)
cm_stepwise_both
acc_stepwise_both <- cm_stepwise_both$overall['Accuracy']
acc_stepwise_both

# -----------------------------------------------------------------------------
# Lasso Regression (Using `caret`)
# -----------------------------------------------------------------------------
tr_control_lasso <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

grid_lasso <- expand.grid(
  alpha = 1, 
  lambda = 10^seq(2, -3, length = 100)
)

# Implement lasso regression
mod_lasso <- train(
  Churn ~ ., 
  data = train, 
  method = "glmnet", 
  family = "binomial", 
  trControl = tr_control_lasso, 
  tuneGrid = grid_lasso, 
  preProcess = c("center", "scale"), 
  metric = "ROC"
)
print(mod_lasso)
best_tune_lasso <- mod_lasso$bestTune
best_lambda_lasso <- best_tune_lasso$lambda
results_lasso <- mod_lasso$results
results_best_lambda_lasso <- results_lasso %>%
  filter(lambda == best_lambda_lasso)
results_best_lambda_lasso

# Make predictions using test data
pred_lasso <- predict(
  object = mod_lasso, 
  newdata = test
)

# Create confusion matrix and show accuracy
cm_lasso <- confusionMatrix(
  data = pred_lasso, 
  reference = test$Churn, 
  positive = "Yes"
)
cm_lasso
acc_lasso <- cm_lasso$overall['Accuracy']
acc_lasso

# -----------------------------------------------------------------------------
# Ridge Regression (Using `caret`)
# -----------------------------------------------------------------------------
tr_control_ridge <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

grid_ridge <- expand.grid(
  alpha = 0, 
  lambda = 10^seq(2, -3, length = 100)
)

# Implement ridge regression
mod_ridge <- train(
  Churn ~ ., 
  data = train, 
  method = "glmnet", 
  family = "binomial", 
  trControl = tr_control_ridge, 
  tuneGrid = grid_ridge, 
  preProcess = c("center", "scale"), 
  metric = "ROC"
)
print(mod_ridge)
best_tune_ridge <- mod_ridge$bestTune
best_lambda_ridge <- best_tune_ridge$lambda
results_ridge <- mod_ridge$results
results_best_lambda_ridge <- results_ridge %>%
  filter(lambda == best_lambda_ridge)
results_best_lambda_ridge

# Make predictions using test data
pred_ridge <- predict(
  object = mod_ridge, 
  newdata = test
)

# Create confusion matrix and show accuracy
cm_ridge <- confusionMatrix(
  data = pred_ridge, 
  reference = test$Churn, 
  positive = "Yes"
)
cm_ridge
acc_ridge <- cm_ridge$overall['Accuracy']
acc_ridge

# -----------------------------------------------------------------------------
# Decision Tree (Using `caret`)
# -----------------------------------------------------------------------------
tr_control_tree <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

grid_tree = expand.grid(
  cp = seq(0.001, 0.1, length.out = 20)
)

# Implement standard decision tree
mod_tree <- train(
  Churn ~ ., 
  data = train, 
  method = "rpart", 
  trControl = tr_control_tree, 
  tuneGrid = grid_tree, 
  metric = "ROC"
)
print(mod_tree)
best_cp_tree <- mod_tree$bestTune$cp
results_tree <- mod_tree$results
results_best_cp_tree <- results_tree %>%
  filter(cp == best_cp_tree)

# Plot the fitted tree
rpart.plot(
  mod_tree$finalModel, 
  main = "Decision Tree"
)

# Make predictions using test data
pred_tree <- predict(
  object = mod_tree, 
  newdata = test
)

# Create confusion matrix and show accuracy
cm_tree <- confusionMatrix(
  data = pred_tree, 
  reference = test$Churn
)
cm_tree
acc_tree <- cm_tree$overall['Accuracy']
acc_tree

# -----------------------------------------------------------------------------
# Bagging (Using `caret`)
# -----------------------------------------------------------------------------
tr_control_bagging <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

# Implement bagging
mod_bagging <- train(
  Churn ~ ., 
  data = train, 
  method = "treebag", 
  trControl = tr_control_bagging, 
  metric = "ROC"
)
print(mod_bagging)
results_bagging <- mod_bagging$results
results_bagging

# Make predictions using test data
pred_bagging <- predict(
  object = mod_bagging, 
  newdata = test
)

# Create confusion matrix and show accuracy
cm_bagging <- confusionMatrix(
  data = pred_bagging, 
  reference = test$Churn
)
cm_bagging
acc_bagging <- cm_bagging$overall['Accuracy']
acc_bagging

# -----------------------------------------------------------------------------
# Random Forest (Using `caret`)
# -----------------------------------------------------------------------------
tr_control_rf <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

grid_rf <- expand.grid(
  mtry = c(2, 4, 6, 8, 10), 
  n.trees = c(100, 500, 1000)
)

# Implement random forest
mod_rf <- train(
  Churn ~ ., 
  data = train, 
  method = "rf", 
  trControl = tr_control_rf, 
  tuneGrid = grid_rf, 
  metric = "ROC"
)
print(mod_rf)
best_tune_rf <- mod_rf$bestTune
best_mtry_rf <- best_tune_rf$mtry
results_rf <- mod_rf$results
results_best_mtry_rf <- results_rf %>%
  filter(mtry == best_mtry_rf)
results_best_mtry_rf

# Make predictions using test data
pred_rf <- predict(
  object = mod_rf, 
  newdata = test
)

# Create confusion matrix and show accuracy
cm_rf <- confusionMatrix(
  data = pred_rf, 
  reference = test$Churn, 
  positive = "Yes")
cm_rf
acc_rf <- cm_rf$overall['Accuracy']
acc_rf

# -----------------------------------------------------------------------------
# Boosting (Using `caret`)
# -----------------------------------------------------------------------------
tr_control_boosting <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

grid_boosting <- expand.grid(
  interaction.depth = c(1, 3, 5), 
  n.trees = c(100, 500, 1000), 
  shrinkage = c(0.01, 0.05, 0.1), 
  n.minobsinnode = c(10, 20)
)

# Implement boosting
mod_boosting <- train(
  Churn ~ ., 
  data = train, 
  method = "gbm", 
  distribution = "bernoulli", 
  trControl = tr_control_boosting, 
  tuneGrid = grid_boosting, 
  metric = "ROC", 
  verbose = TRUE
)
print(mod_boosting)
best_tune_boosting <- mod_boosting$bestTune

# Make predictions using test data
pred_boosting <- predict(
  object = mod_boosting, 
  newdata = test
)

# Create confusion matrix and show accuracy
cm_boosting <- confusionMatrix(
  data = pred_boosting, 
  reference = test$Churn, 
  positive = "Yes"
)
cm_boosting
acc_boosting <- cm_boosting$overall['Accuracy']
acc_boosting

# -----------------------------------------------------------------------------
# BART (Using `BART`)
# -----------------------------------------------------------------------------
# Implement BART
mod_bart <- pbart(
  x.train = x_train, 
  y.train = y_train, 
  x.test = x_test, 
  ntree = 200, 
  nskip = 100, 
  ndpost = 1000
)

# Make predictions
pred_probs_bart <- predict(
  object = mod_bart, 
  newdata = x_test
)
pred_probs_bart <- pred_probs_bart$prob.test.mean
pred_class_bart <- ifelse(pred_probs_bart > 0.5, 1, 0)

# Show confusion matrix and classification error rate
cm_bart <- table(
  Predicted = pred_class_bart, 
  Actual = test$Churn
)
cm_bart
class_error_bart <- 1 - (sum(diag(cm_bart)) / sum(cm_bart))   # 19.20% of the observations were misclassified.

# -----------------------------------------------------------------------------
# SVMs
# -----------------------------------------------------------------------------
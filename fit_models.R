# -----------------------------------------------------------------------------
# Predictive Analytics Project - Fit Models
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Possible Methods
# -----------------------------------------------------------------------------
# Logistic regression
# LDA and QDA
# KNN classifier
# Naive Bayes
# Ridge and Lasso regression
# Linear, cubic, and smoothing splines
# Decision tree
# Random forest
# Bagging and boosting
# BART
# SVMs

# Import libraries
library(BART)           # for BART
library(caret)          # for train() function
library(MASS)           # for forward and backward selection
library(randomForest)   # for bagging
library(tidyverse)      # for data cleaning
library(tree)           # for standard tree

# Set the seed for reproducibility
set.seed(67)

# Import cleaned data
df_clean <- read.csv("df_clean.csv")
df_clean <- df_clean %>%
  mutate(across(where(is.character), as_factor))

# Use 80/20 split for train/test data
train_index <- createDataPartition(df_clean$Churn, p = 0.8, list = FALSE)
train <- df_clean[train_index, ]
test <- df_clean[-train_index, ]

# -----------------------------------------------------------------------------
# Logistic Regression (Using `caret`)
# -----------------------------------------------------------------------------
# Define CV method for logistic regression
tr_control_logistic <- trainControl(method = "cv", number = 10)

# Implement logistic regression
logistic_mod <- train(
  Churn ~ ., 
  data = train, 
  method = "glm", 
  family = "binomial", 
  trControl = tr_control_logistic, 
  metric = "Accuracy"
)
print(logistic_mod)
summary(logistic_mod$finalModel)

# Make predictions using test data
logistic_pred <- predict(logistic_mod, newdata = test)

# Create and show confusion matrix
logistic_conf_mat <- confusionMatrix(logistic_pred, test$Churn)
logistic_conf_mat

# -----------------------------------------------------------------------------
# Feature Selection Using Forward and Backward Selection (Using `MASS`)
# -----------------------------------------------------------------------------
# Defining the null and full models for forward and backward selection, respectively
null_model <- glm(Churn ~ 1, data = train, family = binomial)
full_model <- glm(Churn ~ ., data = train, family = binomial)

# Forward Selection
forward_model <- stepAIC(
  null_model,
  direction = "forward",
  scope = list(lower = ~ 1, upper = ~ .),
  trace = 0,
  k = 2
)

# Backward Selection
backward_model <- stepAIC(
  full_model, 
  direction = "backward", 
  trace = 0, 
  k = 2
)

# View model summaries
summary(forward_model)
summary(backward_model)

# Make predictions using both models
pred_class_forward <- predict(forward_model, newdata = test, type = "response")
pred_class_forward <- ifelse(pred_class_forward > 0.5, 1, 0)

pred_class_backward <- predict(backward_model, newdata = test, type = "response")
pred_class_backward <- ifelse(pred_class_backward > 0.5, 1, 0)

# Prepare data for confusion matrix
pred_class_forward <- as.factor(unname(pred_class_forward))
pred_class_backward <- as.factor(unname(pred_class_backward))
actual_vec <- as.factor(ifelse(test$Churn == "Yes", 1, 0))

# Make confusion matrices
conf_mat_forward <- confusionMatrix(
  data = pred_class_forward, 
  reference = actual_vec, 
  positive = "1"
)
conf_mat_forward

conf_mat_backward <- confusionMatrix(
  data = pred_class_backward, 
  reference = actual_vec, 
  positive = "1"
)
conf_mat_backward

# Get classification error
class_error_forward <- 1 - unname(conf_mat_forward$overall["Accuracy"])
class_error_forward
# The forward model always predicts "No". Since most of the observations didn't churn, this model gets ~75% accuracy without having learned anything.

class_error_backward <- 1 - unname(conf_mat_backward$overall["Accuracy"])
class_error_backward
# 19.70% of the observations were misclassified.

# -----------------------------------------------------------------------------
# Lasso and Ridge Regression (Using `caret`)
# -----------------------------------------------------------------------------
tr_control_glmnet <- trainControl(method = "cv", number = 10)

grid_lasso <- expand.grid(alpha = 1, lambda = 10^seq(2, -3, length = 100))
grid_ridge <- expand.grid(alpha = 0, lambda = 10^seq(2, -3, length = 100))

# Implement lasso regression
lasso_mod <- train(
  Churn ~ ., 
  data = train, 
  method = "glmnet", 
  trControl = tr_control_glmnet, 
  tuneGrid = grid_lasso, 
  preProcess = c("center", "scale"), 
  metric = "Accuracy"
)
print(lasso_mod)
print(lasso_mod$bestTune)
coef(lasso_mod$finalModel, lasso_mod$bestTune$lambda)

# Implement ridge regression
ridge_mod <- train(
  Churn ~ ., 
  data = train, 
  method = "glmnet", 
  trControl = tr_control_glmnet, 
  tuneGrid = grid_ridge, 
  preProcess = c("center", "scale"), 
  metric = "Accuracy"
)
print(ridge_mod)
print(ridge_mod$bestTune)
coef(ridge_mod$finalModel, ridge_mod$bestTune$lambda)

# Make predictions using test data
lasso_mod_pred <- predict(lasso_mod, newdata = test)
ridge_mod_pred <- predict(ridge_mod, newdata = test)

# Create and show confusion matrix
lasso_mod_conf_mat <- confusionMatrix(lasso_mod_pred, test$Churn)
lasso_mod_conf_mat

ridge_mod_conf_mat <- confusionMatrix(ridge_mod_pred, test$Churn)
ridge_mod_conf_mat

# -----------------------------------------------------------------------------
# Decision Tree (Using `tree`)
# -----------------------------------------------------------------------------
# Create and plot the tree
standard_tree <- tree(Churn ~ ., data = train)
summary(standard_tree)
plot(standard_tree)
text(standard_tree, pretty = 0, digits = 1, cex = 0.5)

# Make predictions
pred_standard_tree <- predict(standard_tree, newdata = test, type = "class")

# Show confusion matrix and classification error rate
conf_matrix_standard_tree <- table(Predicted = pred_standard_tree, Actual = test$Churn)
conf_matrix_standard_tree
class_error_standard_tree <- 1 - (sum(diag(conf_matrix_standard_tree)) / sum(conf_matrix_standard_tree))   # 20.41% of the observations were misclassified.

# -----------------------------------------------------------------------------
# Prune the Decision Tree (Using `tree`)
# -----------------------------------------------------------------------------
# Find optimal number of terminal nodes via CV
cv_results_tree <- cv.tree(standard_tree, FUN = prune.misclass, K = 10)

best_size <- cv_results_tree$size[which.min(cv_results_tree$dev)]

pruned_model <- prune.misclass(standard_tree, best = best_size)

# Plot the pruned tree
plot(pruned_model)
text(pruned_model, pretty = 0, digits = 1, cex = 0.5)

# Make predictions
pred_pruned_tree <- predict(pruned_model, newdata = test, type = "class")

# Show confusion matrix and classification error rate
conf_matrix_pruned_tree <- table(Predicted = pred_pruned_tree, Actual = test$Churn)
conf_matrix_pruned_tree
class_error_pruned_tree <- 1 - (sum(diag(conf_matrix_pruned_tree)) / sum(conf_matrix_pruned_tree))   # 20.41% of the observations were misclassified.

# The standard tree and the pruned tree are identical.

# -----------------------------------------------------------------------------
# Bagging (Using 'randomForest`)
# -----------------------------------------------------------------------------
# Implement bagging
bagging <- randomForest(
  Churn ~ ., 
  data = train, 
  mtry = ncol(train) - 1, 
  ntree = 100, 
  importance = TRUE
)

# Make predictions
pred_bagging <- predict(bagging, newdata = test, type = "class")

# Show confusion matrix and classification error rate
conf_matrix_bagging <- table(Predicted = pred_bagging, Actual = test$Churn)
conf_matrix_bagging
class_error_bagging <- 1 - (sum(diag(conf_matrix_bagging)) / sum(conf_matrix_bagging))   # 22.33% of the observations were misclassified.

importance(bagging)

# -----------------------------------------------------------------------------
# Random Forest (Using `caret`)
# -----------------------------------------------------------------------------
# Define CV method for random forest
tr_control_rf <- trainControl(method = "cv", number = 5)

# Define tuning parameters to try
grid_rf <- expand.grid(mtry = c(2, 4, 6, 8, 10))

# Implement random forest
rf_mod <- train(
  Churn ~ ., 
  data = train, 
  method = "rf", 
  trControl = tr_control_rf, 
  tuneGrid = grid_rf, 
  metric = "Accuracy"
)
print(rf_mod)

# Make predictions using test data
rf_pred <- predict(rf_mod, newdata = test)

# Create and show confusion matrix
rf_conf_mat <- confusionMatrix(rf_pred, test$Churn)
rf_conf_mat

# -----------------------------------------------------------------------------
# Boosting (Using `caret`)
# -----------------------------------------------------------------------------
# Define CV method for boosting
tr_control_boosting <- trainControl(method = "cv", number = 5)

# Define tuning parameters to try
grid_boosting <- expand.grid(
  interaction.depth = c(1, 3, 5), 
  n.trees = c(100, 500, 1000), 
  shrinkage = c(0.01, 0.05, 0.1), 
  n.minobsinnode = c(10, 20)
)

# Implement boosting
boosting_mod <- train(
  Churn ~ ., 
  data = train, 
  method = "gbm", 
  distribution = "bernoulli", 
  trControl = tr_control_boosting, 
  tuneGrid = grid_boosting, 
  metric = "Accuracy", 
  verbose = TRUE
)
print(boosting_mod)
print(boosting_mod$bestTune)

# Make predictions using test data
boosting_mod_pred <- predict(boosting_mod, newdata = test)

# Create and show confusion matrix
boosting_mod_conf_mat <- confusionMatrix(boosting_mod_pred, test$Churn)
boosting_mod_conf_mat

# -----------------------------------------------------------------------------
# BART (Using `BART`)
# -----------------------------------------------------------------------------
bart_mod <- pbart(
  x.train = x_train, 
  y.train = y_train, 
  x.test = x_test, 
  ntree = 200, 
  nskip = 100, 
  ndpost = 1000
)

# Make predictions
pred_probs_bart <- predict(bart_mod, newdata = x_test)
pred_probs_bart <- pred_probs_bart$prob.test.mean
pred_class_bart <- ifelse(pred_probs_bart > 0.5, 1, 0)

# Show confusion matrix and classification error rate
conf_matrix_bart <- table(Predicted = pred_class_bart, Actual = test$Churn)
conf_matrix_bart
class_error_bart <- 1 - (sum(diag(conf_matrix_bart)) / sum(conf_matrix_bart))   # 19.20% of the observations were misclassified.
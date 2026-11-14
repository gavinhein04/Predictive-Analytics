# ------------------------------------------------------------
# Predictive Analytics Project - Fit Models
# ------------------------------------------------------------

# Predict churn based on response variables
# Many different methods to try
# Use cross-validation (validation set approach, K-fold CV, or LOOCV)

# ------------------------------------------------------------
# Methods to Choose From
# ------------------------------------------------------------
# Logistic regression (standard and polynomial)
# LDA and QDA
# KNN classifier
# Naive Bayes
# Ridge and Lasso regression
# Linear, cubic, and smoothing splines
# Decision tree
# Random forest
# Bagging and boosting
# BART

# Import libraries
library(BART)           # for BART
library(boot)           # for K-fold CV
library(gbm)            # for boosting
library(glmnet)         # for ridge and lasso regression
library(MASS)           # for forward and backward selection
library(randomForest)   # for random forests and bagging
library(tidyverse)      # tidyverse
library(tree)           # for standard tree

# Set the seed for reproducibility
set.seed(67)

# Import cleaned data
df_clean <- read.csv("df_clean.csv")
df_clean <- df_clean %>%
  mutate(across(where(is.character), as_factor))

# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------
my_data <- df_clean

# Fitting the full model
mod_logistic <- glm(Churn ~ ., data = my_data, family = binomial)
summary(mod_logistic)
mod_logistic_aic <- mod_logistic$aic

# Defining the cost function for K-fold CV
cost_func <- function(obs, pred) {
  mean((pred < 0.5) & obs==1 | (pred > 0.5) & obs==0)
}

# 10-fold CV
cv_10_fold <- cv.glm(data = my_data, glmfit = mod_logistic, cost = cost_func, K = 10)
test_err_est_10_fold <- cv_10_fold$delta[2]   # 19.58% of the observations were misclassified.

# Defining the null and full models for forward and backward selection, respectively
null_model <- glm(Churn ~ 1, data = my_data, family = binomial)
full_model <- mod_logistic

# Forward Selection
forward_model <- stepAIC(null_model, 
                         direction = "forward", 
                         scope = list(lower = ~ 1, upper = ~ .), 
                         trace = 0, 
                         k = 2)

# Backward Selection
backward_model <- stepAIC(full_model, 
                          direction = "backward", 
                          trace = 0, 
                          k = 2)

# View model summaries and AIC values
summary(forward_model)
summary(backward_model)

forward_aic <- forward_model$aic
backward_aic <- backward_model$aic

# ------------------------------------------------------------
# Lasso and Ridge Regression
# ------------------------------------------------------------
# Use 80/20 split for train/test data
train_size <- round(0.8 * nrow(df_clean))
train_indices <- sample(seq_len(nrow(df_clean)), size = train_size)
train <- df_clean[train_indices, ]
test <- df_clean[-train_indices, ]

# Preparing data
x_train <- model.matrix(Churn ~ ., data = train)[, -1]
y_train <- train$Churn

x_test <- model.matrix(Churn ~ ., data = test)[, -1]
y_test <- test$Churn

# Lasso (alpha = 1)
cv_lasso <- cv.glmnet(x = x_train, 
                      y = y_train, 
                      family = "binomial", 
                      alpha = 1, 
                      type.measure = "class")

# Ridge (alpha = 0)
cv_ridge <- cv.glmnet(x = x_train, 
                      y = y_train, 
                      family = "binomial", 
                      alpha = 0, 
                      type.measure = "class")

optimal_lambda_lasso <- cv_lasso$lambda.min
optimal_lambda_ridge <- cv_ridge$lambda.min

pred_class_lasso <- predict(cv_lasso, newx = x_test, s = "lambda.min", type = "class")
pred_class_ridge <- predict(cv_ridge, newx = x_test, s = "lambda.min", type = "class")

conf_matrix_lasso <- table(Predicted = pred_class_lasso, Actual = y_test)
conf_matrix_lasso
class_error_lasso <- 1 - (sum(diag(conf_matrix_lasso)) / sum(conf_matrix_lasso))   # 19.56% of the observations were misclassified.

conf_matrix_ridge <- table(Predicted = pred_class_ridge, Actual = y_test)
conf_matrix_ridge
class_error_ridge <- 1 - (sum(diag(conf_matrix_ridge)) / sum(conf_matrix_ridge))   # 19.35% of the observations were misclassified.

# ------------------------------------------------------------
# Decision Tree
# ------------------------------------------------------------
standard_tree <- tree(Churn ~ ., data = train)
summary(standard_tree)
plot(standard_tree)
text(standard_tree, pretty = 0, digits = 1, cex = 0.5)

pred_standard_tree <- predict(standard_tree, newdata = test, type = "class")

conf_matrix_standard_tree <- table(Predicted = pred_standard_tree, Actual = test$Churn)
conf_matrix_standard_tree
class_error_standard_tree <- 1 - (sum(diag(conf_matrix_standard_tree)) / sum(conf_matrix_standard_tree))   # 20.41% of the observations were misclassified.

# ------------------------------------------------------------
# Prune the Decision Tree
# ------------------------------------------------------------
cv_results_tree <- cv.tree(standard_tree, FUN = prune.misclass, K = 10)

best_size <- cv_results_tree$size[which.min(cv_results_tree$dev)]

pruned_model <- prune.misclass(standard_tree, best = best_size)

plot(pruned_model)
text(pruned_model, pretty = 0, digits = 1, cex = 0.5)

pred_pruned_tree <- predict(pruned_model, newdata = test, type = "class")

conf_matrix_pruned_tree <- table(Predicted = pred_pruned_tree, Actual = test$Churn)
conf_matrix_pruned_tree
class_error_pruned_tree <- 1 - (sum(diag(conf_matrix_pruned_tree)) / sum(conf_matrix_pruned_tree))   # 20.41% of the observations were misclassified.

# The standard tree and the pruned tree are identical.

# ------------------------------------------------------------
# Bagging
# ------------------------------------------------------------
bagging <- randomForest(Churn ~ ., 
                        data = train, 
                        mtry = ncol(train) - 1,   # consider all p predictors at each split
                        ntree = 100, 
                        importance = TRUE)

pred_bagging <- predict(bagging, newdata = test, type = "Class")

conf_matrix_bagging <- table(Predicted = pred_bagging, Actual = test$Churn)
conf_matrix_bagging
class_error_bagging <- 1 - (sum(diag(conf_matrix_bagging)) / sum(conf_matrix_bagging))   # 22.33% of the observations were misclassified.

importance(bagging)

# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------
random_forest <- randomForest(Churn ~ ., 
                              data = train, 
                              mtry = sqrt(ncol(train) - 1),   # consider sqrt(p) predictors at each split
                              ntree = 100, 
                              importance = TRUE)

pred_random_forest <- predict(random_forest, newdata = test, type = "Class")

conf_matrix_random_forest <- table(Predicted = pred_random_forest, Actual = test$Churn)
conf_matrix_random_forest
class_error_random_forest <- 1 - (sum(diag(conf_matrix_random_forest)) / sum(conf_matrix_random_forest))   # 20.34% of the observations were misclassified.

# ------------------------------------------------------------
# Boosting
# ------------------------------------------------------------


# ------------------------------------------------------------
# BART
# ------------------------------------------------------------
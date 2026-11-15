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
library(caret)          # for confusionMatrix() function
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
# Fitting the full model
mod_logistic <- glm(Churn ~ ., data = df_clean, family = binomial)
summary(mod_logistic)
mod_logistic_aic <- mod_logistic$aic

# Defining the cost function for K-fold CV
cost_func <- function(obs, pred) {
  mean((pred < 0.5) & obs==1 | (pred > 0.5) & obs==0)
}

# 10-fold CV
cv_10_fold <- cv.glm(data = df_clean, glmfit = mod_logistic, cost = cost_func, K = 10)
test_err_est_10_fold <- cv_10_fold$delta[2]   # 19.58% of the observations were misclassified.

# ------------------------------------------------------------
# Feature Selection Using Forward and Backward Selection
# ------------------------------------------------------------
# Use 80/20 split for train/test data
train_size <- round(0.8 * nrow(df_clean))
train_indices <- sample(seq_len(nrow(df_clean)), size = train_size)
train <- df_clean[train_indices, ]
test <- df_clean[-train_indices, ]

# Defining the null and full models for forward and backward selection, respectively
null_model <- glm(Churn ~ 1, data = train, family = binomial)
full_model <- glm(Churn ~ ., data = train, family = binomial)

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
forward_aic

backward_aic <- backward_model$aic
backward_aic

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
conf_mat_forward <- confusionMatrix(data = pred_class_forward, reference = actual_vec, positive = "1")
conf_mat_forward

conf_mat_backward <- confusionMatrix(data = pred_class_backward, reference = actual_vec, positive = "1")
conf_mat_backward

# Get classification error
class_error_forward <- 1 - unname(conf_mat_forward$overall["Accuracy"])
class_error_forward
# The forward model always predicts "No". Since most of the observations didn't churn, this model gets ~75% accuracy without having learned anything.

class_error_backward <- 1 - unname(conf_mat_backward$overall["Accuracy"])
class_error_backward   # 19.70% of the observations were misclassified.

# ------------------------------------------------------------
# Lasso and Ridge Regression
# ------------------------------------------------------------
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

# Show optimal lambda determined through CV
optimal_lambda_lasso <- cv_lasso$lambda.min
optimal_lambda_ridge <- cv_ridge$lambda.min

# Make predictions
pred_class_lasso <- predict(cv_lasso, newx = x_test, s = "lambda.min", type = "class")
pred_class_ridge <- predict(cv_ridge, newx = x_test, s = "lambda.min", type = "class")

# Show confusion matrices and classification error rates
conf_matrix_lasso <- table(Predicted = pred_class_lasso, Actual = y_test)
conf_matrix_lasso
class_error_lasso <- 1 - (sum(diag(conf_matrix_lasso)) / sum(conf_matrix_lasso))   # 19.56% of the observations were misclassified.

conf_matrix_ridge <- table(Predicted = pred_class_ridge, Actual = y_test)
conf_matrix_ridge
class_error_ridge <- 1 - (sum(diag(conf_matrix_ridge)) / sum(conf_matrix_ridge))   # 19.35% of the observations were misclassified.

# ------------------------------------------------------------
# Decision Tree
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Prune the Decision Tree
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Bagging
# ------------------------------------------------------------
# Implement bagging
bagging <- randomForest(Churn ~ ., 
                        data = train, 
                        mtry = ncol(train) - 1,   # consider all p predictors at each split
                        ntree = 100, 
                        importance = TRUE)

# Make predictions
pred_bagging <- predict(bagging, newdata = test, type = "class")

# Show confusion matrix and classification error rate
conf_matrix_bagging <- table(Predicted = pred_bagging, Actual = test$Churn)
conf_matrix_bagging
class_error_bagging <- 1 - (sum(diag(conf_matrix_bagging)) / sum(conf_matrix_bagging))   # 22.33% of the observations were misclassified.

importance(bagging)

# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------
# Create random forest
random_forest <- randomForest(Churn ~ ., 
                              data = train, 
                              mtry = sqrt(ncol(train) - 1),   # consider sqrt(p) predictors at each split
                              ntree = 100, 
                              importance = TRUE)

# Make predictions
pred_random_forest <- predict(random_forest, newdata = test, type = "class")

# Show confusion matrix and classification error rate
conf_matrix_random_forest <- table(Predicted = pred_random_forest, Actual = test$Churn)
conf_matrix_random_forest
class_error_random_forest <- 1 - (sum(diag(conf_matrix_random_forest)) / sum(conf_matrix_random_forest))   # 20.34% of the observations were misclassified.

# ------------------------------------------------------------
# Boosting
# ------------------------------------------------------------
# Prepare data
train$Churn <- as.numeric(ifelse(train$Churn == "Yes", 1, 0))
test$Churn <- as.numeric(ifelse(test$Churn == "Yes", 1, 0))

# Implement boosting
boosting_mod <- gbm(Churn ~ ., 
                    data = train, 
                    distribution = "bernoulli", 
                    interaction.depth = 1, 
                    verbose = TRUE, 
                    cv.folds = 5, 
                    n.cores = NULL)

# Show best number of trees as determined by CV
num_trees_boosting <- gbm.perf(boosting_mod, method = "cv")
num_trees_boosting

# Make predictions
pred_prob_boosting <- predict(boosting_mod, 
                              newdata = test, 
                              n.trees = num_trees_boosting, 
                              type = "response")
pred_class_boosting <- ifelse(pred_prob_boosting > 0.5, 1, 0)

# Show confusion matrix and classification error rate
conf_matrix_boosting <- table(Predicted = pred_class_boosting, Actual = test$Churn)
conf_matrix_boosting
class_error_boosting <- 1 - (sum(diag(conf_matrix_boosting)) / sum(conf_matrix_boosting))   # 19.35% of the observations were misclassified.

# ------------------------------------------------------------
# BART
# ------------------------------------------------------------
bart_mod <- pbart(x.train = x_train, 
                  y.train = y_train, 
                  x.test = x_test, 
                  ntree = 200, 
                  nskip = 100,    # burn-in number 
                  ndpost = 1000   # number of posterior MCMC samples retained after the burn-in period.
)

# Make predictions
pred_probs_bart <- predict(bart_mod, newdata = x_test)
pred_probs_bart <- pred_probs_bart$prob.test.mean
pred_class_bart <- ifelse(pred_probs_bart > 0.5, 1, 0)

# Show confusion matrix and classification error rate
conf_matrix_bart <- table(Predicted = pred_class_bart, Actual = test$Churn)
conf_matrix_bart
class_error_bart <- 1 - (sum(diag(conf_matrix_bart)) / sum(conf_matrix_bart))   # 19.20% of the observations were misclassified.
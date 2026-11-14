# ------------------------------------------------------------
# Predictive Analytics Project - Fit Models
# ------------------------------------------------------------

# Predict churn based on response variables
# Many different methods to try
# Use cross-validation (validation set approach, K-fold CV, or LOOCV)

# ------------------------
# Methods
# ------------------------
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
library(boot)   # for K-fold CV

# Set the seed for reproducibility
set.seed(67)

# Import cleaned data
df_clean <- read.csv("df_clean.csv")
df_clean <- df_clean %>%
  mutate(across(where(is.character), as_factor))

# # Use 80/20 split for train/test data
# train_size <- round(0.8 * nrow(df_clean))
# train_indices <- sample(seq_len(nrow(df_clean)), size = train_size)
# train <- df_clean[train_indices, ]
# test <- df_clean[-train_indices, ]

# ------------------------
# Logistic regression
# ------------------------
my_data <- df_clean

mod_logistic <- glm(Churn ~ ., data = my_data, family = binomial)

cost_func <- function(obs, pred) {
  mean((pred < 0.5) & obs==1 | (pred > 0.5) & obs==0)
}

cv_kfold <- cv.glm(data = my_data, glmfit = mod_logistic, cost = cost_func, K = 10)

test_error_estimate <- cv_kfold$delta[2]   # 20% of the observations were misclassified.

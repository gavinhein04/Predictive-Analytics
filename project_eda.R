# ------------------------------------------------------------
# Predictive Analytics Project - EDA
# ------------------------------------------------------------

# 1) Find a dataset or datasets to answer a question
# 2) Explore the features and their relationships to each other and the dependent variable

# Idea: Customer Churn Prediction
# Predict which customers will stop using a subscription service or e-commerce platform before they actually cancel.

# Import libraries
library(tidyverse)

# Import data
# Dataframe source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data
df <- na.omit(read.csv("telco_customer_churn_df.csv"))

# Data cleaning
df_clean <- df %>%
  select(-customerID) %>%
  rename(Gender = gender, Tenure = tenure) %>%
  mutate(SeniorCitizen = case_when(
    SeniorCitizen == 1 ~ "Yes", 
    SeniorCitizen == 0 ~ "No"
  )) %>%
  mutate(across(where(is.character), as_factor))

# Write cleaned data to a CSV
write.csv(df_clean, file = "df_clean.csv", row.names = FALSE)

summary(df_clean)

# Examining the relationship between `Tenure` and `Churn`
ggplot(df_clean, aes(x = Tenure, fill = Churn)) + 
  geom_histogram(color = "white", bins = 25, show.legend = FALSE) + 
  labs(title = "Churn by Tenure", x = "Tenure", y = "Count") + 
  facet_wrap(~ Churn) + 
  scale_fill_manual(values = c("steelblue", "chartreuse4"))

ggplot(df_clean, aes(x = Churn, y = Tenure, fill = Churn)) + 
  geom_boxplot(outlier.size = 1.5, show.legend = FALSE) + 
  labs(title = "Churn by Tenure", x = "Churn", y = "Tenure") + 
  scale_fill_manual(values = c("steelblue", "chartreuse4"))

# Examining the relationship between `MonthlyCharges` and `Churn`
ggplot(df_clean, aes(x = MonthlyCharges, fill = Churn)) + 
  geom_histogram(color = "white", bins = 25, show.legend = FALSE) + 
  labs(title = "Churn by Monthly Charges", x = "Monthly Charges", y = "Count") + 
  facet_wrap(~ Churn) + 
  scale_fill_manual(values = c("steelblue", "chartreuse4"))

ggplot(df_clean, aes(x = Churn, y = MonthlyCharges, fill = Churn)) + 
  geom_boxplot(outlier.size = 1.5, show.legend = FALSE) + 
  labs(title = "Churn by Monthly Charges", x = "Count", y = "Monthly Charges") + 
  scale_fill_manual(values = c("steelblue", "chartreuse4"))

# Examining the relationship between `TotalCharges` and `Churn`
ggplot(df_clean, aes(x = TotalCharges, fill = Churn)) + 
  geom_histogram(color = "white", bins = 25, show.legend = FALSE) + 
  labs(title = "Churn by Total Charges", x = "Total Charges", y = "Count") + 
  facet_wrap(~ Churn) + 
  scale_fill_manual(values = c("steelblue", "chartreuse4"))

ggplot(df_clean, aes(x = Churn, y = TotalCharges, fill = Churn)) + 
  geom_boxplot(outlier.size = 1.0, show.legend = FALSE) + 
  labs(title = "Churn by Total Charges", x = "Count", y = "Total Charges") + 
  scale_fill_manual(values = c("steelblue", "chartreuse4"))

# Examining the relationship between `PaymentMethod` and `Churn`
ggplot(df_clean, aes(x = Churn, fill = PaymentMethod)) +
  geom_bar(position = "dodge", color = "black") +
  labs(title = "Churn by Payment Method", x = "Churn", y = "Count", fill = "Payment Method")

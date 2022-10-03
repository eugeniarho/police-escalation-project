install.packages(c("irr", "ggplot2", "dplyr", "plyr", "aod", "car", "mice", "caret", "Rcpp"))
library(aod)
library(ggplot2)
library(car)
library(mice)
library(caret)
library(Rcpp)
library(irr)
library(dplyr)
library(plyr)


data <- read.csv("Study 1 data.csv", stringsAsFactors = FALSE)

colnames(data)


# Logistic Regression Model 
mylogit <- glm(escalation ~ 
              correct_new_contains_intro_greeting +
                binary_correct_new_reason +
                correct_new_ID +
                correct_new_details +
                correct_order_total +
                correct_new_legitimacy +
                Sex + 
                officer_race3 +
                officer_gender +
                RateViolentCrime_tract +   
                RatePropertyCrime_tract + 
                RateNarcoticsCrime_tract,
               data = data, family = "binomial")


summary(mylogit)
confint(mylogit)
exp(coef(mylogit))
car::vif(mylogit)

# Stepwise AIC to select Logistic Regression Model
library(MASS)
step.model <- stepAIC(mylogit, trace = FALSE)
summary(step.model)
confint(step.model) 
exp(coef(step.model))
car::vif(step.model)










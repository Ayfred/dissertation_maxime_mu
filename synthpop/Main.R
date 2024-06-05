library("synthpop")

# Load the data.csv in the datasets/data.csv
setwd("/users/pgrad/mamu/Documents/first")

data <- read.csv("datasets/data.csv")

# Define the variables
vars <- c("Disease", "Fever", "Cough", "Fatigue", "Difficulty Breathing",
          "Age", "Gender", "Blood Pressure", "Cholesterol Level",
          "Outcome Variable")

data$Disease <- as.factor(data$Disease)
data$Fever <- as.factor(data$Fever)
data$Cough <- as.factor(data$Cough)
data$Fatigue <- as.factor(data$Fatigue)
data$Difficulty.Breathing <- as.factor(data$Difficulty.Breathing)
data$Gender <- as.factor(data$Gender)
data$Blood.Pressure <- as.factor(data$Blood.Pressure)
data$Cholesterol.Level <- as.factor(data$Cholesterol.Level)
data$Outcome.Variable <- as.factor(data$Outcome.Variable)

synthetic_data <- syn(data)

print(synthetic_data$syn)



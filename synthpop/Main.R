#--------------------------------------------------------#
# Sample synthesis with the provided dataset             #
#--------------------------------------------------------#

# Load the necessary package
library(synthpop)

# Load the data.csv in the datasets/data.csv
setwd("/users/pgrad/mamu/Documents/first")

# Create the data frame from the provided dataset
data <- read.csv("datasets/data.csv")

# Check the summary info about variables
codebook.syn(data)$tab

# Generate synthetic data
synthetic_data <- syn(data)

print(synthetic_data)

# Display summary of synthetic data
summary(synthetic_data)

# Compare the synthetic data to the original data
compare(synthetic_data, data, stat = "counts")



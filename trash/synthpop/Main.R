library(synthpop)

setwd("/users/pgrad/mamu/Documents/first")

data <- read.csv("datasets/data.csv")
#data <- read.csv("llama3-8b/results/synthetic_data_llama_3_8b.csv")

names(data) <- gsub("\\.", "", names(data))

# Convert character variables to factors
character_vars <- sapply(data, is.character)
data[, character_vars] <- lapply(data[, character_vars], as.factor)

missing_values <- sapply(data, function(x) sum(is.na(x) | x == "-9"))
print(missing_values)

# Check the summary info about variables
codebook.syn(data)$tab

mydata <- data[, c(1, 2, 3, 4, 5, 6, 7)] 

# Convert character variables to factors
character_vars <- sapply(mydata, is.character)
mydata[, character_vars] <- lapply(mydata[, character_vars], as.factor)

# Check the summary info about variables
codebook.syn(mydata)$tab

# Convert integer variables to numeric
integer_vars <- sapply(mydata, is.integer)
mydata[, integer_vars] <- lapply(mydata[, integer_vars], as.numeric)

dim(mydata)

# Generate synthetic data
synthetic_data <- syn(mydata)

print(synthetic_data)

# Display summary of synthetic data
# summary(synthetic_data)

# Extract the predictor matrix from the synthetic data object
predictor_matrix <- synthetic_data$predictor.matrix

# Convert the predictor matrix into a data frame
predictor_df <- as.data.frame(predictor_matrix)

# Write the predictor matrix data frame to a CSV file
write.csv(predictor_df, "synthpop/results/predictor_matrix_synthpop.csv", row.names = FALSE)

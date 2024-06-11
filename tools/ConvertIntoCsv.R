# Show the current working directory
print(getwd())

# Load the RData file
print("Load the RData file")
load("./datasets/data2.RData")

# Get the total number of rows in the data frame
total_rows <- nrow(df)

# Calculate the number of rows to keep (half of the total rows)
half_rows <- ceiling(total_rows / 50)

# Select the first half of the rows
df_half <- df[1:half_rows, ]

# Save the selected rows to a CSV file
print("Convert the data frame to a CSV file")
write.csv(df_half, "./datasets/data2.csv", row.names = FALSE)
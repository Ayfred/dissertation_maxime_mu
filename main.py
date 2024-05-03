import preprocessing
import pandas as pd


data = pd.read_csv('data.csv')

preprocess = preprocessing.Preprocessing(data)

# print the first 5 rows of the data

print(preprocess.get_X_train().head())
print(preprocess.get_y_train().head())
print(preprocess.get_X_test().head())
print(preprocess.get_y_test().head())

# print the shape of the data

print(preprocess.get_X_train().shape)
print(preprocess.get_y_train().shape)
print(preprocess.get_X_test().shape)
print(preprocess.get_y_test().shape)

# https://github.com/meta-llama/llama3/blob/main/download.sh
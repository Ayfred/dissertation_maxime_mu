# Preprocessing the data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

class Preprocessing:
    def __init__(self, data):
        self.data = data
        self.X = data.iloc[:, :-1]
        self.y = data.iloc[:, -1]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train_encoded = None
        self.y_test_encoded = None
        self.X_train_encoded = None
        self.X_test_encoded = None

        self.split_data(0.2)
        self.scale_data()
        self.encode_data()
        

    def split_data(self, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    def scale_data(self):
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

    def encode_data(self):
        le = LabelEncoder()
        self.y_train_encoded = le.fit_transform(self.y_train)
        self.y_test_encoded = le.transform(self.y_test)

        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
        self.X_train_encoded = np.array(ct.fit_transform(self.X_train))
        self.X_test_encoded = np.array(ct.transform(self.X_test))

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def get_X_train_scaled(self):
        return self.X_train_scaled

    def get_X_test_scaled(self):
        return self.X_test_scaled

    def get_y_train_encoded(self):
        return self.y_train_encoded

    def get_y_test_encoded(self):
        return self.y_test_encoded

    def get_X_train_encoded(self):
        return self.X_train_encoded

    def get_X_test_encoded(self):
        return self.X_test_encoded

    def get_data(self):
        return self.data

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    
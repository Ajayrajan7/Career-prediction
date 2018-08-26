import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('students_dataSet1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)



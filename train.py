#this is for training the model
import numpy as np
from sklearn.cross_validation import train_test_split

X, y = load_audio("AR", "JA")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

train_data = X_train, X_test
test_data = y_train, y_test

CNN.run_network(train_data, test_data)
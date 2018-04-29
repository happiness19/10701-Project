#this is for training the model
import numpy as np
from sklearn.cross_validation import train_test_split
import audio as load
import model as CNN

# X, y = load.load_audio("IT", "KO")
X, y = load.load_preprocessed_audio()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

train_data = X_train, y_train
test_data = X_test, y_test

CNN.run_network(train_data, test_data)
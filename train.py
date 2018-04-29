#this is for training the model

import numpy as np
from sklearn.cross_validation import train_test_split

X = np.load('usa373_span162_mfcc_13.npy')
y = np.append(np.ones(373), np.zeros(162))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)



import tensorflow as tf
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

def create_model():
    # create model
    model = tf.keras.Sequential([
        layers.Dense(12, input_dim=8, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt("diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Create and configure the KerasClassifier
model = KerasClassifier(model=create_model, epochs=150, batch_size=10, verbose=0)

# Configure the cross-validation procedure
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# Perform the cross-validation
results = cross_val_score(model, X, Y, cv=kfold)

print(f"Mean accuracy: {results.mean():.4f} (+/- {results.std() * 2:.4f})")
#
# # split into 67% for train and 33% for test
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=seed) # create model
#
# # # create model
# # model = tf.keras.Sequential([
# #     layers.Dense(12, input_dim=8, activation='relu'),
# #     layers.Dense(8, activation='relu'),
# #     layers.Dense(1, activation='sigmoid')
# # ])
#
#
#
# # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Fit the model
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=150, batch_size=10)
#
# # evaluate the model
# scores = model.evaluate(X, Y)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
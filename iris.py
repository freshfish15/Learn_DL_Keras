import numpy as np
import pandas
import np_utils
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import utils
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

dataframe = pandas.read_csv("IRIS.csv", skiprows=1)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
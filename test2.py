import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential

ANN_model=Sequential([
    Flatten(input_shape=(28,28)),
    Dense(32,activation='relu'),
    Dropout(0.2),
    Dense(10,activation='softmax')
])
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32)),
    GlobalAveragePooling2D(2,2),
    Conv2D(16, (6, 6), activation='relu'),
    GlobalAveragePooling2D((2, 2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()
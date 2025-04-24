import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set path to your current data folder
data_dir = os.path.join(os.getcwd(), 'data')

# Image data generator with rescaling only
datagen = ImageDataGenerator(rescale=1./255)

# Load all images from data_dir (damaged and good)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=8,
    class_mode='binary'
)

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model (no validation data here)
model.fit(train_generator, epochs=10)

# Save the model in recommended Keras format
model.save('ecommerce_model.keras')
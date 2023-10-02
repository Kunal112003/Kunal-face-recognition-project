import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 

# Define the path to the dataset
train_dir = r'C:\Users\ksr20\OneDrive\Desktop\Kunal-Coding-programming stuff\Kunal-face-recognition-project-1\Data\train'
test_dir = r'C:\Users\ksr20\OneDrive\Desktop\Kunal-Coding-programming stuff\Kunal-face-recognition-project-1\Data\test'

# Define the image data generator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the dataset using the flow_from_directory method
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical')

# Define the model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=50, validation_data=test_generator)

# Save the model
model.save('emotion_detection_model.h5')

# Load the model
model = tf.keras.models.load_model('emotion_detection_model.h5')

# Define the Haar Cascade classifier for face detection

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start capturing video from the webcam

video_capture = cv2.VideoCapture(0)

# Define the emotions label list
emotions_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

while True:
    
    
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
from module import build_multiscale_student  # Assuming module.py has build_multiscale_student function

#  preprocessing for ImageNet dataset
def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.0  # Resize to ImageNet standard and normalize
    label = tf.one_hot(label, 1000)  # One-hot encode for 1,000 classes
    return image, label

# Loading ImageNet dataset
test_data = tfds.load('imagenet2012', split='validation', as_supervised=True)
test_data = test_data.map(preprocess_image).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

# Initialize the student model with ImageNet specifications
input_shape = (224, 224, 3)
student_model = build_multiscale_student(input_shape, num_classes=1000)  # 1,000 output classes for ImageNet
student_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# trained weights
student_model.load_weights('student_model_weights.h5')  # Ensure this weights file is available

# Evaluating the student model on the ImageNet test data
loss, accuracy = student_model.evaluate(test_data, verbose=2)

# Saving evaluation results to the output directory
output_dir = r'D:\manuscript3\ImageNet\output'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

results_file = os.path.join(output_dir, 'validation_results.txt')
with open(results_file, 'w') as f:
    f.write(f"Test Loss: {loss}\n")
    f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")

print(f"Validation results saved to {results_file}")

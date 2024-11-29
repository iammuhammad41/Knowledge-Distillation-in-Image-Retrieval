import tensorflow as tf
import numpy as np
import os
from module import build_multiscale_student

# Loading and preprocess the ImageNet dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = tf.image.resize(tf.expand_dims(x_test, -1) / 255.0, [32, 32])

# Initializing the student model and load weights
input_shape = (32, 32, 1)
student_model = build_multiscale_student(input_shape)
student_model.load_weights(r'D:\manuscript3\ImageNet\models\student_model_weights.h5')

# Creating model that outputs intermediate layer for hash generation
hash_layer_model = tf.keras.Model(inputs=student_model.input, outputs=student_model.layers[-2].output)

# Generating hash codes for test set
hash_codes = np.sign(hash_layer_model.predict(x_test))

# Save hash codes to output directory
output_dir = r'D:\manuscript3\ImageNet\output'
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, 'test_hash_codes.npy'), hash_codes)

print(f"Hash codes saved to {output_dir}")

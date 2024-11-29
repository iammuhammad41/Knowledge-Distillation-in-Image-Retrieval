import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score
from module import build_multiscale_student
import tensorflow_datasets as tfds

#ImageNet validation set
def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.0
    return image, label

# test data from ImageNet and preprocess
test_data = tfds.load('imagenet2012', split='validation', as_supervised=True)
test_data = test_data.map(preprocess_image).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

# Initializing student model for feature extraction
input_shape = (224, 224, 3)
student_model = build_multiscale_student(input_shape, num_classes=1000)
student_intermediate = tf.keras.Model(inputs=student_model.input, outputs=student_model.outputs[:-1])

# Function to compute hash codes in batches
def compute_hash_codes(model, data):
    hash_codes = []
    labels = []
    for images, label_batch in data:
        intermediate_outputs = model.predict(images)
        # Convert intermediate output to binary hash code
        hash_batch = np.sign(intermediate_outputs[-1])  # Using the last layer's intermediate output
        hash_codes.append(hash_batch)
        labels.append(label_batch)
    return np.vstack(hash_codes), np.hstack(labels)

# Generating hash codes for retrieval
hash_codes, labels = compute_hash_codes(student_intermediate, test_data)

# Function to compute MAP
def compute_map(query_codes, retrieval_codes, labels, top_k=10):
    ap_scores = []
    for i, query in enumerate(query_codes):
        distances = np.sum(np.abs(query - retrieval_codes), axis=1)
        nearest_indices = np.argsort(distances)[:top_k]
        relevance = labels[nearest_indices] == labels[i]
        ap_scores.append(average_precision_score(relevance, relevance))
    return np.mean(ap_scores)

# Calculating MAP score
map_score = compute_map(hash_codes, hash_codes, labels.numpy())
print(f"Mean Average Precision (MAP): {map_score}")

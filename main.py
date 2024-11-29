import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
from module import build_multiscale_teacher, build_multiscale_student, total_loss


# Loading and preprocessing the ImageNet dataset
def preprocess_image(image, label):
    # Resize and normalize image
    image = tf.image.resize(image, [224, 224]) / 255.0
    label = tf.one_hot(label, 1000)
    return image, label


# training and validation data from ImageNet
train_data, test_data = tfds.load('imagenet2012', split=['train', 'validation'], as_supervised=True)

# Preprocessing data
train_data = train_data.map(preprocess_image).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.map(preprocess_image).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

# Initializing input shape for ImageNet
input_shape = (224, 224, 3)
teacher_model = build_multiscale_teacher(input_shape, num_classes=1000)
student_model = build_multiscale_student(input_shape, num_classes=1000)

# Compiling and train the teacher model
teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
teacher_model.fit(train_data, epochs=5, validation_data=test_data)

# Extracting intermediate outputs from teacher and student for distillation
teacher_intermediate = tf.keras.Model(inputs=teacher_model.input, outputs=teacher_model.outputs[:-1])
student_intermediate = tf.keras.Model(inputs=student_model.input, outputs=student_model.outputs[:-1])

# distillation training loop for student model
epochs = 5
alpha = 0.5

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    for x_batch, y_batch in train_data:
        with tf.GradientTape() as tape:
            student_preds = student_model(x_batch, training=True)
            teacher_outputs = teacher_intermediate(x_batch)
            student_outputs = student_intermediate(x_batch)

            #  hash codes for teacher and student
            teacher_hash = tf.sign(teacher_outputs[-1])
            student_hash = tf.sign(student_outputs[-1])

            # distillation loss
            loss = total_loss(y_batch, student_preds[-1], teacher_outputs, student_outputs, teacher_hash, student_hash,
                              alpha)

        # Updating student model weights
        grads = tape.gradient(loss, student_model.trainable_variables)
        student_model.optimizer.apply_gradients(zip(grads, student_model.trainable_variables))

# Applying pruning to the trained student model
pruning_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.2, decay_steps=1000,
                                                                 end_learning_rate=0.0)
pruned_student = prune_low_magnitude(student_model, pruning_schedule=pruning_schedule)

# Converting the pruned student model to an int8 quantized TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_student)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_student_model = converter.convert()

# Save model
with open(r"D:\manuscript2\ImageNet\models\quantized_student_model.tflite", "wb") as f:
    f.write(quantized_student_model)

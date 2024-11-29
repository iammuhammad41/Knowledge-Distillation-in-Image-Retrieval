import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import KLDivergence, MeanSquaredError, BinaryCrossentropy

# Building multi-output teacher model for ImageNet with 1000 classes
def build_multiscale_teacher(input_shape, num_classes=1000):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    outputs = []

    # Adding auxiliary outputs at different layers for multiscale distillation
    for layer_name in ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']:
        x = base_model.get_layer(layer_name).output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        outputs.append(x)

    # Final classification layer for 1000 classes
    final_output = Dense(num_classes, activation='softmax')(outputs[-1])
    outputs.append(final_output)

    return Model(inputs=base_model.input, outputs=outputs, name="Teacher_Model_Multiscale")

# Building a student model to receive intermediate outputs for ImageNet with 1000 classes
def build_multiscale_student(input_shape, num_classes=1000):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    outputs = []

    # Adding intermediate layers for receiving multiscale distillation
    for layer_name in ['block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']:
        x = base_model.get_layer(layer_name).output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        outputs.append(x)

    # Final classification layer for 1000 classes
    final_output = Dense(num_classes, activation='softmax')(outputs[-1])
    outputs.append(final_output)

    return Model(inputs=base_model.input, outputs=outputs, name="Student_Model_Multiscale")

# Loss functions for multi-output knowledge distillation
def label_loss(y_true, y_pred):
    return BinaryCrossentropy()(y_true, y_pred)

def intermediate_loss(teacher_outputs, student_outputs):
    kl_divergence = KLDivergence()
    return sum(kl_divergence(t, s) for t, s in zip(teacher_outputs[:-1], student_outputs[:-1]))

def pairwise_similarity_loss(teacher_hash, student_hash):
    mse = MeanSquaredError()
    return mse(teacher_hash, student_hash)

def total_loss(y_true, y_pred, teacher_outputs, student_outputs, teacher_hash, student_hash, alpha=0.5):
    l_label = label_loss(y_true, y_pred)
    l_intermediate = intermediate_loss(teacher_outputs, student_outputs)
    l_pairwise = pairwise_similarity_loss(teacher_hash, student_hash)
    return l_label + l_intermediate + alpha * l_pairwise

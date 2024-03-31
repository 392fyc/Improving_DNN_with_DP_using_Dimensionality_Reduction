from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentGaussianOptimizer
from tensorflow_privacy.privacy.analysis import rdp_accountant
import umap.umap_ as umap
from absl import app, flags, logging
import time
from sklearn.preprocessing import StandardScaler

# Define hyper-parameters
flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.5, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_integer('microbatches', 250, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_integer('umap_components', 100, 'Number of UMAP components if UMAP is applied')
flags.DEFINE_boolean('use_umap', True, 'Whether to use UMAP')
flags.DEFINE_float('delta', 1e-5, 'Delta for DP-SGD')
flags.DEFINE_integer('umap_n_neighbors', 30, 'n_neighbors parameter for UMAP, consider experimenting in the range of 5-50 based on dataset size and complexity')
flags.DEFINE_float('umap_min_dist', 0.01, 'min_dist parameter for UMAP, start with 0.1 and consider values down to 0.0 for tighter clustering or up to 0.99 for more global structure preservation')

FLAGS = flags.FLAGS


def compute_epsilon(steps):
    """Computes epsilon value for given hyperparameters."""
    if FLAGS.noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    # Assume the training set size is 60000
    sampling_probability = FLAGS.batch_size / 60000
    rdp = rdp_accountant.compute_rdp(q=sampling_probability,
                                     noise_multiplier=FLAGS.noise_multiplier,
                                     steps=steps,
                                     orders=orders)
    # Now handle all returned values properly
    epsilon, best_order, *rest = rdp_accountant.get_privacy_spent(orders, rdp, target_delta=FLAGS.delta)
    return epsilon


def load_data_with_augmentation(use_umap):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    if use_umap:
        # Add color channel dimension
        train_images = np.expand_dims(train_images, axis=-1)
        test_images = np.expand_dims(test_images, axis=-1)

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)

        # Fit the data generator
        datagen.fit(train_images)

        # Apply data augmentation
        train_images_augmented = []
        for batch in datagen.flow(train_images, batch_size=len(train_images), shuffle=False):
            train_images_augmented.append(batch)
            break  # Only take one augmented batch

        train_images_augmented = np.concatenate(train_images_augmented)

        # Flatten the images for UMAP
        train_images_flattened = train_images_augmented.reshape(train_images_augmented.shape[0], -1)
        test_images_flattened = test_images.reshape(test_images.shape[0], -1)

        umap_model = umap.UMAP(
            n_components=FLAGS.umap_components,
            n_neighbors=FLAGS.umap_n_neighbors,
            min_dist=FLAGS.umap_min_dist
        )
        train_images_umap = umap_model.fit_transform(train_images_flattened)
        test_images_umap = umap_model.transform(test_images_flattened)

        return train_images_umap, train_labels, test_images_umap, test_labels

    else:
        # Reshape the images for the model without UMAP
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
        return train_images, train_labels, test_images, test_labels


def create_model(input_shape):
    # Model architecture adjustment considerations
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),  # Added dropout for regularization
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),  # Added dropout
        Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),  # Increased complexity may help in capturing more subtle patterns in UMAP transformed data
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(10, activation='softmax')
    ])
    return model


def main(argv):
    del argv  # Unused.
    start_time = time.time()  # Capture start time

    train_data, train_labels, test_data, test_labels = load_data_with_augmentation(FLAGS.use_umap)

    if FLAGS.use_umap:
        input_shape = (FLAGS.umap_components,)
    else:
        input_shape = (28, 28, 1)

    model = create_model(input_shape)

    optimizer = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train model
    model.fit(train_data, train_labels, epochs=FLAGS.epochs, validation_data=(test_data, test_labels),
              batch_size=FLAGS.batch_size)

    # Calculate training time
    total_time = time.time() - start_time  # Calculate total time by subtracting start time from current time

    # Calculate epsilon
    steps = (train_data.shape[0] // FLAGS.batch_size) * FLAGS.epochs
    epsilon = compute_epsilon(steps)

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_data, test_labels)

    logging.info(f"Training finished in {total_time:.2f} seconds")
    logging.info(f"Test accuracy: {test_acc}")
    logging.info(f"Epsilon: {epsilon}")
    logging.info("Hyperparameters:")
    logging.info(
        f"Use UMAP: {FLAGS.use_umap}, DPSGD: {FLAGS.dpsgd}, Noise Multiplier: {FLAGS.noise_multiplier}, Epochs: {FLAGS.epochs}, UMAP Components: {FLAGS.umap_components if FLAGS.use_umap else 'N/A'}, UMAP n_neighbors: {FLAGS.umap_n_neighbors}, UMAP min_dist: {FLAGS.umap_min_dist}")


if __name__ == '__main__':
    app.run(main)

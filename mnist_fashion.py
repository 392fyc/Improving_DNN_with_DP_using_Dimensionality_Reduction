from __future__ import absolute_import, division, print_function

import csv
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
import time
from dp_optimizer import DPGradientDescentGaussianOptimizer
from dp_utils import get_epsilon_noise_multiplier, DynamicEpsilonPrintingCallback

flags.DEFINE_float('learning_rate', 0.00315, 'Learning rate for training')
flags.DEFINE_float('l2_norm_clip', 2.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 312, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_integer('microbatches', 312, 'Number of microbatches')
flags.DEFINE_integer('num_runs', 10, 'Number of training runs per epsilon')
flags.DEFINE_float('delta', 1e-5, 'Target delta for differential privacy')

FLAGS = flags.FLAGS
DELTA = 1e-5


def load_fashion_mnist():
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    return train_data, train_labels, test_data, test_labels


def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model


def train_model(model, train_data, train_labels, test_data, test_labels, noise_multiplier, target_epsilon):
    optimizer = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    eps_callback = DynamicEpsilonPrintingCallback(
        batch_size=FLAGS.batch_size,
        num_examples=len(train_data),
        initial_noise_multiplier=noise_multiplier,
        target_epsilon=target_epsilon,
        target_delta=FLAGS.delta,
        total_epochs=FLAGS.epochs
    )

    # Ensure that the batch size is consistent
    steps_per_epoch = len(train_data) // FLAGS.batch_size

    history = model.fit(
        train_data, train_labels,
        epochs=FLAGS.epochs,
        validation_data=(test_data, test_labels),
        batch_size=FLAGS.batch_size,
        steps_per_epoch=steps_per_epoch,  # Add this line
        callbacks=[eps_callback],
        verbose=1
    )

    test_acc = history.history['val_accuracy'][-1]
    final_epsilon = eps_callback.eps_history[-1]

    return test_acc, final_epsilon


def main(argv):
    del argv

    train_data, train_labels, test_data, test_labels = load_fashion_mnist()

    epsilon_list = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0]
    results = []

    for target_epsilon in epsilon_list:
        print(f"\nCalculating for epsilon: {target_epsilon:.4f}")
        steps = FLAGS.epochs * (len(train_data) // FLAGS.batch_size)
        noise_multiplier = get_epsilon_noise_multiplier(
            target_epsilon, steps, FLAGS.batch_size, len(train_data), FLAGS.delta)

        print(f"Calculated noise multiplier: {noise_multiplier:.4f}")

        for run in range(FLAGS.num_runs):
            print(f"Run {run + 1}/{FLAGS.num_runs}")
            start_time = time.time()

            model = create_model(train_data.shape[1:])
            test_acc, actual_epsilon = train_model(model, train_data, train_labels, test_data, test_labels,
                                                   noise_multiplier, target_epsilon)

            train_time = time.time() - start_time

            result = {
                'run': run + 1,
                'target_epsilon': target_epsilon,
                'actual_epsilon': actual_epsilon,
                'noise_multiplier': noise_multiplier,
                'test_accuracy': test_acc,
                'train_time': train_time,
            }
            results.append(result)

            print(f"Test accuracy: {test_acc:.4f}")
            print(f"Actual epsilon: {actual_epsilon:.2f}")
            print(f"Training time: {train_time:.2f} seconds")

    # Save results to CSV
    with open('results.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Results saved to fashion_mnist_results.csv")


if __name__ == '__main__':
    app.run(main)
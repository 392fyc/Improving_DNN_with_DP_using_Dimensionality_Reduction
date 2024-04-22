from __future__ import absolute_import, division, print_function

import csv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentGaussianOptimizer
from tensorflow_privacy.privacy.analysis import rdp_accountant
from absl import app, flags, logging
import time
from sklearn.decomposition import PCA

# Define hyper-parameters
flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.5, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 600, 'Batch size')
flags.DEFINE_integer('epochs', 400, 'Number of epochs')
flags.DEFINE_integer('microbatches', 600, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_integer('pca_components', 60, 'Number of PCA components for dimensionality reduction')
flags.DEFINE_boolean('use_pca', True, 'Whether to use PCA for dimensionality reduction')
flags.DEFINE_float('delta', 1e-5, 'Delta for DP-SGD')
flags.DEFINE_float('epsilon_accountant', 8.0, 'Epsilon threshold to stop training for DP-SGD.')

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


class EpsilonCallback(tf.keras.callbacks.Callback):
    def __init__(self, epsilon_accountant, train_data_size):
        super(EpsilonCallback, self).__init__()
        self.epsilon_accountant = epsilon_accountant
        self.train_data_size = train_data_size
        self.global_epsilon = 0  # 添加一个属性来存储全局epsilon

    def on_epoch_end(self, epoch, logs=None):
        steps = (epoch + 1) * (self.train_data_size // FLAGS.batch_size)
        self.global_epsilon = compute_epsilon(steps)
        logging.info(f'Epoch {epoch + 1}: Epsilon = {self.global_epsilon:.2f}')
        if self.global_epsilon >= self.epsilon_accountant:
            logging.info(f'Epsilon {self.global_epsilon:.2f} has reached or exceeded the threshold {self.epsilon_accountant}. Stopping training.')
            self.model.stop_training = True


def load_data(use_pca):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    if use_pca:
        train_images_flattened = train_images.reshape(train_images.shape[0], -1)
        test_images_flattened = test_images.reshape(test_images.shape[0], -1)

        pca = PCA(n_components=FLAGS.pca_components)
        train_images_pca = pca.fit_transform(train_images_flattened)
        test_images_pca = pca.transform(test_images_flattened)

        return train_images_pca, train_labels, test_images_pca, test_labels
    else:
        train_images = np.expand_dims(train_images, axis=-1)
        test_images = np.expand_dims(test_images, axis=-1)
        return train_images, train_labels, test_images, test_labels


def create_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1000, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(10))
    return model

def save_results_to_csv(n_components, noise_multiplier, epochs, total_time, test_acc, epsilon, run_number=None):
    # 文件路径
    file_path = 'training_results.csv'
    # 检查文件是否已存在
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 如果文件不存在，写入列标题
        if not file_exists:
            writer.writerow(['Run Number', 'PCA Components', 'Noise Multiplier', 'Epochs', 'Total Time (s)',
                             'Test Accuracy', 'Epsilon'])
        # 追加数据，注意run_number可能为None，如果是，可以选择不写入或标记特殊值
        row = [run_number if run_number is not None else 'N/A', n_components, noise_multiplier, epochs, total_time,
               test_acc, epsilon]
        writer.writerow(row)


def main(argv):
    del argv  # Unused.

    for run in range(30):
        print(f"Starting run {run + 1}/30")
        start_time = time.time()

        train_data, train_labels, test_data, test_labels = load_data(FLAGS.use_pca)
        input_shape = (FLAGS.pca_components,) if FLAGS.use_pca else (28, 28, 1)
        model = create_model(input_shape)

        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.microbatches,
            learning_rate=FLAGS.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        epsilon_callback = EpsilonCallback(FLAGS.epsilon_accountant, train_data.shape[0])

        model.fit(train_data, train_labels, epochs=FLAGS.epochs, validation_data=(test_data, test_labels),
                  batch_size=FLAGS.batch_size, callbacks=[epsilon_callback])

        total_time = time.time() - start_time

        # 使用回调中更新的全局epsilon值
        final_epsilon = epsilon_callback.global_epsilon

        test_loss, test_acc = model.evaluate(test_data, test_labels)

        # 确定n_components的值
        n_components = FLAGS.pca_components if FLAGS.use_pca else 784

        # 调用save_results_to_csv时，使用final_epsilon作为epsilon的值
        save_results_to_csv(n_components, FLAGS.noise_multiplier, FLAGS.epochs, total_time, test_acc, final_epsilon, run + 1)

        logging.info(f"Run {run + 1}/30 finished")
        logging.info(f"Training finished in {total_time:.2f} seconds")
        logging.info(f"Test accuracy: {test_acc:.4f}")
        logging.info(f"Epsilon: {final_epsilon:.2f}")
        logging.info("Hyperparameters:")
        logging.info(f"Dimensions: {n_components}, DPSGD: {FLAGS.dpsgd}, Noise Multiplier: {FLAGS.noise_multiplier}, Epochs: {FLAGS.epochs}")


if __name__ == '__main__':
    app.run(main)
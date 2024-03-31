from __future__ import absolute_import, division, print_function

import csv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentGaussianOptimizer
from tensorflow_privacy.privacy.analysis import rdp_accountant
from absl import app, flags, logging
import time
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Define hyper-parameters
flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.5, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 400, 'Number of epochs')
flags.DEFINE_integer('microbatches', 250, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_integer('lda_components', 1, 'Number of LDA components for dimensionality reduction')
flags.DEFINE_boolean('use_lda', True, 'Whether to use LDA for dimensionality reduction')
flags.DEFINE_float('delta', 1e-5, 'Delta for DP-SGD')

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


def load_data_with_augmentation(use_lda):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    if use_lda:
        # Ensure n_components does not exceed n_classes - 1
        n_components = min(FLAGS.lda_components, len(np.unique(train_labels)) - 1)

        # Flatten the images for LDA
        train_images_flattened = train_images.reshape(train_images.shape[0], -1)
        test_images_flattened = test_images.reshape(test_images.shape[0], -1)

        # Apply LDA
        lda = LDA(n_components=n_components)
        train_images_lda = lda.fit_transform(train_images_flattened, train_labels)
        test_images_lda = lda.transform(test_images_flattened)

        return train_images_lda, train_labels, test_images_lda, test_labels
    else:
        train_images = np.expand_dims(train_images, axis=-1)
        test_images = np.expand_dims(test_images, axis=-1)
        return train_images, train_labels, test_images, test_labels


def create_model(use_lda, input_shape):
    model = Sequential()
    if not use_lda:  # 如果不使用LDA，需要一个Flatten层来处理28x28图像
        model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
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
            writer.writerow(['Run Number', 'Number of LDA Components', 'Noise Multiplier', 'Epochs', 'Total Time (s)',
                             'Test Accuracy', 'Epsilon'])
        # 追加数据，注意run_number可能为None，如果是，可以选择不写入或标记特殊值
        row = [run_number if run_number is not None else 'N/A', n_components, noise_multiplier, epochs, total_time,
               test_acc, epsilon]
        writer.writerow(row)


def main(argv):
    del argv  # Unused

    for run in range(30):  # 运行模型30次
        print(f"Starting run {run + 1}/30")
        start_time = time.time()
        train_data, train_labels, test_data, test_labels = load_data_with_augmentation(FLAGS.use_lda)

        input_shape = (FLAGS.lda_components,) if FLAGS.use_lda else (28, 28, 1)
        model = create_model(FLAGS.use_lda, input_shape)

        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.microbatches,
            learning_rate=FLAGS.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=FLAGS.epochs, validation_data=(test_data, test_labels),
                  batch_size=FLAGS.batch_size)

        total_time = time.time() - start_time
        epsilon = compute_epsilon((train_data.shape[0] // FLAGS.batch_size) * FLAGS.epochs)
        test_loss, test_acc = model.evaluate(test_data, test_labels)

        # 确定n_components的值
        n_components = FLAGS.lda_components if FLAGS.use_lda else 784

        # 调用修改后的save_results_to_csv函数
        save_results_to_csv(n_components, FLAGS.noise_multiplier, FLAGS.epochs, total_time, test_acc, epsilon, run + 1)

        # 计算并记录当前使用的维度
        if FLAGS.use_lda:
            used_dimensions = FLAGS.lda_components
        else:
            # 假设原始数据是28x28的图像，所以原始维度是784
            used_dimensions = 784  # 28 * 28

        logging.info(f"Run {run + 1}/30 finished")
        logging.info(f"Training finished in {total_time:.2f} seconds")
        logging.info(f"Test accuracy: {test_acc:.4f}")
        logging.info(f"Epsilon: {epsilon:.2f}")
        logging.info("Hyperparameters:")
        logging.info(f"Used dimensions: {used_dimensions}, DPSGD: {FLAGS.dpsgd}, Noise Multiplier: {FLAGS.noise_multiplier}, Epochs: {FLAGS.epochs}")

if __name__ == '__main__':
    app.run(main)

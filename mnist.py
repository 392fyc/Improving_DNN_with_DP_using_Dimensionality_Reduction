from __future__ import absolute_import, division, print_function

import csv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentGaussianOptimizer
from tensorflow_privacy.privacy.analysis import rdp_accountant
from absl import app, flags, logging
import time
from sklearn.decomposition import PCA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define hyper-parameters using the flags module
flags.DEFINE_boolean('dpsgd', True, 'Train with DP-SGD if True, otherwise with vanilla SGD')
flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 8, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 1000, 'Batch size')
flags.DEFINE_integer('epochs', 400, 'Number of epochs')
flags.DEFINE_integer('microbatches', 1000, 'Number of microbatches, must evenly divide batch_size')
flags.DEFINE_boolean('use_pca', True, 'Use PCA for dimensionality reduction')
flags.DEFINE_float('delta', 1e-5, 'Delta for DP-SGD')
flags.DEFINE_float('epsilon_accountant', 8.0, 'Epsilon threshold to stop training for DP-SGD.')
flags.DEFINE_integer('runs_per_dim', 1, 'Number of runs for each dimension')

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


def load_data(n_components):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    if FLAGS.use_pca:
        pca = PCA(n_components=n_components)
        train_images = pca.fit_transform(train_images.reshape(-1, 28*28))
        test_images = pca.transform(test_images.reshape(-1, 28*28))
    else:
        train_images = train_images.reshape(-1, 28*28)
        test_images = test_images.reshape(-1, 28*28)

    return train_images, train_labels, test_images, test_labels

def create_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(1000, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(10)
    ])
    return model

def calculate_score(run_time, test_acc, max_run_time, max_acc=1.0, time_weight=1.0, accuracy_weight=1.0):
    """
    Calculate a weighted score based on normalized run time and accuracy.

    Args:
        run_time (float): Actual run time of the model.
        test_acc (float): Accuracy of the model on test data.
        max_run_time (float): Maximum acceptable run time.
        max_acc (float): Maximum possible accuracy, usually 1.0 for classification tasks.
        time_weight (float): Weight for the time score.
        accuracy_weight (float): Weight for the accuracy score.

    Returns:
        float: Weighted score considering both time and accuracy.
    """
    # Normalize run time and accuracy
    normalized_time = max(0, (max_run_time - run_time) / max_run_time)  # 更高的运行时间得分较低
    normalized_accuracy = test_acc / max_acc

    # Calculate weighted score
    score = (time_weight * normalized_time) + (accuracy_weight * normalized_accuracy)
    return score / (time_weight + accuracy_weight)  # Normalize the final score by total weight

# Example usage
max_run_time = 200.0  # seconds
score = calculate_score(run_time=100, test_acc=0.92, max_run_time=max_run_time, time_weight=1, accuracy_weight=1)
print("Score:", score)


class EpsilonCallback(tf.keras.callbacks.Callback):
    def __init__(self, epsilon_accountant, train_data_size):
        super(EpsilonCallback, self).__init__()
        self.epsilon_accountant = epsilon_accountant
        self.train_data_size = train_data_size
        self.global_epsilon = 0

    def on_epoch_end(self, epoch, logs=None):
        steps = (epoch + 1) * (self.train_data_size // FLAGS.batch_size)
        self.global_epsilon = compute_epsilon(steps)
        if self.global_epsilon >= self.epsilon_accountant:
            self.model.stop_training = True


def evaluate(n_components, max_run_time):
    start_time = time.time()
    train_data, train_labels, test_data, test_labels = load_data(n_components)
    input_shape = (n_components,)
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

    logging.info(f"Evaluation for {n_components} dimensions finished")
    logging.info(f"Training finished in {total_time:.2f} seconds")
    logging.info(f"Test accuracy: {test_acc:.4f}")
    logging.info(f"Epsilon: {final_epsilon:.2f}")
    logging.info("Hyperparameters:")
    logging.info(f"Dimensions: {n_components}, DPSGD: {FLAGS.dpsgd}, Noise Multiplier: {FLAGS.noise_multiplier}, Epochs: {FLAGS.epochs}")

    score = calculate_score(total_time, test_acc, max_run_time)
    logging.info(f"Score: {score:.4f}")

    info = {
        'epochs': FLAGS.epochs,
        'total_time': total_time,
        'test_acc': test_acc,
        'final_epsilon': final_epsilon
    }

    return score, info

def find_optimal_dimensions(max_run_time, original_dims):
    left, right = 1, original_dims  # 初始搜索范围为 [1, original_dims]
    step = (right - left) // 2  # 初始步长为搜索范围的一半

    evaluated_dims = []  # 记录评估过的维度

    # 先训练中间维度
    mid = min(max(left, (left + right) // 2), right)
    mid_score, mid_info = evaluate(mid, max_run_time)
    evaluated_dims.append((mid, mid_score, mid_info))

    while step > 1:
        # 训练较小维度
        lower_dim = min(max(left, mid - step), right)
        lower_score, lower_info = evaluate(lower_dim, max_run_time)
        evaluated_dims.append((lower_dim, lower_score, lower_info))

        # 训练较大维度
        higher_dim = min(max(left, mid + step), right)
        higher_score, higher_info = evaluate(higher_dim, max_run_time)
        evaluated_dims.append((higher_dim, higher_score, higher_info))

        # 根据分数决定下一步搜索范围
        if lower_score > mid_score and lower_score >= higher_score:
            right = mid
            mid = lower_dim
            mid_score = lower_score
        elif higher_score > mid_score and higher_score >= lower_score:
            left = mid
            mid = higher_dim
            mid_score = higher_score
        else:
            # 如果 lower 和 higher 都差于 mid,减半步长并继续搜索
            step //= 2
            continue

        step //= 2  # 步长减半

    return mid, evaluated_dims


def save_results_to_csv(n_components, noise_multiplier, epochs, total_time, test_acc, epsilon, score):
    # 文件路径
    file_path = 'training_results.csv'
    # 检查文件是否已存在
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 如果文件不存在,写入列标题
        if not file_exists:
            writer.writerow(['PCA Components', 'Noise Multiplier', 'Epochs', 'Total Time (s)',
                             'Test Accuracy', 'Epsilon', 'Score'])
        # 追加数据
        row = [n_components, noise_multiplier, epochs, total_time, test_acc, epsilon, score]
        writer.writerow(row)


def main(argv):
    del argv  # Unused.

    # 先用原始数据维度进行测试
    original_dims = 784  # MNIST 数据集的原始维度
    initial_max_run_time = 1800  # 初始的最大运行时间设置为 30 分钟
    _, original_info = evaluate(original_dims, initial_max_run_time)
    max_run_time = original_info['total_time']
    print(f"原始数据维度 ({original_dims}) 的运行时间: {max_run_time:.2f} 秒")

    # 寻找最优的PCA维度
    optimal_dims, evaluated_dims = find_optimal_dimensions(max_run_time, original_dims)
    print(f"最优的PCA维度为: {optimal_dims}")

    # 输出汇总信息
    logging.info("Evaluated dimensions:")
    for dim, score, info in evaluated_dims:
        logging.info(f"Dimensions: {dim}, Score: {score:.4f}")
        logging.info(
            f"  Epochs: {info['epochs']}, Noise Multiplier: {FLAGS.noise_multiplier}, Total Time: {info['total_time']:.2f}s, Test Accuracy: {info['test_acc']:.4f}, Epsilon: {info['final_epsilon']:.2f}")

    # 将结果写入 CSV 文件
    for dim, score, info in evaluated_dims:
        save_results_to_csv(dim, FLAGS.noise_multiplier, info['epochs'], info['total_time'], info['test_acc'],
                            info['final_epsilon'], score)

    for run in range(FLAGS.runs_per_dim):
        print(f"Starting run {run + 1}/{FLAGS.runs_per_dim} with {optimal_dims} dimensions")
        start_time = time.time()

        train_data, train_labels, test_data, test_labels = load_data(optimal_dims)
        input_shape = (optimal_dims,)
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

        logging.info(f"Optimizing finished")
        logging.info(f"Training finished in {total_time:.2f} seconds")
        logging.info(f"Test accuracy: {test_acc:.4f}")
        logging.info(f"Epsilon: {final_epsilon:.2f}")
        logging.info("Hyperparameters:")
        logging.info(
            f"Dimensions: {optimal_dims}, DPSGD: {FLAGS.dpsgd}, Noise Multiplier: {FLAGS.noise_multiplier}, Epochs: {FLAGS.epochs}")

if __name__ == '__main__':
    app.run(main)
from __future__ import absolute_import, division, print_function

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentLaplaceOptimizer
from absl import app, flags, logging
import time
from sklearn.decomposition import PCA
import os
from dp_utils import evaluate_epsilon_laplace, EpsilonCallback


logging.set_verbosity(logging.ERROR)  # This will show only ERROR messages and higher


flags.DEFINE_boolean('dpsgd', True, 'Train with DP-SGD if True, otherwise with vanilla SGD')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 6.4, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l1_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 1000, 'Batch size')
flags.DEFINE_integer('epochs', 400, 'Number of epochs')
flags.DEFINE_integer('microbatches', 1000, 'Number of microbatches, must evenly divide batch_size')
flags.DEFINE_boolean('use_pca', True, 'Use PCA for dimensionality reduction')
flags.DEFINE_float('delta', 1e-5, 'Delta for DP-SGD')
flags.DEFINE_float('epsilon_accountant', 2.0, 'Epsilon threshold to stop training for DP-SGD.')
flags.DEFINE_integer('runs_per_dim', 1, 'Number of runs for each dimension')

FLAGS = flags.FLAGS


def tune_noise_multiplier(target_epsilon, tolerance=1e-3):
    lower = 1.0
    upper = 20.0
    while upper - lower > tolerance:
        noise_multiplier = (lower + upper) / 2  # 不进行四舍五入
        FLAGS.noise_multiplier = noise_multiplier
        (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
        epsilon = evaluate_epsilon_laplace(1, noise_multiplier, FLAGS.batch_size, FLAGS.epochs, len(train_images))
        print(f"Trying noise_multiplier={noise_multiplier:.3f}, got epsilon={epsilon:.3f}")
        if abs(epsilon - target_epsilon) <= tolerance:
            optimal_noise_multiplier = noise_multiplier
            FLAGS.noise_multiplier = optimal_noise_multiplier
            print(f"Optimal noise_multiplier={optimal_noise_multiplier:.3f}, epsilon={epsilon:.3f}")
            return optimal_noise_multiplier
        elif epsilon < target_epsilon:
            upper = noise_multiplier
        else:
            lower = noise_multiplier
    optimal_noise_multiplier = noise_multiplier
    FLAGS.noise_multiplier = optimal_noise_multiplier
    print(f"Optimal noise_multiplier={optimal_noise_multiplier:.3f}, epsilon={epsilon:.3f}")
    return optimal_noise_multiplier


def load_mnist_data(n_components):
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
        Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_shape,)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    return model


def epsilon_vs_noise_multiplier(noise_multipliers, original_dims, max_run_time):
    epsilons = []
    for noise_multiplier in noise_multipliers:
        _, infos, _ = evaluate(original_dims, max_run_time, num_runs=1, noise_multiplier=noise_multiplier)
        epsilons.append(infos[0]['final_epsilon'])
    return epsilons


def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def evaluate(n_components, noise_multiplier, max_run_time, num_runs=None):
    if num_runs is None:
        num_runs = FLAGS.runs_per_dim

    scores = []
    infos = []

    for run in range(num_runs):
        print(f"Training with {n_components} dimensions, run {run + 1}/{num_runs}")
        start_time = time.time()
        train_data, train_labels, test_data, test_labels = load_mnist_data(n_components)
        input_shape = (n_components,)
        model = create_model(input_shape)

        optimizer = DPGradientDescentLaplaceOptimizer(
            l1_norm_clip=FLAGS.l1_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=FLAGS.microbatches,
            learning_rate=FLAGS.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        epsilon_callback = EpsilonCallback(FLAGS.epsilon_accountant, train_data.shape[0])

        model.fit(train_data, train_labels, epochs=FLAGS.epochs, validation_data=(test_data, test_labels),
                  batch_size=FLAGS.batch_size, callbacks=[epsilon_callback])

        total_time = time.time() - start_time

        final_epsilon = epsilon_callback.global_epsilon
        print(f"Epsilon for noise_multiplier={noise_multiplier}: {final_epsilon:.3f}")

        test_loss, test_acc = model.evaluate(test_data, test_labels)

        score = calculate_score(total_time, test_acc, max_run_time)
        scores.append(score)
        infos.append({
            'epochs': FLAGS.epochs,
            'total_time': total_time,
            'test_acc': test_acc,
            'final_epsilon': final_epsilon
        })

    mean_score = np.mean(scores)
    mean_time = np.mean([info['total_time'] for info in infos])
    mean_acc = np.mean([info['test_acc'] for info in infos])
    mean_epsilon = np.mean([info['final_epsilon'] for info in infos])
    dispersion = np.std(scores)
    outliers = sum(abs(score - mean_score) > 2 * dispersion for score in scores)

    summary = {
        'mean_score': mean_score,
        'mean_time': mean_time,
        'mean_acc': mean_acc,
        'mean_epsilon': mean_epsilon,
        'dispersion': dispersion,
        'outliers': outliers
    }

    return mean_score, infos, summary


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


def find_optimal_dimensions(max_run_time, original_dims, noise_multiplier):
    evaluated_dims = []

    # 对原始维度进行训练,获取时间阈值
    original_score, original_infos, original_summary = evaluate(original_dims, noise_multiplier, max_run_time)
    evaluated_dims.append((original_dims, original_score, original_infos, original_summary))
    time_threshold = original_summary['mean_time']

    def search_recursive(center, step):
        if step == 1:
            left_dim, center_dim, right_dim = center - 1, center, center + 1

            left_score, left_infos, left_summary = evaluate(left_dim, noise_multiplier, max_run_time)
            evaluated_dims.append((left_dim, left_score, left_infos, left_summary))

            center_score, center_infos, center_summary = evaluate(center_dim, noise_multiplier, max_run_time)
            evaluated_dims.append((center_dim, center_score, center_infos, center_summary))

            right_score, right_infos, right_summary = evaluate(right_dim, noise_multiplier, max_run_time)
            evaluated_dims.append((right_dim, right_score, right_infos, right_summary))

            if center_score >= left_score and center_score >= right_score:
                return center_dim
            elif left_score >= center_score and left_score >= right_score:
                return left_dim
            else:
                return right_dim
        else:
            left_dim, right_dim = max(1, center - step), min(original_dims, center + step)

            left_score, left_infos, left_summary = evaluate(left_dim, noise_multiplier, max_run_time)
            evaluated_dims.append((left_dim, left_score, left_infos, left_summary))

            center_score, center_infos, center_summary = evaluate(center, noise_multiplier, max_run_time)
            evaluated_dims.append((center, center_score, center_infos, center_summary))

            right_score, right_infos, right_summary = evaluate(right_dim, noise_multiplier, max_run_time)
            evaluated_dims.append((right_dim, right_score, right_infos, right_summary))

            if center_score >= left_score and center_score >= right_score:
                return search_recursive(center, step // 2)
            elif left_score >= center_score and left_score >= right_score:
                return search_recursive(left_dim, step // 2)
            else:
                return search_recursive(right_dim, step // 2)

    def calculate_score(summary, time_threshold):
        run_time = summary['mean_time']
        test_acc = summary['mean_acc']
        normalized_time = max(0, (time_threshold - run_time) / time_threshold)
        normalized_accuracy = test_acc
        score = (normalized_time + normalized_accuracy) / 2
        return score

    initial_center = original_dims // 2
    initial_step = original_dims // 4
    optimal_dim = search_recursive(initial_center, initial_step)

    # 重新计算所有评估维度的分数,使用时间阈值进行归一化
    for i in range(len(evaluated_dims)):
        dim, _, _, summary = evaluated_dims[i]
        score = calculate_score(summary, time_threshold)
        evaluated_dims[i] = (dim, score, _, summary)

    return optimal_dim, evaluated_dims


def save_results_to_csv(evaluated_dims, noise_multiplier):
    # 文件路径
    file_path = 'training_results.csv'
    # 检查文件是否已存在
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 如果文件不存在,写入列标题
        if not file_exists:
            writer.writerow(['PCA Components', 'Run', 'Epochs', 'Total Time (s)', 'Test Accuracy', 'Epsilon', 'Noise Multiplier'])

        for dim, _, infos, _ in evaluated_dims:
            for i, info in enumerate(infos):
                row = [dim, i+1, info['epochs'], info['total_time'], info['test_acc'], info['final_epsilon'], noise_multiplier]
                writer.writerow(row)


def main(argv):
    del argv

    target_epsilon = FLAGS.epsilon_accountant

    print("Epsilon optimization:")
    noise_multiplier = tune_noise_multiplier(target_epsilon)

    print(f"\nDimension optimization:")
    print(f"Using noise multiplier: {noise_multiplier}")

    original_dims = 784
    _, original_info, original_summary = evaluate(original_dims, noise_multiplier, num_runs=FLAGS.runs_per_dim, max_run_time=float('inf'))
    max_run_time = original_summary['mean_time']  # 使用原始维度的平均运行时间作为max_run_time
    print(f"Mean running time for original dimensions ({original_dims}): {max_run_time:.2f} seconds")

    optimal_dim, evaluated_dims = find_optimal_dimensions(max_run_time, original_dims, noise_multiplier)
    print(f"\nOptimal PCA dimensions: {optimal_dim}")

    print("\nEvaluated dimensions:")
    for dim, score, _, summary in sorted(evaluated_dims, key=lambda x: -x[1]):
        print(f"Dimensions: {dim}, Score: {score:.4f}")
        print(f"  Mean Time: {summary['mean_time']:.2f}s, Mean Accuracy: {summary['mean_acc']:.4f}")
        print(f"  Dispersion: {summary['dispersion']:.4f}, Outliers: {summary['outliers']}")

    print(f"\nOptimal dimension {optimal_dim}:")
    for dim, score, _, summary in evaluated_dims:
        if dim == optimal_dim:
            print(
                f"  Score: {score:.4f}, Mean Time: {summary['mean_time']:.2f}s, Mean Accuracy: {summary['mean_acc']:.4f}")
            break

    save_results_to_csv(evaluated_dims, noise_multiplier)

    print("\nHyperparameters:")
    print(f"  Noise Multiplier: {noise_multiplier}, Epochs: {FLAGS.epochs}")

if __name__ == '__main__':
    app.run(main)

from __future__ import absolute_import, division, print_function

import csv
import pandas as pd
import time
import numpy as np
import math
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentLaplaceOptimizer
from absl import app, flags, logging
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.set_verbosity(logging.ERROR)

flags.DEFINE_boolean('dpsgd', True, 'Train with DP-SGD if True, otherwise with vanilla SGD')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 6.4, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l1_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('epochs', 50, 'Number of epochs')
flags.DEFINE_integer('microbatches', 1, 'Number of microbatches, must evenly divide batch_size')
flags.DEFINE_float('delta', 1e-5, 'Delta for DP-SGD')
flags.DEFINE_float('epsilon_accountant', 8.0, 'Epsilon threshold to stop training for DP-SGD.')

FLAGS = flags.FLAGS


def compute_epsilon_laplace(l1_norm_clip, noise_multiplier, num_steps, delta):
    """
    Compute epsilon for Laplace mechanism.

    :param l1_norm_clip: L1 norm clipping threshold
    :param noise_multiplier: Noise multiplier
    :param num_steps: Number of steps (epochs * steps_per_epoch)
    :param delta: Target delta
    :return: Computed epsilon
    """
    sensitivity = 2 * l1_norm_clip
    epsilon_single = sensitivity / (l1_norm_clip * noise_multiplier)
    epsilon_total = epsilon_single * math.sqrt(2 * num_steps * math.log(1 / delta))
    return epsilon_total


def evaluate_epsilon_laplace(noise_multiplier, batch_size, epochs, train_data_size):
    print(f"Evaluating epsilon for noise_multiplier={noise_multiplier}, batch_size={batch_size}, epochs={epochs}")
    steps = epochs * (train_data_size // batch_size)
    epsilon = compute_epsilon_laplace(FLAGS.l1_norm_clip, noise_multiplier, steps, FLAGS.delta)
    print(f"Computed epsilon: {epsilon}")
    return epsilon


def tune_noise_multiplier_laplace(target_epsilon, tolerance=1e-3):
    print("Starting noise multiplier tuning...")
    lower = 0.1
    upper = 20.0
    iterations = 0
    max_iterations = 50
    while upper - lower > tolerance and iterations < max_iterations:
        iterations += 1
        noise_multiplier = (lower + upper) / 2
        FLAGS.noise_multiplier = noise_multiplier
        print(f"Iteration {iterations}: Testing noise_multiplier = {noise_multiplier}")
        X_train, _, _, _ = load_diabetes_data()
        if X_train is None:
            print("Error loading data. Exiting noise multiplier tuning.")
            return None
        epsilon = evaluate_epsilon_laplace(noise_multiplier, FLAGS.batch_size, FLAGS.epochs, len(X_train))
        print(f"Iteration {iterations}: Trying noise_multiplier={noise_multiplier:.3f}, got epsilon={epsilon:.3f}")
        if abs(epsilon - target_epsilon) <= tolerance:
            optimal_noise_multiplier = noise_multiplier
            FLAGS.noise_multiplier = optimal_noise_multiplier
            print(f"Optimal noise_multiplier={optimal_noise_multiplier:.3f}, epsilon={epsilon:.3f}")
            return optimal_noise_multiplier
        elif epsilon > target_epsilon:
            lower = noise_multiplier
        else:
            upper = noise_multiplier

    print(f"Reached maximum iterations or tolerance. Using final noise_multiplier: {noise_multiplier}")
    return noise_multiplier


def load_diabetes_data():
    print("Loading Diabetes data...")
    try:
        data = pd.read_csv('data/diabetic_data.csv')
    except FileNotFoundError:
        print("Error: The file 'data/diabetic_data.csv' was not found.")
        return None, None, None, None

    # 预处理数据
    columns_to_drop = ['encounter_id', 'patient_nbr']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    # 检查 'readmitted' 列的类型和唯一值
    print(f"'readmitted' column type: {data['readmitted'].dtype}")
    unique_values = data['readmitted'].unique()
    print(f"Unique values in 'readmitted' column: {unique_values}")

    # 如果 'readmitted' 已经是数值类型，我们直接使用它
    if np.issubdtype(data['readmitted'].dtype, np.number):
        y = data['readmitted'].astype(int)
    else:
        # 如果是字符串类型，我们进行映射
        readmitted_map = {'NO': 0, '<30': 1, '>30': 0}
        y = data['readmitted'].map(readmitted_map).fillna(-1).astype(int)

    X = data.drop('readmitted', axis=1)

    # 将分类变量转换为数值
    cat_columns = X.select_dtypes(include=['object']).columns
    for col in cat_columns:
        X[col] = pd.Categorical(X[col]).codes

    # 处理特征中的缺失值
    X = X.fillna(X.mean())

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 确保所有特征值非负
    X_train_scaled = np.clip(X_train_scaled, 0, None)
    X_test_scaled = np.clip(X_test_scaled, 0, None)

    print(f"Data loaded. Shape of training data: {X_train_scaled.shape}")
    print(f"Positive samples in training set: {sum(y_train == 1)}/{len(y_train)} ({sum(y_train == 1) / len(y_train) * 100:.2f}%)")
    return X_train_scaled, y_train, X_test_scaled, y_test


def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_shape,)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    return model


class JTree:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.tree = defaultdict(list)
        self.conditionals = {}
        self.feature_sizes = {}

    def mutual_information(self, x, y):
        joint_hist = np.histogram2d(x, y, bins=20)[0]
        p_xy = joint_hist / np.sum(joint_hist)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        p_xy_flat = p_xy.flatten()
        p_ind = p_x[:, np.newaxis] * p_y[np.newaxis, :]
        p_ind_flat = p_ind.flatten()

        # Avoid division by zero and log(0)
        mask = (p_xy_flat > 0) & (p_ind_flat > 0)
        return np.sum(p_xy_flat[mask] * np.log(p_xy_flat[mask] / p_ind_flat[mask]))

    def exponential_mechanism(self, scores, epsilon):
        scores = np.array(scores)  # Convert to NumPy array
        if len(scores) == 0:
            return 0
        max_score = np.max(scores)
        scores = scores - max_score  # For numerical stability
        probabilities = np.exp(epsilon * scores / (2 * len(scores)))
        probabilities /= np.sum(probabilities)
        return np.random.choice(len(scores), p=probabilities)

    def fit(self, data):
        n_features = data.shape[1]
        remaining_attributes = set(range(n_features))

        # 计算每个特征的唯一值数量
        for i in range(n_features):
            self.feature_sizes[i] = int(np.max(data[:, i])) + 1

        # 选择第一个属性
        first_attribute = np.random.choice(list(remaining_attributes))
        remaining_attributes.remove(first_attribute)
        self.tree[first_attribute] = []

        for _ in range(1, n_features):
            scores = []
            candidates = []

            for attr in remaining_attributes:
                for parent in self.tree:
                    score = self.mutual_information(data[:, attr], data[:, parent])
                    scores.append(score)
                    candidates.append((attr, parent))

            if scores:
                chosen_idx = self.exponential_mechanism(scores, self.epsilon / (2 * n_features))
                chosen_attr, chosen_parent = candidates[chosen_idx]
            else:
                chosen_attr = np.random.choice(list(remaining_attributes))
                chosen_parent = first_attribute

            self.tree[chosen_parent].append(chosen_attr)
            remaining_attributes.remove(chosen_attr)

            # 计算条件概率
            parent_data = data[:, chosen_parent]
            attr_data = data[:, chosen_attr]

            unique_parent_values = np.unique(parent_data)
            self.conditionals[chosen_attr] = {}
            for parent_value in unique_parent_values:
                mask = parent_data == parent_value
                conditional_counts = np.bincount(attr_data[mask].astype(int), minlength=self.feature_sizes[chosen_attr])
                probs = conditional_counts / np.sum(conditional_counts)
                noisy_probs = probs + np.random.laplace(0, 1 / self.epsilon, self.feature_sizes[chosen_attr])
                noisy_probs = np.clip(noisy_probs, 0, 1)
                self.conditionals[chosen_attr][parent_value] = noisy_probs / np.sum(noisy_probs)

    def sample(self, n_samples):
        samples = np.zeros((n_samples, len(self.feature_sizes)), dtype=int)
        root = next(iter(self.tree))  # 获取根节点
        samples[:, root] = np.random.choice(self.feature_sizes[root], size=n_samples)

        for parent in self.tree:
            for child in self.tree[parent]:
                parent_values = samples[:, parent]
                child_probs = np.array([self.conditionals[child].get(pv, np.ones(self.feature_sizes[child])/self.feature_sizes[child]) for pv in parent_values])
                samples[:, child] = np.array([np.random.choice(self.feature_sizes[child], p=p) for p in child_probs])

        return samples


def evaluate_jtree(noise_multiplier):
    print("Loading Diabetes data...")
    X_train, y_train, X_test, y_test = load_diabetes_data()
    if X_train is None or y_train is None or X_test is None or y_test is None:
        print("Error: Failed to load data.")
        return None, None, None

    # 检查标签的唯一值
    unique_labels = np.unique(y_train)
    print(f"Unique labels in training set: {unique_labels}")

    print("Constructing JTree model...")
    start_time = time.time()
    jtree = JTree(epsilon=FLAGS.epsilon_accountant)
    jtree.fit(X_train)
    jtree_time = time.time() - start_time
    print(f"JTree construction time: {jtree_time:.2f} seconds")

    print("Sampling from JTree model...")
    start_time = time.time()
    synthetic_X_train = jtree.sample(len(X_train))
    sampling_time = time.time() - start_time
    print(f"Sampling time: {sampling_time:.2f} seconds")

    # 检查生成的样本
    print(f"Synthetic data shape: {synthetic_X_train.shape}")
    print(f"Synthetic data min: {np.min(synthetic_X_train)}, max: {np.max(synthetic_X_train)}")
    print(f"Synthetic data contains NaN: {np.isnan(synthetic_X_train).any()}")
    print(f"Synthetic data contains Inf: {np.isinf(synthetic_X_train).any()}")

    model = create_model(X_train.shape[1])

    optimizer = DPGradientDescentLaplaceOptimizer(
        l1_norm_clip=FLAGS.l1_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate
    )

    # 如果是二分类问题，使用 'binary_crossentropy'；如果是多分类问题，使用 'sparse_categorical_crossentropy'
    loss = 'binary_crossentropy' if len(unique_labels) == 2 else 'sparse_categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    print("Training model...")
    start_time = time.time()
    history = model.fit(synthetic_X_train, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
                        validation_split=0.2, verbose=1)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")

    print("Evaluating model...")
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

    eps = compute_epsilon_laplace(FLAGS.l1_norm_clip, noise_multiplier, FLAGS.epochs * (len(X_train) // FLAGS.batch_size), FLAGS.delta)

    info = [{
        'epochs': FLAGS.epochs,
        'jtree_time': jtree_time,
        'sampling_time': sampling_time,
        'train_time': train_time,
        'total_time': jtree_time + sampling_time + train_time,
        'test_acc': test_acc,
        'final_epsilon': eps
    }]

    return test_acc, info, {'mean_time': jtree_time + sampling_time + train_time, 'mean_acc': test_acc}


def save_results_to_csv(evaluated_results, noise_multiplier):
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'Epochs', 'Total Time', 'Test Accuracy', 'Epsilon', 'Noise Multiplier'])

        for idx, (_, infos, _) in enumerate(evaluated_results):
            for i, info in enumerate(infos):
                row = [idx, info['epochs'], info['total_time'], info['test_acc'], info['final_epsilon'],
                       noise_multiplier]
                writer.writerow(row)


def main(argv):
    del argv

    target_epsilon = FLAGS.epsilon_accountant

    print("Epsilon optimization:")
    start_time = time.time()
    noise_multiplier = tune_noise_multiplier_laplace(target_epsilon)
    if noise_multiplier is None:
        print("Failed to tune noise multiplier. Exiting.")
        return
    end_time = time.time()
    print(f"Noise multiplier tuning took {end_time - start_time:.2f} seconds")

    print(f"Using noise multiplier: {noise_multiplier}")

    evaluated_results = []
    for i in range(1):  # 训练模型n次
        test_acc, info, summary = evaluate_jtree(noise_multiplier)
        if test_acc is None:
            print("Failed to evaluate JTree. Skipping this iteration.")
            continue
        evaluated_results.append((0, info, summary))
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Training time: {summary['mean_time']:.2f} seconds")
        print(f"Epsilon: {info[0]['final_epsilon']:.4f}")

    if evaluated_results:
        save_results_to_csv(evaluated_results, noise_multiplier)

    print("\nHyperparameters:")
    print(f"  Noise Multiplier: {noise_multiplier}, Epochs: {FLAGS.epochs}")


if __name__ == '__main__':
    app.run(main)
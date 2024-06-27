from __future__ import absolute_import, division, print_function

import csv
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentLaplaceOptimizer
from tensorflow_privacy.privacy.analysis import rdp_accountant
from absl import app, flags, logging
import time
import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
logging.set_verbosity(logging.ERROR)  # This will show only ERROR messages and higher

flags.DEFINE_boolean('dpsgd', True, 'Train with DP-SGD if True, otherwise with vanilla SGD')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 6.4, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l1_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 1000, 'Batch size')
flags.DEFINE_integer('epochs', 400, 'Number of epochs')
flags.DEFINE_integer('microbatches', 1000, 'Number of microbatches, must evenly divide batch_size')
flags.DEFINE_float('delta', 1e-5, 'Delta for DP-SGD')
flags.DEFINE_float('epsilon_accountant', 2.0, 'Epsilon threshold to stop training for DP-SGD.')

FLAGS = flags.FLAGS


def compute_epsilon(steps):
    if FLAGS.noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = FLAGS.batch_size / 60000
    rdp = rdp_accountant.compute_rdp(q=sampling_probability,
                                     noise_multiplier=FLAGS.noise_multiplier,
                                     steps=steps,
                                     orders=orders)
    return rdp_accountant.get_privacy_spent(orders, rdp, target_delta=FLAGS.delta)[0]


def evaluate_epsilon(noise_multiplier, batch_size, epochs, train_data_size):
    print(f"Evaluating epsilon for noise_multiplier={noise_multiplier}, batch_size={batch_size}, epochs={epochs}")
    steps = epochs * (train_data_size // batch_size)
    sampling_probability = batch_size / train_data_size
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    rdp = rdp_accountant.compute_rdp(q=sampling_probability,
                                     noise_multiplier=noise_multiplier,
                                     steps=steps,
                                     orders=orders)
    epsilon = rdp_accountant.get_privacy_spent(orders, rdp, target_delta=FLAGS.delta)[0]
    print(f"Computed epsilon: {epsilon}")
    return epsilon


def tune_noise_multiplier(target_epsilon, tolerance=1e-3):
    print("Starting noise multiplier tuning...")
    lower = 1.0
    upper = 20.0
    iterations = 0
    max_iterations = 50
    while upper - lower > tolerance and iterations < max_iterations:
        iterations += 1
        noise_multiplier = (lower + upper) / 2
        FLAGS.noise_multiplier = noise_multiplier
        print(f"Iteration {iterations}: Testing noise_multiplier = {noise_multiplier}")
        X_train, _, _, _ = load_mnist_data()  # 使用 Adult 数据集
        epsilon = evaluate_epsilon(noise_multiplier, FLAGS.batch_size, FLAGS.epochs, len(X_train))
        print(f"Iteration {iterations}: Trying noise_multiplier={noise_multiplier:.3f}, got epsilon={epsilon:.3f}")
        if abs(epsilon - target_epsilon) <= tolerance:
            optimal_noise_multiplier = noise_multiplier
            FLAGS.noise_multiplier = optimal_noise_multiplier
            print(f"Optimal noise_multiplier={optimal_noise_multiplier:.3f}, epsilon={epsilon:.3f}")
            return optimal_noise_multiplier
        elif epsilon < target_epsilon:
            upper = noise_multiplier
        else:
            lower = noise_multiplier

    print(f"Reached maximum iterations or tolerance. Using final noise_multiplier: {noise_multiplier}")
    return noise_multiplier


def load_mnist_data():
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def create_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(1000, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(10)
    ])
    return model


class JTree:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.tree = defaultdict(list)
        self.conditionals = {}

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

        # 选择第一个属性
        first_attribute = np.random.choice(list(remaining_attributes))
        remaining_attributes.remove(first_attribute)
        self.tree[first_attribute] = []  # 确保第一个属性被添加到树中

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

            # 将数据缩放到一个正数范围以避免负值
            min_value = np.min(attr_data)
            if min_value < 0:
                attr_data = attr_data - min_value

            unique_parent_values, counts = np.unique(parent_data, return_counts=True)
            self.conditionals[chosen_attr] = {}
            for parent_value, count in zip(unique_parent_values, counts):
                mask = parent_data == parent_value
                conditional_counts = np.bincount(attr_data[mask].astype(int), minlength=256)
                probs = conditional_counts / count
                noisy_probs = probs + np.random.laplace(0, 1 / self.epsilon, 256)
                noisy_probs = np.clip(noisy_probs, 0, 1)
                self.conditionals[chosen_attr][parent_value] = noisy_probs / np.sum(noisy_probs)

    def sample(self, n_samples):
        n_features = max(max(self.tree.keys()), max(max(v) for v in self.tree.values())) + 1
        samples = np.zeros((n_samples, n_features), dtype=int)
        root = next(iter(self.tree))  # 获取根节点
        root_feature_size = len(np.unique(samples[:, root]))  # 确定根节点特征数
        samples[:, root] = np.random.choice(root_feature_size, size=n_samples)

        for parent in self.tree:
            for child in self.tree[parent]:
                parent_values = samples[:, parent]
                child_probs = np.array([self.conditionals[child].get(pv, np.ones(256)/256) for pv in parent_values])
                samples[:, child] = np.array([np.random.choice(256, p=p) for p in child_probs])

        return samples


def evaluate_jtree(noise_multiplier):
    print("Loading MNIST data...")
    x_train, y_train, x_test, y_test = load_mnist_data()

    print("Constructing JTree model...")
    start_time = time.time()
    jtree = JTree(epsilon=FLAGS.epsilon_accountant)
    jtree.fit(x_train.reshape(-1, 28*28))
    jtree_time = time.time() - start_time
    print(f"JTree construction time: {jtree_time:.2f} seconds")

    print("Sampling from JTree model...")
    start_time = time.time()
    synthetic_x_train = jtree.sample(len(x_train))
    synthetic_x_train = synthetic_x_train.reshape(-1, 28, 28)
    sampling_time = time.time() - start_time
    print(f"Sampling time: {sampling_time:.2f} seconds")

    model = create_model((28, 28))

    optimizer = DPGradientDescentLaplaceOptimizer(
        l1_norm_clip=FLAGS.l1_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate
    )

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training model...")
    start_time = time.time()
    model.fit(synthetic_x_train, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, verbose=1)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")

    print("Evaluating model...")
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)

    eps = compute_epsilon(FLAGS.epochs * (len(x_train) // FLAGS.batch_size))

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
    noise_multiplier = tune_noise_multiplier(target_epsilon)
    end_time = time.time()
    print(f"Noise multiplier tuning took {end_time - start_time:.2f} seconds")

    print(f"Using noise multiplier: {noise_multiplier}")

    evaluated_results = []
    for _ in range(5):  # 训练模型5次
        test_acc, info, summary = evaluate_jtree(noise_multiplier)
        evaluated_results.append((0, info, summary))
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Training time: {summary['mean_time']:.2f} seconds")
        print(f"Epsilon: {info[0]['final_epsilon']:.4f}")

    save_results_to_csv(evaluated_results, noise_multiplier)

    print("\nHyperparameters:")
    print(f"  Noise Multiplier: {noise_multiplier}, Epochs: {FLAGS.epochs}")

if __name__ == '__main__':
    app.run(main)

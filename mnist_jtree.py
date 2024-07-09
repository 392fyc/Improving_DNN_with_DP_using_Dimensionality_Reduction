from __future__ import absolute_import, division, print_function

import csv
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentGaussianOptimizer
from dp_utils import get_epsilon_noise_multiplier, DynamicEpsilonPrintingCallback
from absl import app, flags, logging
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

logging.set_verbosity(logging.ERROR)

flags.DEFINE_float('delta', 1e-5, 'Target delta for differential privacy')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('microbatches', 1, 'Number of microbatches, must evenly divide batch_size')
flags.DEFINE_float('epsilon_accountant', 0.5, 'Target epsilon for differential privacy')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_integer('num_runs', 1, 'Number of training runs')

FLAGS = flags.FLAGS

def load_mnist_data():
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

def prepare_data(x_train, y_train, x_test, y_test, test_size=0.5):
    # Flatten the images
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    # Split the training data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_val_scaled, y_val, x_test_scaled, y_test

def create_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(1000, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(10, activation='softmax')
    ])
    return model

class JTree:
    def __init__(self, epsilon, delta, n_features):
        self.epsilon = epsilon
        self.delta = delta
        self.n_features = n_features
        self.tree = defaultdict(list)
        self.conditionals = {}
        self.feature_sizes = {}

        self.epsilon_structure = max(epsilon * 0.1, 1e-5)
        self.epsilon_conditional = epsilon - self.epsilon_structure
        self.epsilon_per_round = self.epsilon_structure / (n_features - 1)
        self.remaining_epsilon_conditional = self.epsilon_conditional
        self.edge_count = 0

    def mutual_information(self, x, y):
        joint_hist = np.histogram2d(x, y, bins=20)[0]
        p_xy = joint_hist / np.sum(joint_hist)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        p_xy_flat = p_xy.flatten()
        p_ind = p_x[:, np.newaxis] * p_y[np.newaxis, :]
        p_ind_flat = p_ind.flatten()

        epsilon = 1e-10
        p_xy_flat += epsilon
        p_ind_flat += epsilon

        mask = (p_xy_flat > 0) & (p_ind_flat > 0)
        return np.sum(p_xy_flat[mask] * np.log(p_xy_flat[mask] / p_ind_flat[mask]))

    def exponential_mechanism(self, scores):
        sensitivity = 2.0
        scores = np.array(scores)
        probabilities = np.exp(self.epsilon_per_round * scores / (2 * sensitivity))
        probabilities /= np.sum(probabilities)
        return np.random.choice(len(scores), p=probabilities)

    def gaussian_mechanism(self, true_value, sensitivity):
        if self.edge_count == 0 or self.remaining_epsilon_conditional <= 0:
            return true_value
        epsilon_per_edge = max(self.remaining_epsilon_conditional / (self.edge_count + 1), 1e-10)
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / epsilon_per_edge
        noise = np.random.normal(0, sigma)
        noisy_value = true_value + noise
        self.remaining_epsilon_conditional -= epsilon_per_edge
        return max(noisy_value, 1e-10)

    def discretize(self, data, num_bins=10):
        discretized = np.zeros_like(data, dtype=int)
        for i in range(data.shape[1]):
            discretized[:, i] = np.digitize(data[:, i], bins=np.linspace(data[:, i].min(), data[:, i].max(), num_bins))
        return discretized

    def fit(self, data):
        # 在fit方法开始时调用离散化
        discretized_data = self.discretize(data)

        remaining_attributes = set(range(self.n_features))

        for i in range(self.n_features):
            self.feature_sizes[i] = int(np.max(discretized_data[:, i])) + 1

        first_attribute = np.random.choice(list(remaining_attributes))
        remaining_attributes.remove(first_attribute)
        self.tree[first_attribute] = []

        for _ in range(1, self.n_features):
            scores = []
            candidates = []

            for attr in remaining_attributes:
                for parent in self.tree:
                    score = self.mutual_information(discretized_data[:, attr], discretized_data[:, parent])
                    scores.append(score)
                    candidates.append((attr, parent))

            chosen_idx = self.exponential_mechanism(scores)
            chosen_attr, chosen_parent = candidates[chosen_idx]

            self.tree[chosen_parent].append(chosen_attr)
            remaining_attributes.remove(chosen_attr)

            parent_data = discretized_data[:, chosen_parent]
            attr_data = discretized_data[:, chosen_attr]

            unique_parent_values = np.unique(parent_data)
            self.conditionals[chosen_attr] = {}
            for parent_value in unique_parent_values:
                self.edge_count += 1
                mask = parent_data == parent_value
                conditional_counts = np.bincount(attr_data[mask], minlength=self.feature_sizes[chosen_attr])
                probs = conditional_counts / np.sum(conditional_counts)

                sensitivity = 2 / max(len(mask), 1)
                noisy_probs = np.array([self.gaussian_mechanism(p, sensitivity) for p in probs])

                noisy_probs = np.clip(noisy_probs, 1e-10, 1)
                noisy_probs /= np.sum(noisy_probs)

                self.conditionals[chosen_attr][parent_value] = noisy_probs

    def get_epsilon_used(self):
        return self.epsilon - self.remaining_epsilon_conditional

    def sample(self, n_samples):
        samples = np.zeros((n_samples, len(self.feature_sizes)), dtype=int)
        root = next(iter(self.tree))
        samples[:, root] = np.random.choice(self.feature_sizes[root], size=n_samples)

        for parent in self.tree:
            for child in self.tree[parent]:
                parent_values = samples[:, parent]
                child_probs = np.array(
                    [self.conditionals[child].get(pv, np.ones(self.feature_sizes[child]) / self.feature_sizes[child])
                     for pv in parent_values])
                child_probs = np.apply_along_axis(lambda x: x / np.sum(x), 1, child_probs)
                child_probs = np.nan_to_num(child_probs, 0)
                zero_sum_rows = np.sum(child_probs, axis=1) == 0
                child_probs[zero_sum_rows] = np.ones(self.feature_sizes[child]) / self.feature_sizes[child]
                samples[:, child] = np.array([np.random.choice(self.feature_sizes[child], p=p) for p in child_probs])

        return samples

def evaluate_jtree(x_train, y_train, x_val, y_val, x_test, y_test, target_epsilon, delta):
    start_time = time.time()

    # JTree构建和采样
    print("Constructing and sampling from JTree model...")
    jtree = JTree(epsilon=target_epsilon * 0.5, delta=delta, n_features=x_train.shape[1])
    jtree.fit(x_train)
    synthetic_x_train = jtree.sample(len(x_train))
    jtree_epsilon_used = jtree.get_epsilon_used()

    # 将合成数据转换回连续值
    max_discrete_value = np.max(jtree.discretize(x_train))
    synthetic_x_train = synthetic_x_train.astype(float) / max_discrete_value

    # 剩余epsilon预算用于SGD训练
    remaining_epsilon = max(target_epsilon - jtree_epsilon_used, 1e-5)

    # 计算噪声乘数
    steps = FLAGS.epochs * (len(synthetic_x_train) // FLAGS.batch_size)
    noise_multiplier = get_epsilon_noise_multiplier(
        remaining_epsilon, steps, FLAGS.batch_size, len(synthetic_x_train), FLAGS.delta)

    print(f"JTree epsilon used: {jtree_epsilon_used:.4f}")
    print(f"Remaining epsilon for SGD: {remaining_epsilon:.4f}")
    print(f"Calculated noise multiplier: {noise_multiplier:.4f}")

    model = create_model(x_train.shape[1])

    optimizer = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate
    )

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    eps_callback = DynamicEpsilonPrintingCallback(
        batch_size=FLAGS.batch_size,
        num_examples=len(synthetic_x_train),
        initial_noise_multiplier=noise_multiplier,
        target_epsilon=remaining_epsilon,
        target_delta=FLAGS.delta
    )

    print("Training model...")
    history = model.fit(synthetic_x_train, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
                        validation_data=(x_val, y_val), callbacks=[eps_callback], verbose=1)

    train_time = time.time() - start_time

    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    actual_epsilon = jtree_epsilon_used + eps_callback.eps_history[-1]

    info = {
        'train_time': train_time,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'target_epsilon': target_epsilon,
        'actual_epsilon': actual_epsilon,
        'epochs_trained': FLAGS.epochs,
        'jtree_epsilon': jtree_epsilon_used,
        'sgd_epsilon': eps_callback.eps_history[-1],
        'noise_multiplier': noise_multiplier
    }

    return info

def save_results_to_csv(results):
    file_path = 'results_jtree.csv'

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Target Epsilon', 'Run', 'Total Time (s)',
                         'Test Acc', 'Actual Epsilon', 'JTree Epsilon', 'SGD Epsilon', 'Noise Multiplier'])

        for result in results:
            writer.writerow([result['target_epsilon'], result['run'],
                             result['train_time'],
                             result['test_acc'],
                             result['actual_epsilon'],
                             result['jtree_epsilon'],
                             result['sgd_epsilon'],
                             result['noise_multiplier']])
    print(f"Results saved to {file_path}")


def main(argv):
    del argv

    print("Loading data...")
    x_train, y_train, x_test, y_test = load_mnist_data()
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(x_train, y_train, x_test, y_test)

    print(f"Data loaded and preprocessed. Shape of training data: {x_train.shape}")

    #0.2, 0.3, 0.7, 1.0, 1.5, 3.0, 4.0, 6.0,
    epsilon_list = [0.5, 2.0, 8.0]

    all_results = []
    for target_epsilon in epsilon_list:
        print(f"\nExperimenting with epsilon: {target_epsilon:.4f}")

        for run in range(FLAGS.num_runs):
            print(f"Run: {run + 1}/{FLAGS.num_runs}")

            results = evaluate_jtree(x_train, y_train, x_val, y_val, x_test, y_test, target_epsilon, FLAGS.delta)
            results['run'] = run + 1
            all_results.append(results)

            print(f"Test Accuracy: {results['test_acc']:.4f}")
            print(f"Target epsilon: {results['target_epsilon']:.4f}, Actual epsilon: {results['actual_epsilon']:.4f}")
            print(f"JTree Epsilon: {results['jtree_epsilon']:.4f}, SGD Epsilon: {results['sgd_epsilon']:.4f}")
            print(f"Epochs trained: {results['epochs_trained']}")

    save_results_to_csv(all_results)

    # Calculate and print average results
    avg_results = {key: np.mean([r[key] for r in all_results]) for key in all_results[0] if key != 'run'}

    print("\nAverage Results:")
    print(f"Average Train time: {avg_results['train_time']:.2f} seconds")
    print(f"Average Test Accuracy: {avg_results['test_acc']:.4f}")
    print(f"Average Actual Epsilon: {avg_results['actual_epsilon']:.4f}")
    print(f"Average JTree Epsilon: {avg_results['jtree_epsilon']:.4f}")
    print(f"Average SGD Epsilon: {avg_results['sgd_epsilon']:.4f}")
    print(f"Average Noise Multiplier: {avg_results['noise_multiplier']:.4f}")


if __name__ == '__main__':
    app.run(main)
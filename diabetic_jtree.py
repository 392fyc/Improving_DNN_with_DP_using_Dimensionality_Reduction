from __future__ import absolute_import, division, print_function

import csv
import numpy as np
import pandas as pd
import time
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentLaplaceOptimizer
from dp_utils import evaluate_epsilon_laplace, EpsilonCallback
from absl import app, flags, logging
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.set_verbosity(logging.ERROR)

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
flags.DEFINE_float('l1_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 2048, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_integer('microbatches', 1, 'Number of microbatches, must evenly divide batch_size')
flags.DEFINE_float('epsilon', 1.0, 'Privacy budget for JTree and training combined')
flags.DEFINE_float('delta', 1e-5, 'Delta for differential privacy')

FLAGS = flags.FLAGS


class JTree:
    def __init__(self, epsilon, delta):
        self.epsilon = epsilon
        self.delta = delta
        self.tree = defaultdict(list)
        self.conditionals = {}
        self.feature_sizes = {}

    def exponential_mechanism(self, scores, epsilon):
        sensitivity = 2.0  # 假设互信息的敏感度为2
        scores = np.array(scores)
        probabilities = np.exp(epsilon * scores / (2 * sensitivity))
        probabilities /= np.sum(probabilities)
        return np.random.choice(len(scores), p=probabilities)

    def laplace_mechanism(self, true_value, sensitivity, epsilon):
        noise = np.random.laplace(0, sensitivity / epsilon)
        return true_value + noise

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

    def fit(self, data):
        n_features = data.shape[1]
        remaining_attributes = set(range(n_features))

        epsilon_per_round = self.epsilon / (2 * n_features)

        for i in range(n_features):
            self.feature_sizes[i] = int(np.max(data[:, i])) + 1

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

            chosen_idx = self.exponential_mechanism(scores, epsilon_per_round)
            chosen_attr, chosen_parent = candidates[chosen_idx]

            self.tree[chosen_parent].append(chosen_attr)
            remaining_attributes.remove(chosen_attr)

            parent_data = data[:, chosen_parent]
            attr_data = data[:, chosen_attr]

            unique_parent_values = np.unique(parent_data)
            self.conditionals[chosen_attr] = {}
            for parent_value in unique_parent_values:
                mask = parent_data == parent_value
                conditional_counts = np.bincount(attr_data[mask].astype(int), minlength=self.feature_sizes[chosen_attr])
                probs = conditional_counts / np.sum(conditional_counts)

                sensitivity = 2 / len(mask)  # 敏感度为2/n，其中n是样本数量
                noisy_probs = np.array([self.laplace_mechanism(p, sensitivity, epsilon_per_round) for p in probs])

                noisy_probs = np.clip(noisy_probs, 0, 1)
                noisy_probs /= np.sum(noisy_probs)

                self.conditionals[chosen_attr][parent_value] = noisy_probs

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


def load_diabetes_data():
    print("Loading Diabetes data...")
    try:
        data = pd.read_csv('data/diabetic_data.csv')
    except FileNotFoundError:
        print("Error: The file 'data/diabetic_data.csv' was not found.")
        return None, None, None, None, None, None, None

    columns_to_drop = ['encounter_id', 'patient_nbr']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    if np.issubdtype(data['readmitted'].dtype, np.number):
        y = data['readmitted'].astype(int)
    else:
        readmitted_map = {'NO': 0, '<30': 1, '>30': 0}
        y = data['readmitted'].map(readmitted_map).fillna(-1).astype(int)

    X = data.drop('readmitted', axis=1)

    cat_columns = X.select_dtypes(include=['object']).columns
    for col in cat_columns:
        X[col] = pd.Categorical(X[col]).codes

    X = X.fillna(X.mean())

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = np.clip(X_train_scaled, 0, None)
    X_val_scaled = np.clip(X_val_scaled, 0, None)
    X_test_scaled = np.clip(X_test_scaled, 0, None)

    original_dims = X_train_scaled.shape[1]

    print(f"Data loaded. Shape of training data: {X_train_scaled.shape}")
    print(f"Shape of validation data: {X_val_scaled.shape}")
    print(f"Shape of test data: {X_test_scaled.shape}")
    print(
        f"Positive samples in training set: {sum(y_train == 1)}/{len(y_train)} ({sum(y_train == 1) / len(y_train) * 100:.2f}%)")
    print(
        f"Positive samples in validation set: {sum(y_val == 1)}/{len(y_val)} ({sum(y_val == 1) / len(y_val) * 100:.2f}%)")
    print(
        f"Positive samples in test set: {sum(y_test == 1)}/{len(y_test)} ({sum(y_test == 1) / len(y_test) * 100:.2f}%)")

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, original_dims


def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_shape,)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    return model


def evaluate_jtree(epsilon_jtree, epsilon_sgd):
    X_train, y_train, X_val, y_val, X_test, y_test, original_dims = load_diabetes_data()
    if X_train is None:
        return None, None, None

    print("Constructing JTree model...")
    start_time = time.time()
    jtree = JTree(epsilon=epsilon_jtree, delta=FLAGS.delta)
    jtree.fit(X_train)
    jtree_time = time.time() - start_time
    print(f"JTree construction time: {jtree_time:.2f} seconds")

    print("Sampling from JTree model...")
    start_time = time.time()
    synthetic_X_train = jtree.sample(len(X_train))
    sampling_time = time.time() - start_time
    print(f"Sampling time: {sampling_time:.2f} seconds")

    model = create_model(X_train.shape[1])

    optimizer = DPGradientDescentLaplaceOptimizer(
        l1_norm_clip=FLAGS.l1_norm_clip,
        noise_multiplier=1.0,  # 初始值，将在训练过程中根据epsilon_sgd调整
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate
    )

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    epsilon_callback = EpsilonCallback(
        epsilon_accountant=epsilon_sgd,
        train_data_size=len(synthetic_X_train),
        batch_size=FLAGS.batch_size,
        l1_norm_clip=FLAGS.l1_norm_clip,
        noise_multiplier=1.0  # 初始值，将在训练过程中调整
    )

    print("Training model...")
    start_time = time.time()
    history = model.fit(synthetic_X_train, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
                        validation_data=(X_val, y_val), verbose=1, callbacks=[epsilon_callback])
    train_time = time.time() - start_time
    total_time = jtree_time + sampling_time + train_time

    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    final_epsilon = epsilon_jtree + epsilon_callback.global_epsilon

    info = {
        'epochs': FLAGS.epochs,
        'jtree_time': jtree_time,
        'sampling_time': sampling_time,
        'train_time': train_time,
        'total_time': total_time,
        'test_acc': test_acc,
        'final_epsilon': final_epsilon
    }

    return test_acc, [info], {'mean_time': total_time, 'mean_acc': test_acc}


def save_results_to_csv(evaluated_results):
    with open('results_jtree.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'Epochs', 'Total Time', 'Test Accuracy', 'Epsilon'])

        for idx, (_, infos, _) in enumerate(evaluated_results):
            for info in infos:
                row = [idx, info['epochs'], info['total_time'], info['test_acc'], info['final_epsilon']]
                writer.writerow(row)


def main(argv):
    del argv

    epsilon_jtree = FLAGS.epsilon * 0.5  # 分配一半的隐私预算给JTree
    epsilon_sgd = FLAGS.epsilon * 0.5  # 分配一半的隐私预算给SGD训练

    evaluated_results = []
    for i in range(1):  # 运行一次
        test_acc, info, summary = evaluate_jtree(epsilon_jtree, epsilon_sgd)
        if test_acc is None:
            print("Failed to evaluate JTree. Skipping this iteration.")
            continue
        evaluated_results.append((0, info, summary))
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Total time: {summary['mean_time']:.2f} seconds")
        print(f"Final Epsilon: {info[0]['final_epsilon']:.4f}")

    if evaluated_results:
        save_results_to_csv(evaluated_results)

    print("\nHyperparameters:")
    print(f"  Total Epsilon: {FLAGS.epsilon}, Epochs: {FLAGS.epochs}")


if __name__ == '__main__':
    app.run(main)
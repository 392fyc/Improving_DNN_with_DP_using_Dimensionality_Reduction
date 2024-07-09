from __future__ import absolute_import, division, print_function

import csv
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from dp_optimizer import DPGradientDescentGaussianOptimizer
from dp_utils import get_epsilon_noise_multiplier, DynamicEpsilonPrintingCallback
from absl import app, flags, logging
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import tensorflow as tf
from risk_evaluator import evaluate_model_privacy

logging.set_verbosity(logging.ERROR)

flags.DEFINE_float('delta', 1e-5, 'Target delta for differential privacy')
flags.DEFINE_float('l2_norm_clip', 1.2, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('microbatches', 1, 'Number of microbatches')
flags.DEFINE_float('epsilon_accountant', 0.5, 'Target epsilon for differential privacy')
flags.DEFINE_integer('epochs', 50, 'Number of epochs')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_integer('num_runs', 1, 'Number of training runs')

FLAGS = flags.FLAGS


def load_adult_data():
    print("Loading Adult Census Income data...")
    try:
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                        'hours-per-week', 'native-country', 'income']
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                           names=column_names, skipinitialspace=True)
    except Exception as e:
        print(f"Error: Failed to load the Adult Census Income dataset. {str(e)}")
        return None, None

    # 创建目标变量
    y = (data['income'] == '>50K').astype(int)

    # 从特征中移除目标变量
    X = data.drop('income', axis=1)

    return X, y


def feature_engineering(X):
    # 数值特征
    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    # 分类特征
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'native-country']

    # 对数值特征进行标准化
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # 对分类特征进行独热编码
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(X[categorical_features])

    # 创建新的特征名称
    feature_names = numeric_features + encoder.get_feature_names(categorical_features).tolist()

    # 合并数值特征和编码后的分类特征
    X_engineered = np.hstack((X[numeric_features].values, encoded_features))

    # 创建交互特征
    X_engineered = np.hstack((X_engineered, X_engineered[:, 0:1] * X_engineered[:, 5:6]))  # age * hours-per-week
    feature_names.append('age_hours_interaction')

    return X_engineered, feature_names


def prepare_data(X, y, test_size=0.2):
    # 特征工程
    X_engineered, feature_names = feature_engineering(X)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=test_size, random_state=np.random.randint(10000))

    return X_train, y_train, X_test, y_test, len(feature_names)


def create_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
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

        # 更保守和安全的 epsilon 分配
        self.epsilon_structure = max(epsilon * 0.1, 1e-5)  # 至少分配一个很小的值
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

    def fit(self, data):
        remaining_attributes = set(range(self.n_features))

        for i in range(self.n_features):
            self.feature_sizes[i] = int(np.max(data[:, i])) + 1

        first_attribute = np.random.choice(list(remaining_attributes))
        remaining_attributes.remove(first_attribute)
        self.tree[first_attribute] = []

        for _ in range(1, self.n_features):
            scores = []
            candidates = []

            for attr in remaining_attributes:
                for parent in self.tree:
                    score = self.mutual_information(data[:, attr], data[:, parent])
                    scores.append(score)
                    candidates.append((attr, parent))

            chosen_idx = self.exponential_mechanism(scores)
            chosen_attr, chosen_parent = candidates[chosen_idx]

            self.tree[chosen_parent].append(chosen_attr)
            remaining_attributes.remove(chosen_attr)

            parent_data = data[:, chosen_parent]
            attr_data = data[:, chosen_attr]

            unique_parent_values = np.unique(parent_data)
            self.conditionals[chosen_attr] = {}
            for parent_value in unique_parent_values:
                self.edge_count += 1
                mask = parent_data == parent_value
                conditional_counts = np.bincount(attr_data[mask].astype(int), minlength=self.feature_sizes[chosen_attr])
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


def evaluate_jtree(X, y, target_epsilon, delta, noise_multiplier):
    start_time = time.time()

    # 准备数据
    X_train, y_train, X_test, y_test, n_features = prepare_data(X, y)

    # 为JTree分配一半的epsilon预算
    jtree_epsilon = target_epsilon * 0.5

    # JTree构建和采样
    print("Constructing and sampling from JTree model...")
    jtree = JTree(epsilon=jtree_epsilon, delta=delta, n_features=n_features)
    jtree.fit(X_train)
    synthetic_X_train = jtree.sample(len(X_train))
    jtree_epsilon_used = jtree.get_epsilon_used()

    # 剩余epsilon预算用于SGD训练
    remaining_epsilon = max(target_epsilon - jtree_epsilon_used, 1e-5)

    print(f"JTree epsilon used: {jtree_epsilon_used:.4f}")
    print(f"Remaining epsilon for SGD: {remaining_epsilon:.4f}")
    print(f"Using noise multiplier: {noise_multiplier:.4f}")

    model = create_model(n_features)

    optimizer = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate,
        target_epsilon=remaining_epsilon,
        total_epochs=FLAGS.epochs
    )

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    eps_callback = DynamicEpsilonPrintingCallback(
        batch_size=FLAGS.batch_size,
        num_examples=len(synthetic_X_train),
        initial_noise_multiplier=noise_multiplier,
        target_epsilon=remaining_epsilon,
        target_delta=delta,
        total_epochs=FLAGS.epochs
    )

    print("Training model...")
    history = model.fit(synthetic_X_train, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
                        validation_split=0.2, callbacks=[eps_callback], verbose=1)

    train_time = time.time() - start_time

    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # 计算隐私评分
    privacy_results = evaluate_model_privacy(model, synthetic_X_train, y_train, X_test, y_test)

    sgd_epsilon_used = min(eps_callback.eps_history[-1], remaining_epsilon)
    actual_epsilon = jtree_epsilon_used + sgd_epsilon_used

    info = {
        'train_time': train_time,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'target_epsilon': target_epsilon,
        'actual_epsilon': actual_epsilon,
        'epochs_trained': FLAGS.epochs,
        'jtree_epsilon': jtree_epsilon_used,
        'sgd_epsilon': sgd_epsilon_used,
        'noise_multiplier': noise_multiplier,
        'privacy_score': privacy_results['privacy_score'],
        'attack_accuracy': privacy_results['attack_accuracy'],
        'attack_auc': privacy_results['attack_auc']
    }

    return info


def save_results_to_csv(results):
    file_path = 'results.csv'
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'Target Epsilon', 'Total Time (s)',
                         'Test Acc', 'Actual Epsilon',
                         'JTree Epsilon', 'SGD Epsilon', 'Noise Multiplier',
                         'Privacy Score', 'Attack Accuracy', 'Attack AUC'])

        for result in results:
            writer.writerow([
                result['run'], result['target_epsilon'],
                result['train_time'], result['test_acc'],
                result['actual_epsilon'], result['jtree_epsilon'],
                result['sgd_epsilon'], result['noise_multiplier'],
                result['privacy_score'], result['attack_accuracy'],
                result['attack_auc']
            ])

    print(f"Results saved to {file_path}")


def main(argv):
    del argv

    print("Loading data...")
    X, y = load_adult_data()
    if X is None or y is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Data loaded. Shape of data: {X.shape}")
    print(f"Positive samples: {sum(y)}/{len(y)} ({sum(y) / len(y) * 100:.2f}%)")

    # 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0
    epsilon_list = [0.5, 2.0, 8.0]

    all_results = []
    for target_epsilon in epsilon_list:
        print(f"\nExperimenting with epsilon: {target_epsilon:.4f}")

        steps = FLAGS.epochs * (len(X) // FLAGS.batch_size)
        noise_multiplier = get_epsilon_noise_multiplier(
            target_epsilon, steps, FLAGS.batch_size, len(X), FLAGS.delta)

        print(f"Calculated noise multiplier: {noise_multiplier:.4f}")

        for run in range(FLAGS.num_runs):
            print(f"Run: {run + 1}/{FLAGS.num_runs}")

            results = evaluate_jtree(X, y, target_epsilon, FLAGS.delta, noise_multiplier)
            results['run'] = run + 1
            all_results.append(results)

            print(f"Test Accuracy: {results['test_acc']:.4f}")
            print(f"Target epsilon: {results['target_epsilon']:.4f}, Actual epsilon: {results['actual_epsilon']:.4f}")
            print(f"JTree Epsilon: {results['jtree_epsilon']:.4f}, SGD Epsilon: {results['sgd_epsilon']:.4f}")
            print(f"Epochs trained: {results['epochs_trained']}")
            print(f"Privacy Score: {results['privacy_score']:.4f}")
            print(f"Attack Accuracy: {results['attack_accuracy']:.4f}")
            print(f"Attack AUC: {results['attack_auc']:.4f}")

    save_results_to_csv(all_results)

    # Calculate and print average results
    avg_results = defaultdict(float)
    for key in all_results[0].keys():
        if key != 'run':
            avg_results[key] = np.mean([r[key] for r in all_results])

    print("\nAverage Results:")
    print(f"Average Train time: {avg_results['train_time']:.2f} seconds")
    print(f"Average Test Accuracy: {avg_results['test_acc']:.4f}")
    print(f"Average Actual Epsilon: {avg_results['actual_epsilon']:.4f}")
    print(f"Average JTree Epsilon: {avg_results['jtree_epsilon']:.4f}")
    print(f"Average SGD Epsilon: {avg_results['sgd_epsilon']:.4f}")
    print(f"Average Noise Multiplier: {avg_results['noise_multiplier']:.4f}")
    print(f"Average Privacy Score: {avg_results['privacy_score']:.4f}")
    print(f"Average Attack Accuracy: {avg_results['attack_accuracy']:.4f}")
    print(f"Average Attack AUC: {avg_results['attack_auc']:.4f}")

if __name__ == '__main__':
    app.run(main)
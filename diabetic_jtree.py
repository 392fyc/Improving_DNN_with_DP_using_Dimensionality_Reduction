from __future__ import absolute_import, division, print_function

import csv
import numpy as np
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentLaplaceOptimizer
from dp_utils import evaluate_epsilon_laplace, EpsilonCallback
from absl import app, flags, logging
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

logging.set_verbosity(logging.ERROR)

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
flags.DEFINE_float('l1_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 2048, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_integer('microbatches', 1, 'Number of microbatches, must evenly divide batch_size')
flags.DEFINE_float('epsilon_accountant', 0.5, 'Privacy budget for JTree and training combined')
flags.DEFINE_integer('num_runs', 10, 'Number of training runs')

FLAGS = flags.FLAGS


def tune_noise_multiplier_laplace(target_epsilon, l1_norm_clip, batch_size, epochs, train_data_size):
    print("Starting noise multiplier tuning...")
    lower = 0.1
    upper = 2000.0
    tolerance = 1e-3
    iterations = 0
    max_iterations = 50

    while upper - lower > tolerance and iterations < max_iterations:
        iterations += 1
        noise_multiplier = (lower + upper) / 2
        print(f"Iteration {iterations}: Testing noise_multiplier = {noise_multiplier}")

        epsilon = evaluate_epsilon_laplace(
            l1_norm_clip=l1_norm_clip,
            noise_multiplier=noise_multiplier,
            batch_size=batch_size,
            epochs=epochs,
            train_data_size=train_data_size
        )
        print(f"Iteration {iterations}: Trying noise_multiplier={noise_multiplier:.3f}, got epsilon={epsilon:.3f}")
        if abs(epsilon - target_epsilon) <= tolerance:
            print(f"Optimal noise_multiplier={noise_multiplier:.3f}, epsilon={epsilon:.3f}")
            return noise_multiplier
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
        return None, None

    # 预处理数据
    columns_to_drop = ['encounter_id', 'patient_nbr', 'gender']  # 移除 gender
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    # 创建两个目标变量
    y1 = (data['race'] == 'AfricanAmerican').astype(int)  # 种族是否为黑人
    y2 = (data['readmitted'] == '<30').astype(int)  # 30天内是否再次入院

    # 从特征中移除目标变量
    X = data.drop(['race', 'readmitted'], axis=1)

    # 将分类变量转换为数值
    cat_columns = X.select_dtypes(include=['object']).columns
    for col in cat_columns:
        X[col] = pd.Categorical(X[col]).codes

    # 处理特征中的缺失值
    X = X.fillna(X.mean())

    return X, [y1, y2]

def prepare_data(X, y, test_size=0.2):
    # 分割数据集
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        X, y[0], y[1], test_size=test_size, random_state=np.random.randint(10000))

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 确保所有特征值非负
    X_train_scaled = np.clip(X_train_scaled, 0, None)
    X_test_scaled = np.clip(X_test_scaled, 0, None)

    return X_train_scaled, [y1_train, y2_train], X_test_scaled, [y1_test, y2_test]


def create_model(input_shape):
    inputs = Input(shape=(input_shape,))

    # Race-specific layers
    race_x = Dense(3 * input_shape, activation='relu', kernel_regularizer=l2(0.01))(inputs)
    race_x = Dense(2 * input_shape, activation='relu', kernel_regularizer=l2(0.01))(race_x)
    race_x = Dense(input_shape, activation='relu', kernel_regularizer=l2(0.01))(race_x)

    # Shared layers
    x = Dense(2 * input_shape, activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = Dense(input_shape, activation='relu', kernel_regularizer=l2(0.01))(x)

    # Concatenate specific and shared layers
    race_combined = Concatenate()([race_x, x])

    output1 = Dense(1, activation='sigmoid', name='race')(race_combined)
    output2 = Dense(1, activation='sigmoid', name='readmission')(x)

    model = Model(inputs=inputs, outputs=[output1, output2])
    return model


class JTree:
    def __init__(self, epsilon, n_features):
        self.epsilon = epsilon
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

    def laplace_mechanism(self, true_value, sensitivity):
        if self.edge_count == 0 or self.remaining_epsilon_conditional <= 0:
            return true_value
        epsilon_per_edge = max(self.remaining_epsilon_conditional / (self.edge_count + 1), 1e-10)
        noise_scale = sensitivity / epsilon_per_edge
        noise = np.random.laplace(0, noise_scale)
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

                sensitivity = 2 / max(len(mask), 1)  # 避免除以零
                noisy_probs = np.array([self.laplace_mechanism(p, sensitivity) for p in probs])

                noisy_probs = np.clip(noisy_probs, 1e-10, 1)  # 确保所有概率都是正数
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


def evaluate_jtree(X, y):
    # 准备数据
    start_time = time.time()
    X_train, y_train, X_test, y_test = prepare_data(X, y)
    data_prep_time = time.time() - start_time

    # 分配epsilon预算
    epsilon_jtree = FLAGS.epsilon_accountant * 0.5  # 50% 用于JTree
    epsilon_sgd = FLAGS.epsilon_accountant * 0.5  # 50% 用于SGD训练

    print("Constructing JTree model...")
    jtree_start_time = time.time()
    jtree = JTree(epsilon=epsilon_jtree, n_features=X_train.shape[1])
    jtree.fit(X_train)
    jtree_time = time.time() - jtree_start_time
    print(f"JTree construction time: {jtree_time:.2f} seconds")
    print(f"JTree epsilon used: {jtree.get_epsilon_used():.4f}")

    # 检查 JTree 是否使用了过多的 epsilon
    if jtree.get_epsilon_used() > epsilon_jtree:
        print(f"Warning: JTree used more epsilon than allocated. Used: {jtree.get_epsilon_used():.4f}, Allocated: {epsilon_jtree:.4f}")
        epsilon_sgd = max(FLAGS.epsilon_accountant - jtree.get_epsilon_used(), 0)
    else:
        epsilon_sgd = FLAGS.epsilon_accountant - jtree.get_epsilon_used()

    print(f"Remaining epsilon for SGD: {epsilon_sgd:.4f}")

    print("Sampling from JTree model...")
    sampling_start_time = time.time()
    synthetic_X_train = jtree.sample(len(X_train))
    sampling_time = time.time() - sampling_start_time

    # 调整 SGD 训练的 epsilon
    remaining_epsilon = epsilon_sgd - jtree.get_epsilon_used()
    if remaining_epsilon <= 0:
        print("Warning: No epsilon budget left for SGD training")
        remaining_epsilon = 1e-5  # Set a small positive value to allow some training

    # 调整噪声乘数
    noise_multiplier = tune_noise_multiplier_laplace(
        target_epsilon=remaining_epsilon,
        l1_norm_clip=FLAGS.l1_norm_clip,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        train_data_size=len(synthetic_X_train)
    )

    model = create_model(X_train.shape[1])

    optimizer = DPGradientDescentLaplaceOptimizer(
        l1_norm_clip=FLAGS.l1_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate
    )

    model.compile(optimizer=optimizer,
                  loss={'race': 'binary_crossentropy', 'readmission': 'binary_crossentropy'},
                  metrics=['accuracy'])

    epsilon_callback = EpsilonCallback(
        epsilon_accountant=epsilon_sgd,
        train_data_size=len(synthetic_X_train),
        batch_size=FLAGS.batch_size,
        l1_norm_clip=FLAGS.l1_norm_clip,
        noise_multiplier=noise_multiplier
    )

    print("Training model...")
    model_train_start_time = time.time()
    history = model.fit(synthetic_X_train, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
                        validation_split=0.2, verbose=1, callbacks=[epsilon_callback])
    model_train_time = time.time() - model_train_start_time

    train_time = data_prep_time + jtree_time + sampling_time + model_train_time

    print("Evaluating model...")
    test_loss, race_loss, readmission_loss, race_acc, readmission_acc = model.evaluate(X_test, y_test, verbose=0)

    final_epsilon = jtree.get_epsilon_used() + epsilon_callback.global_epsilon

    info = {
        'epochs': FLAGS.epochs,
        'jtree_time': jtree_time,
        'sampling_time': sampling_time,
        'model_train_time': model_train_time,
        'train_time': train_time,
        'race_acc': race_acc,
        'readmission_acc': readmission_acc,
        'final_epsilon': final_epsilon,
        'jtree_epsilon': jtree.get_epsilon_used(),
        'sgd_epsilon': epsilon_callback.global_epsilon,
        'noise_multiplier': noise_multiplier
    }

    return (race_acc, readmission_acc), [info], {'mean_time': train_time, 'mean_race_acc': race_acc,
                                                 'mean_readmission_acc': readmission_acc}


def save_results_to_csv(results):
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'Epochs', 'Train Time', 'Race Accuracy', 'Readmission Accuracy',
                         'JTree Epsilon', 'SGD Epsilon', 'Final Epsilon'])

        for idx, result in enumerate(results):
            writer.writerow([idx + 1, result['epochs'], result['train_time'],
                             result['race_acc'], result['readmission_acc'],
                             result['jtree_epsilon'], result['sgd_epsilon'], result['final_epsilon']])


def main(argv):
    del argv

    print("Loading Diabetes data...")
    X, y = load_diabetes_data()
    if X is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Data loaded. Shape of data: {X.shape}")
    print(f"Positive samples for race: {sum(y[0] == 1)}/{len(y[0])} ({sum(y[0] == 1) / len(y[0]) * 100:.2f}%)")
    print(f"Positive samples for readmission: {sum(y[1] == 1)}/{len(y[1])} ({sum(y[1] == 1) / len(y[1]) * 100:.2f}%)")

    all_results = []
    for run in range(FLAGS.num_runs):
        start_time = time.time()
        print(f"\nStarting run {run + 1}/{FLAGS.num_runs}")

        accuracies, info, summary = evaluate_jtree(X, y)
        if accuracies is None:
            print("Failed to evaluate JTree. Skipping this run.")
            continue

        run_time = time.time() - start_time
        all_results.append(info[0])  # info is a list with one dictionary

        print(f"Run {run + 1} completed in {run_time:.2f} seconds.")
        print(f"Test Race Accuracy: {info[0]['race_acc']:.4f}, "
              f"Test Readmission Accuracy: {info[0]['readmission_acc']:.4f}")
        print(f"JTree Epsilon: {info[0]['jtree_epsilon']:.4f}")
        print(f"SGD Epsilon: {info[0]['sgd_epsilon']:.4f}")
        print(f"Total Epsilon: {info[0]['final_epsilon']:.4f}")
        print(f"Noise Multiplier: {info[0]['noise_multiplier']:.4f}")

    if all_results:
        print("\nAll runs completed. Calculating average results...")
        avg_results = {
            'race_acc': np.mean([r['race_acc'] for r in all_results]),
            'readmission_acc': np.mean([r['readmission_acc'] for r in all_results]),
            'train_time': np.mean([r['train_time'] for r in all_results]),
            'final_epsilon': np.mean([r['final_epsilon'] for r in all_results]),
            'noise_multiplier': np.mean([r['noise_multiplier'] for r in all_results])
        }

        print("\nAverage Results:")
        print(f"Average Train time: {avg_results['train_time']:.2f} seconds")
        print(f"Average Test Race Accuracy: {avg_results['race_acc']:.4f}")
        print(f"Average Test Readmission Accuracy: {avg_results['readmission_acc']:.4f}")
        print(f"Average Epsilon: {avg_results['final_epsilon']:.4f}")
        print(f"Average Noise Multiplier: {avg_results['noise_multiplier']:.4f}")

        save_results_to_csv(all_results)
        print("Results saved to results_jtree.csv")
    else:
        print("No successful runs completed.")

    print("\nHyperparameters:")
    print(f"  Target Epsilon: {FLAGS.epsilon_accountant}, Epochs: {FLAGS.epochs}, Number of Runs: {FLAGS.num_runs}")

if __name__ == '__main__':
    app.run(main)
from __future__ import absolute_import, division, print_function

import csv
import numpy as np
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentLaplaceOptimizer
from dp_utils import evaluate_epsilon_laplace
from absl import app, flags, logging
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import os

logging.set_verbosity(logging.ERROR)

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
flags.DEFINE_float('l1_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 2048, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_integer('microbatches', 1, 'Number of microbatches, must evenly divide batch_size')
flags.DEFINE_integer('pca_components', 17, 'Number of PCA components to use')
flags.DEFINE_float('epsilon_accountant', 0.5, 'Target epsilon for differential privacy')
flags.DEFINE_integer('num_runs', 50, 'Number of training runs')

FLAGS = flags.FLAGS


def tune_noise_multiplier_laplace(X, y, target_epsilon, n_components, tolerance=1e-3):
    print("Starting noise multiplier tuning...")
    lower = 0.1
    upper = 200000.0
    iterations = 0
    max_iterations = 50

    train_data, _, _, _ = prepare_data(X, y, n_components)
    train_data_size = len(train_data)
    steps = FLAGS.epochs * (train_data_size // FLAGS.batch_size)

    while upper - lower > tolerance and iterations < max_iterations:
        iterations += 1
        noise_multiplier = (lower + upper) / 2
        print(f"Iteration {iterations}: Testing noise_multiplier = {noise_multiplier}")

        epsilon = (2 * FLAGS.l1_norm_clip * steps) / (noise_multiplier * FLAGS.batch_size)
        print(f"Iteration {iterations}: Trying noise_multiplier={noise_multiplier:.3f}, got epsilon={epsilon:.3f}")
        if abs(epsilon - target_epsilon) <= tolerance:
            optimal_noise_multiplier = noise_multiplier
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


def cumulative_explained_variance(X, target_variance=0.90):
    """累计解释方差法选择最优维度"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    n_components = np.argmax(cumulative_variance >= target_variance) + 1
    return n_components, cumulative_variance


def prepare_data(X, y, n_components, test_size=0.2):
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

    # 应用PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, [y1_train, y2_train], X_test_pca, [y1_test, y2_test]


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


def evaluate(X, y, noise_multiplier, n_components):
    start_time = time.time()

    # 准备数据
    train_data, train_labels, test_data, test_labels = prepare_data(X, y, n_components)

    input_shape = train_data.shape[1]
    model = create_model(input_shape)

    optimizer = DPGradientDescentLaplaceOptimizer(
        l1_norm_clip=FLAGS.l1_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)

    model.compile(optimizer=optimizer,
                  loss={'race': 'binary_crossentropy',
                        'readmission': 'binary_crossentropy'},
                  metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=FLAGS.epochs,
              batch_size=FLAGS.batch_size,
              verbose=0)

    total_time = time.time() - start_time

    # 评估测试集性能
    test_loss, race_loss, readmission_loss, race_acc, readmission_acc = model.evaluate(test_data, test_labels, verbose=0)

    epsilon = evaluate_epsilon_laplace(
        l1_norm_clip=FLAGS.l1_norm_clip,
        noise_multiplier=noise_multiplier,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        train_data_size=len(train_data)
    )

    info = {
        'total_time': total_time,
        'test_race_acc': race_acc,
        'test_readmission_acc': readmission_acc,
        'test_race_loss': race_loss,
        'test_readmission_loss': readmission_loss,
        'epsilon': epsilon,
    }

    return info


def save_results_to_csv(results):
    file_path = 'results.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Run', 'PCA Components', 'Total Time (s)',
                             'Test Race Acc', 'Test Readmission Acc',
                             'Epsilon'])

        for run, result in enumerate(results):
            writer.writerow([run + 1, FLAGS.pca_components, result['total_time'],
                             result['test_race_acc'], result['test_readmission_acc'],
                             result['epsilon']])


def main(argv):
    del argv

    print("Loading data...")
    X, y = load_diabetes_data()
    if X is None or y is None:
        print("Failed to load data. Exiting.")
        return

    # 选择最佳PCA维度
    optimal_components, cumulative_variance = cumulative_explained_variance(X)
    print(f"Optimal number of components according to CEV: {optimal_components}")

    target_epsilon = FLAGS.epsilon_accountant
    noise_multiplier = tune_noise_multiplier_laplace(X, y, target_epsilon, optimal_components)
    if noise_multiplier is None:
        print("Failed to tune noise multiplier. Exiting.")
        return

    print(f"Tuned noise multiplier: {noise_multiplier}")

    all_results = []
    for run in range(FLAGS.num_runs):
        start_time = time.time()
        print(f"\nStarting run {run + 1}/{FLAGS.num_runs}")
        results = evaluate(X, y, noise_multiplier, optimal_components)
        run_time = time.time() - start_time
        all_results.append(results)
        print(f"Run {run + 1} completed in {run_time:.2f} seconds.")
        print(f"Test Race Accuracy: {results['test_race_acc']:.4f}, "
              f"Test Readmission Accuracy: {results['test_readmission_acc']:.4f}")
        print(f"Actual epsilon: {results['epsilon']:.4f}")

    print("\nAll runs completed. Calculating average results...")
    avg_results = {
        'test_race_acc': np.mean([r['test_race_acc'] for r in all_results]),
        'test_readmission_acc': np.mean([r['test_readmission_acc'] for r in all_results]),
        'total_time': np.mean([r['total_time'] for r in all_results]),
        'epsilon': np.mean([r['epsilon'] for r in all_results])
    }

    print("\nAverage Results:")
    print(f"Average Total time: {avg_results['total_time']:.2f} seconds")
    print(f"Average Test Race Accuracy: {avg_results['test_race_acc']:.4f}")
    print(f"Average Test Readmission Accuracy: {avg_results['test_readmission_acc']:.4f}")
    print(f"Average Epsilon: {avg_results['epsilon']:.4f}")

    save_results_to_csv(all_results)
    print("Results saved to results.csv")

if __name__ == '__main__':
    app.run(main)
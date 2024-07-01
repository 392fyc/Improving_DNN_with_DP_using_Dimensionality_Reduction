from __future__ import absolute_import, division, print_function

import csv
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentLaplaceOptimizer
from dp_utils import evaluate_epsilon_laplace, EpsilonCallback
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
flags.DEFINE_boolean('use_pca', True, 'Use PCA for dimensionality reduction')
flags.DEFINE_float('epsilon_accountant', 0.5, 'Epsilon threshold to stop training for DP-SGD.')

FLAGS = flags.FLAGS

def tune_noise_multiplier_laplace(target_epsilon, tolerance=1e-3):
    print("Starting noise multiplier tuning...")
    lower = 0.1
    upper = 200000.0
    iterations = 0
    max_iterations = 50

    X_train, _, _, _, _ = load_diabetes_data()
    if X_train is None:
        print("Error loading data. Exiting noise multiplier tuning.")
        return None

    train_data_size = len(X_train)

    while upper - lower > tolerance and iterations < max_iterations:
        iterations += 1
        noise_multiplier = (lower + upper) / 2
        print(f"Iteration {iterations}: Testing noise_multiplier = {noise_multiplier}")

        epsilon = evaluate_epsilon_laplace(
            l1_norm_clip=FLAGS.l1_norm_clip,
            noise_multiplier=noise_multiplier,
            batch_size=FLAGS.batch_size,
            epochs=FLAGS.epochs,
            train_data_size=train_data_size
        )
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

def load_diabetes_data(n_components=None):
    print("Loading Diabetes data...")
    try:
        data = pd.read_csv('data/diabetic_data.csv')
    except FileNotFoundError:
        print("Error: The file 'data/diabetic_data.csv' was not found.")
        return None, None, None, None, None

    # 预处理数据
    columns_to_drop = ['encounter_id', 'patient_nbr']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    # 创建三个目标变量
    y1 = (data['gender'] == 'Male').astype(int)  # 是否为男性
    y2 = (data['race'] == 'AfricanAmerican').astype(int)  # 种族是否为黑人
    y3 = (data['readmitted'] == '<30').astype(int)  # 30天内是否再次入院

    # 从特征中移除目标变量
    X = data.drop(['gender', 'race', 'readmitted'], axis=1)

    # 将分类变量转换为数值
    cat_columns = X.select_dtypes(include=['object']).columns
    for col in cat_columns:
        X[col] = pd.Categorical(X[col]).codes

    # 处理特征中的缺失值
    X = X.fillna(X.mean())

    # 分割数据集
    X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
        X, y1, y2, y3, test_size=0.2, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 确保所有特征值非负
    X_train_scaled = np.clip(X_train_scaled, 0, None)
    X_test_scaled = np.clip(X_test_scaled, 0, None)

    original_dims = X_train_scaled.shape[1]

    if FLAGS.use_pca and n_components is not None:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        return X_train_pca, [y1_train, y2_train, y3_train], X_test_pca, [y1_test, y2_test, y3_test], original_dims
    else:
        return X_train_scaled, [y1_train, y2_train, y3_train], X_test_scaled, [y1_test, y2_test, y3_test], original_dims


def create_model(input_shape):
    inputs = Input(shape=(input_shape,))

    # 将隐藏层大小与输入维度绑定
    x = Dense(2 * input_shape, activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = Dense(input_shape, activation='relu', kernel_regularizer=l2(0.01))(x)

    output1 = Dense(1, activation='sigmoid', name='gender')(x)
    output2 = Dense(1, activation='sigmoid', name='race')(x)
    output3 = Dense(1, activation='sigmoid', name='readmission')(x)

    model = Model(inputs=inputs, outputs=[output1, output2, output3])
    return model

def epsilon_vs_noise_multiplier(noise_multipliers, original_dims, max_run_time):
    epsilons = []
    for noise_multiplier in noise_multipliers:
        info = evaluate(original_dims, noise_multiplier, max_run_time)
        epsilons.append(info['epsilon'])
    return epsilons

def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def evaluate(n_components, noise_multiplier, max_run_time):
    start_time = time.time()
    train_data, train_labels, test_data, test_labels, original_dims = load_diabetes_data(n_components)

    input_shape = train_data.shape[1]
    model = create_model(input_shape)

    optimizer = DPGradientDescentLaplaceOptimizer(
        l1_norm_clip=FLAGS.l1_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)

    model.compile(optimizer=optimizer,
                  loss={'gender': 'binary_crossentropy',
                        'race': 'binary_crossentropy',
                        'readmission': 'binary_crossentropy'},
                  metrics=['accuracy'])

    epsilon_callback = EpsilonCallback(
        epsilon_accountant=FLAGS.epsilon_accountant,
        train_data_size=len(train_data),
        batch_size=FLAGS.batch_size,
        l1_norm_clip=FLAGS.l1_norm_clip,
        noise_multiplier=noise_multiplier
    )

    model.fit(train_data, train_labels,
              epochs=FLAGS.epochs,
              batch_size=FLAGS.batch_size,
              verbose=0,
              callbacks=[epsilon_callback])

    total_time = time.time() - start_time

    # 评估测试集性能
    test_loss, gender_loss, race_loss, readmission_loss, gender_acc, race_acc, readmission_acc = model.evaluate(test_data, test_labels, verbose=0)

    info = {
        'pca_components': n_components,
        'total_time': total_time,
        'test_gender_acc': gender_acc,
        'test_race_acc': race_acc,
        'test_readmission_acc': readmission_acc,
        'test_gender_loss': gender_loss,
        'test_race_loss': race_loss,
        'test_readmission_loss': readmission_loss,
        'epsilon': epsilon_callback,
    }

    return info

def calculate_score(run_time, test_acc, max_run_time, max_acc=1.0, time_weight=1.0, accuracy_weight=1.0):
    normalized_time = max(0, (max_run_time - run_time) / max_run_time)
    normalized_accuracy = test_acc / max_acc
    score = (time_weight * normalized_time + accuracy_weight * normalized_accuracy) / (time_weight + accuracy_weight)
    return score


def find_optimal_dimensions(max_run_time, original_dims, noise_multiplier):
    evaluated_dims = []

    # 对原始维度进行训练，获取时间阈值
    original_info = evaluate(original_dims, noise_multiplier, max_run_time)
    original_score = (original_info['test_gender_acc'] + original_info['test_race_acc'] + original_info['test_readmission_acc']) / 3
    evaluated_dims.append((original_dims, original_score, original_info))
    time_threshold = original_info['total_time']

    def search_recursive(center, step):
        if step == 1:
            left_dim, center_dim, right_dim = center - 1, center, center + 1

            left_info = evaluate(left_dim, noise_multiplier, max_run_time)
            left_score = (left_info['test_gender_acc'] + left_info['test_race_acc'] + left_info['test_readmission_acc']) / 3
            evaluated_dims.append((left_dim, left_score, left_info))

            center_info = evaluate(center_dim, noise_multiplier, max_run_time)
            center_score = (center_info['test_gender_acc'] + center_info['test_race_acc'] + center_info['test_readmission_acc']) / 3
            evaluated_dims.append((center_dim, center_score, center_info))

            right_info = evaluate(right_dim, noise_multiplier, max_run_time)
            right_score = (right_info['test_gender_acc'] + right_info['test_race_acc'] + right_info['test_readmission_acc']) / 3
            evaluated_dims.append((right_dim, right_score, right_info))

            if center_score >= left_score and center_score >= right_score:
                return center_dim
            elif left_score >= center_score and left_score >= right_score:
                return left_dim
            else:
                return right_dim
        else:
            left_dim, right_dim = max(1, center - step), min(original_dims, center + step)

            left_info = evaluate(left_dim, noise_multiplier, max_run_time)
            left_score = (left_info['test_gender_acc'] + left_info['test_race_acc'] + left_info['test_readmission_acc']) / 3
            evaluated_dims.append((left_dim, left_score, left_info))

            center_info = evaluate(center, noise_multiplier, max_run_time)
            center_score = (center_info['test_gender_acc'] + center_info['test_race_acc'] + center_info['test_readmission_acc']) / 3
            evaluated_dims.append((center, center_score, center_info))

            right_info = evaluate(right_dim, noise_multiplier, max_run_time)
            right_score = (right_info['test_gender_acc'] + right_info['test_race_acc'] + right_info['test_readmission_acc']) / 3
            evaluated_dims.append((right_dim, right_score, right_info))

            if center_score >= left_score and center_score >= right_score:
                return search_recursive(center, step // 2)
            elif left_score >= center_score and left_score >= right_score:
                return search_recursive(left_dim, step // 2)
            else:
                return search_recursive(right_dim, step // 2)

    initial_center = original_dims // 2
    initial_step = original_dims // 4
    optimal_dim = search_recursive(initial_center, initial_step)

    # 重新计算所有评估维度的分数，使用时间阈值进行归一化
    for i in range(len(evaluated_dims)):
        dim, score, info = evaluated_dims[i]
        normalized_score = calculate_score(info['total_time'], score, time_threshold)
        evaluated_dims[i] = (dim, normalized_score, info)

    return optimal_dim, evaluated_dims


def save_results_to_csv(all_results):
    file_path = 'results.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Run', 'PCA Components', 'Total Time (s)',
                             'Test Gender Acc', 'Test Race Acc', 'Test Readmission Acc',
                             'Epsilon'])

        for idx, info in enumerate(all_results):
            row = [idx, info['pca_components'], info['total_time'],
                   info['test_gender_acc'], info['test_race_acc'], info['test_readmission_acc'],
                   info['epsilon']]
            writer.writerow(row)

def main(argv):
    del argv

    target_epsilon = FLAGS.epsilon_accountant
    total_runs = 20  # 设置总训练次数，可以根据需要调整

    print("Epsilon optimization:")
    noise_multiplier = tune_noise_multiplier_laplace(target_epsilon)

    print(f"\nDimension optimization:")
    print(f"Using noise multiplier: {noise_multiplier}")

    # Load the full dataset to get original dimensions
    X_train, _, _, _, original_dims = load_diabetes_data()
    print(f"Original data dimensions: {original_dims}")

    original_info = evaluate(original_dims, noise_multiplier, float('inf'))
    max_run_time = original_info['total_time']
    print(f"Mean running time for original dimensions ({original_dims}): {max_run_time:.2f} seconds")

    optimal_dim, evaluated_dims = find_optimal_dimensions(max_run_time, original_dims, noise_multiplier)
    print(f"\nOptimal PCA dimensions: {optimal_dim}")

    optimization_runs = len(evaluated_dims)
    remaining_runs = max(0, total_runs - optimization_runs)

    all_results = [info for _, _, info in evaluated_dims]

    # 进行剩余的训练
    for i in range(remaining_runs):
        print(f"\nAdditional training run {i + 1}/{remaining_runs}")
        info = evaluate(optimal_dim, noise_multiplier, max_run_time)
        all_results.append(info)

    # 计算并打印最终结果
    mean_gender_acc = np.mean([info['test_gender_acc'] for info in all_results])
    mean_race_acc = np.mean([info['test_race_acc'] for info in all_results])
    mean_readmission_acc = np.mean([info['test_readmission_acc'] for info in all_results])
    mean_epsilon = np.mean([info['epsilon'] for info in all_results])

    print("\nFinal results:")
    print(f"Optimal PCA dimensions: {optimal_dim}")
    print(f"Noise multiplier: {noise_multiplier:.4f}")
    print(f"Mean gender accuracy: {mean_gender_acc:.4f}")
    print(f"Mean race accuracy: {mean_race_acc:.4f}")
    print(f"Mean readmission accuracy: {mean_readmission_acc:.4f}")
    print(f"Mean epsilon: {mean_epsilon:.4f}")

    # 保存所有结果
    save_results_to_csv(all_results)

    print("\nHyperparameters:")
    print(f"  Noise Multiplier: {noise_multiplier}, Epochs: {FLAGS.epochs}")

if __name__ == '__main__':
    app.run(main)
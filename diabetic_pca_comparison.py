from __future__ import absolute_import, division, print_function

import csv
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from absl import app, flags, logging
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf

logging.set_verbosity(logging.ERROR)

flags.DEFINE_float('delta', 1e-5, 'Target delta for differential privacy')
flags.DEFINE_float('epsilon', 1.0, 'Privacy budget for input perturbation')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_boolean('use_pca', True, 'Whether to use PCA for dimensionality reduction')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate for training')
flags.DEFINE_integer('num_runs', 1, 'Number of training runs')

FLAGS = flags.FLAGS


def load_diabetes_data():
    print("Loading Diabetes data...")
    try:
        data = pd.read_csv('data/diabetic_data.csv')
    except FileNotFoundError:
        print("Error: The file 'data/diabetic_data.csv' was not found.")
        return None, None

    columns_to_drop = ['encounter_id', 'patient_nbr', 'gender']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    y1 = (data['race'] == 'AfricanAmerican').astype(int)
    y2 = (data['readmitted'] == '<30').astype(int)

    X = data.drop(['race', 'readmitted'], axis=1)

    cat_columns = X.select_dtypes(include=['object']).columns
    for col in cat_columns:
        X[col] = pd.Categorical(X[col]).codes

    X = X.fillna(X.mean())

    return X, [y1, y2]


def add_noise_to_data(data, epsilon, l2_sensitivity):
    """Add Gaussian noise to achieve differential privacy"""
    sigma = np.sqrt(2 * np.log(1.25 / FLAGS.delta)) * l2_sensitivity / epsilon
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise


def cumulative_explained_variance(X, target_variance=0.90):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    n_components = np.argmax(cumulative_variance >= target_variance) + 1
    return n_components, cumulative_variance


def prepare_data(X, y, n_components=None, test_size=0.5):
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        X, y[0], y[1], test_size=test_size, random_state=np.random.randint(10000))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if FLAGS.use_pca and n_components is not None:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # 计算L2敏感度（使用PCA后的数据）
        l2_sensitivity = np.linalg.norm(X_train_pca, axis=1).max()

        # 添加噪音到PCA处理后的训练数据
        X_train_noisy = add_noise_to_data(X_train_pca, FLAGS.epsilon, l2_sensitivity)

        X_train_final = X_train_noisy
        X_test_final = X_test_pca
    else:
        # 如果不使用PCA，直接对标准化后的数据添加噪音
        l2_sensitivity = np.linalg.norm(X_train_scaled, axis=1).max()
        X_train_noisy = add_noise_to_data(X_train_scaled, FLAGS.epsilon, l2_sensitivity)

        X_train_final = X_train_noisy
        X_test_final = X_test_scaled

    return X_train_final, [y1_train, y2_train], X_test_final, [y1_test, y2_test]


def create_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(input_shape, activation='sigmoid', kernel_regularizer=l2(0.01)),
        Dense(2 * input_shape, activation='sigmoid', kernel_regularizer=l2(0.01)),
        Dense(input_shape, activation='sigmoid', kernel_regularizer=l2(0.01)),
        Dense(2, activation='sigmoid', name='output')
    ])

    return model


def evaluate(X, y, n_components, epsilon):
    start_time = time.time()

    train_data, train_labels, test_data, test_labels = prepare_data(X, y, n_components)

    input_shape = train_data.shape[1]
    model = create_model(input_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    y_train = np.column_stack(train_labels)
    y_test = np.column_stack(test_labels)

    history = model.fit(train_data, y_train,
                        epochs=FLAGS.epochs,
                        batch_size=FLAGS.batch_size,
                        validation_split=0.2,
                        verbose=1)

    train_time = time.time() - start_time

    test_loss, test_acc = model.evaluate(test_data, y_test, verbose=0)

    y_pred = model.predict(test_data)
    race_acc = np.mean((y_pred[:, 0] > 0.5) == y_test[:, 0])
    readmission_acc = np.mean((y_pred[:, 1] > 0.5) == y_test[:, 1])

    info = {
        'train_time': train_time,
        'test_acc': test_acc,
        'test_race_acc': race_acc,
        'test_readmission_acc': readmission_acc,
        'test_loss': test_loss,
        'epsilon': epsilon,
        'epochs_trained': FLAGS.epochs
    }

    return info

def save_results_to_csv(results, dimension_optimization_time, optimal_components):
    file_path = 'results.csv'
    avg_dim_opt_time = dimension_optimization_time / len(results) if FLAGS.use_pca else 0

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epsilon', 'Run', 'PCA Components', 'Total Time (s)',
                         'Test Race Acc', 'Test Readmission Acc'])

        for result in results:
            adjusted_time = result['train_time'] + avg_dim_opt_time
            writer.writerow([result['epsilon'], result['run'],
                             optimal_components if FLAGS.use_pca else 'N/A',
                             adjusted_time,
                             result['test_race_acc'],
                             result['test_readmission_acc']])
    print(f"Results saved to {file_path}")


def main(argv):
    del argv

    print("Loading data...")
    X, y = load_diabetes_data()
    if X is None or y is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Data loaded. Shape of data: {X.shape}")
    print(f"Positive samples for race: {sum(y[0])}/{len(y[0])} ({sum(y[0]) / len(y[0]) * 100:.2f}%)")
    print(f"Positive samples for readmission: {sum(y[1])}/{len(y[1])} ({sum(y[1]) / len(y[1]) * 100:.2f}%)")

    optimal_components = None
    dimension_optimization_time = 0

    if FLAGS.use_pca:
        dim_opt_start_time = time.time()
        optimal_components, cumulative_variance = cumulative_explained_variance(X)
        dimension_optimization_time = time.time() - dim_opt_start_time
        print(f"Optimal number of components according to CEV: {optimal_components}")
        print(f"Dimension optimization time: {dimension_optimization_time:.2f} seconds")
    else:
        print("PCA is disabled. Using all original features.")

    epsilon_list = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]

    all_results = []
    for epsilon in epsilon_list:
        for run in range(FLAGS.num_runs):
            print(f"\nExperimenting with epsilon: {epsilon:.4f}, Run: {run + 1}/{FLAGS.num_runs}")

            results = evaluate(X, y, optimal_components, epsilon)
            results['run'] = run + 1
            results['epsilon'] = epsilon
            all_results.append(results)

            print(f"Test Race Accuracy: {results['test_race_acc']:.4f}, "
                  f"Test Readmission Accuracy: {results['test_readmission_acc']:.4f}")
            print(f"Overall Test Accuracy: {results['test_acc']:.4f}")
            print(f"Epsilon: {epsilon:.4f}")
            print(f"Epochs trained: {results['epochs_trained']}")

    save_results_to_csv(all_results, dimension_optimization_time, optimal_components)

if __name__ == '__main__':
    app.run(main)
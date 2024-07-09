from __future__ import absolute_import, division, print_function

import csv
import numpy as np
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from dp_optimizer import DPGradientDescentGaussianOptimizer
from dp_utils import get_epsilon_noise_multiplier, DynamicEpsilonPrintingCallback
from absl import app, flags, logging
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf
from risk_evaluator import evaluate_model_privacy
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

logging.set_verbosity(logging.ERROR)

flags.DEFINE_float('delta', 1e-5, 'Target delta for differential privacy')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 2048, 'Batch size')
flags.DEFINE_integer('microbatches', 128, 'Number of microbatches, must evenly divide batch_size')
flags.DEFINE_boolean('use_pca', True, 'Whether to use PCA for dimensionality reduction')
flags.DEFINE_float('epsilon_accountant', 0.5, 'Target epsilon for differential privacy')
flags.DEFINE_integer('epochs', 200, 'Number of epochs')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for training')
flags.DEFINE_integer('num_runs', 1, 'Number of training runs')

FLAGS = flags.FLAGS


def load_diabetes_data():
    print("Loading Diabetes data...")
    try:
        data = pd.read_csv('data/preprocessed_data.csv')
        y = data['readmitted']
        X = data.drop('readmitted', axis=1)
        return X, y
    except FileNotFoundError:
        print("Error: The file 'data/preprocessed_data.csv' was not found.")
        return None, None


def prepare_data(X, y, n_components=None, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if FLAGS.use_pca and n_components is not None:
        pca = PCA(n_components=n_components)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    # Apply SMOTE only to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train)

    return X_train_resampled, y_train_resampled, X_test_final, y_test


def cumulative_explained_variance(X, target_variance=0.95):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    n_components = np.argmax(cumulative_variance >= target_variance) + 1
    return n_components, cumulative_variance


def create_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    return model


def evaluate(X, y, noise_multiplier, n_components, target_epsilon):
    start_time = time.time()

    X_train, y_train, X_test, y_test = prepare_data(X, y, n_components)

    input_shape = X_train.shape[1]
    model = create_model(input_shape)

    optimizer = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    eps_callback = DynamicEpsilonPrintingCallback(
        batch_size=FLAGS.batch_size,
        num_examples=len(X_train),
        initial_noise_multiplier=noise_multiplier,
        target_epsilon=target_epsilon,
        target_delta=FLAGS.delta,
        total_epochs=FLAGS.epochs
    )

    history = model.fit(X_train, y_train,
                        epochs=FLAGS.epochs,
                        batch_size=FLAGS.batch_size,
                        validation_split=0.2,
                        callbacks=[eps_callback],
                        verbose=1)

    train_time = time.time() - start_time

    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    actual_epsilon = eps_callback.eps_history[-1]

    info = {
        'train_time': train_time,
        'test_acc': accuracy,
        'test_f1': f1,
        'confusion_matrix': conf_matrix,
        'target_epsilon': target_epsilon,
        'actual_epsilon': actual_epsilon,
        'epochs_trained': FLAGS.epochs,
        'noise_multiplier': noise_multiplier
    }

    privacy_results = evaluate_model_privacy(model, X_train, y_train, X_test, y_test)
    info.update(privacy_results)

    return info


def save_results_to_csv(results, dimension_optimization_time, optimal_components):
    file_path = 'results.csv'
    avg_dim_opt_time = dimension_optimization_time / len(results) if FLAGS.use_pca else 0

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'Target Epsilon', 'PCA Components', 'Total Time (s)',
                         'Test Accuracy', 'Test F1 Score', 'Actual Epsilon',
                         'Noise Multiplier', 'Privacy Score', 'Attack Accuracy', 'Attack AUC'])

        for result in results:
            adjusted_time = result['train_time'] + avg_dim_opt_time
            writer.writerow([
                result['run'], result['target_epsilon'],
                optimal_components if FLAGS.use_pca else 'N/A',
                adjusted_time, result['test_acc'],
                result['test_f1'], result['actual_epsilon'],
                result['noise_multiplier'], result['privacy_score'],
                result['attack_accuracy'], result['attack_auc']
            ])

    print(f"Results saved to {file_path}")


def main(argv):
    del argv

    X, y = load_diabetes_data()
    if X is None or y is None:
        return

    print(f"Data loaded. Shape of data: {X.shape}")
    print(f"Class distribution: {y.value_counts()}")

    optimal_components = None
    if FLAGS.use_pca:
        optimal_components, cumulative_variance = cumulative_explained_variance(X)
        print(f"Optimal number of PCA components: {optimal_components}")
        print(f"Cumulative explained variance: {cumulative_variance[optimal_components-1]:.4f}")

    epsilon_list = [0.5, 2.0, 8.0]

    all_results = []
    for target_epsilon in epsilon_list:
        print(f"\nExperimenting with epsilon: {target_epsilon:.4f}")

        steps = FLAGS.epochs * (len(X) // FLAGS.batch_size)
        noise_multiplier = get_epsilon_noise_multiplier(
            target_epsilon, steps, FLAGS.batch_size, len(X), FLAGS.delta)

        print(f"Calculated noise multiplier: {noise_multiplier:.4f}")

        for run in range(FLAGS.num_runs):
            print(f"\nRun: {run + 1}/{FLAGS.num_runs}")

            results = evaluate(X, y, noise_multiplier, optimal_components, target_epsilon)
            results['run'] = run + 1
            all_results.append(results)

            print(f"Test Accuracy: {results['test_acc']:.4f}")
            print(f"Test F1 Score: {results['test_f1']:.4f}")
            print(f"Actual epsilon: {results['actual_epsilon']:.4f}")
            print("Confusion Matrix:")
            print(results['confusion_matrix'])

    # Save results to CSV
    pd.DataFrame(all_results).to_csv('diabetic_pca_results.csv', index=False)


if __name__ == '__main__':
    app.run(main)
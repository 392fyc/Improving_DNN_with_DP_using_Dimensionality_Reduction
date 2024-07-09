from __future__ import absolute_import, division, print_function

import csv
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from absl import app, flags, logging
from dp_optimizer import DPGradientDescentGaussianOptimizer
from dp_utils import get_epsilon_noise_multiplier, DynamicEpsilonPrintingCallback
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
flags.DEFINE_float('l2_norm_clip', 4.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 600, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_integer('microbatches', 600, 'Number of microbatches')
flags.DEFINE_integer('num_runs', 1, 'Number of training runs per epsilon')
flags.DEFINE_float('delta', 1e-5, 'Target delta for differential privacy')

FLAGS = flags.FLAGS
DELTA = 1e-5

def load_fashion_mnist():
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    return train_data, train_labels, test_data, test_labels


class FastRandomizedPCA:
    def __init__(self, n_components=None, batch_size=1000, random_state=None):
        self.n_components = n_components
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.svd = None
        self.n_samples_seen = 0

    def partial_fit(self, X):
        if self.svd is None:
            self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
            self.svd.fit(X)
        else:
            batch_components = TruncatedSVD(n_components=self.n_components, random_state=self.random_state).fit(X).components_
            alpha = X.shape[0] / self.n_samples_seen
            self.svd.components_ = (1 - alpha) * self.svd.components_ + alpha * batch_components
        self.n_samples_seen += X.shape[0]

    def transform(self, X):
        return self.svd.transform(X)


def evaluate_components(n_components, X_train, X_test, train_labels, test_labels):
    rpca = FastRandomizedPCA(n_components=n_components)
    for i in range(0, len(X_train), rpca.batch_size):
        batch = X_train[i:i + rpca.batch_size]
        rpca.partial_fit(batch.reshape(len(batch), -1))

    X_train_pca = rpca.transform(X_train.reshape(len(X_train), -1))
    X_test_pca = rpca.transform(X_test.reshape(len(X_test), -1))

    model = create_model(X_train_pca.shape[1:])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_pca, train_labels, epochs=5, validation_data=(X_test_pca, test_labels), verbose=0)
    return n_components, history.history['val_accuracy'][-1]


def find_optimal_components_parallel(X_train, X_test, train_labels, test_labels):

    # 内部参数，可以根据需要调整
    min_components = 100  # 最小PCA组件数
    max_components = 700  # 最大PCA组件数
    component_step = 50  # PCA组件数的步长
    patience = 3  # 早停的耐心值
    n_jobs = -1  # 并行作业数，-1表示使用所有可用的处理器

    start_time = time.time()

    results = []
    best_score = 0
    best_n_components = 0
    no_improvement_count = 0

    for n_components in range(min_components, max_components + 1, component_step):
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_components)(n_components, X_train, X_test, train_labels, test_labels)
            for _ in range(1)  # 每个组件数只评估一次
        )

        _, score = batch_results[0]
        results.append((n_components, score))

        if score > best_score:
            best_score = score
            best_n_components = n_components
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping at {n_components} components")
            break

    end_time = time.time()
    search_time = end_time - start_time
    return best_n_components, search_time


def create_model(input_shape):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dense(10, kernel_regularizer=regularizers.l2(0.0001))
    ])
    return model


def train_model(model, train_data, train_labels, test_data, test_labels, noise_multiplier, target_epsilon):
    optimizer = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    eps_callback = DynamicEpsilonPrintingCallback(
        batch_size=FLAGS.batch_size,
        num_examples=len(train_data),
        initial_noise_multiplier=noise_multiplier,
        target_epsilon=target_epsilon,
        target_delta=FLAGS.delta,
        total_epochs=FLAGS.epochs
    )

    steps_per_epoch = len(train_data) // FLAGS.batch_size

    history = model.fit(
        train_data, train_labels,
        epochs=FLAGS.epochs,
        validation_data=(test_data, test_labels),
        batch_size=FLAGS.batch_size,
        steps_per_epoch=steps_per_epoch,
        callbacks=[eps_callback],
        verbose=1
    )

    test_acc = history.history['val_accuracy'][-1]
    final_epsilon = eps_callback.eps_history[-1]

    return test_acc, final_epsilon


def main(argv):
    del argv

    train_data, train_labels, test_data, test_labels = load_fashion_mnist()

    best_n_components, search_time = find_optimal_components_parallel(
        train_data, test_data, train_labels, test_labels)

    # 使用最佳组件数进行PCA转换
    rpca = FastRandomizedPCA(n_components=best_n_components)
    for i in range(0, len(train_data), rpca.batch_size):
        batch = train_data[i:i + rpca.batch_size]
        rpca.partial_fit(batch.reshape(len(batch), -1))

    train_data_pca = rpca.transform(train_data.reshape(len(train_data), -1))
    test_data_pca = rpca.transform(test_data.reshape(len(test_data), -1))

    epsilon_list = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0]
    results = []

    for target_epsilon in epsilon_list:
        print(f"\nCalculating for epsilon: {target_epsilon:.4f}")
        steps = FLAGS.epochs * (len(train_data_pca) // FLAGS.batch_size)
        noise_multiplier = get_epsilon_noise_multiplier(
            target_epsilon, steps, FLAGS.batch_size, len(train_data_pca), FLAGS.delta)

        print(f"Calculated noise multiplier: {noise_multiplier:.4f}")

        for run in range(FLAGS.num_runs):
            print(f"Run {run + 1}/{FLAGS.num_runs}")
            start_time = time.time()

            model = create_model(train_data_pca.shape[1:])
            test_acc, actual_epsilon = train_model(model, train_data_pca, train_labels, test_data_pca, test_labels,
                                                   noise_multiplier, target_epsilon)

            train_time = time.time() - start_time

            result = {
                'run': run + 1,
                'target_epsilon': target_epsilon,
                'actual_epsilon': actual_epsilon,
                'noise_multiplier': noise_multiplier,
                'test_accuracy': test_acc,
                'train_time': train_time
            }
            results.append(result)

            print(f"Test accuracy: {test_acc:.4f}")
            print(f"Actual epsilon: {actual_epsilon:.2f}")
            print(f"Training time: {train_time:.2f} seconds")

    # 保存结果到单个CSV文件
    with open('results.csv', 'w', newline='') as file:
        fieldnames = ['run', 'target_epsilon', 'actual_epsilon', 'noise_multiplier', 'test_accuracy', 'train_time']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Results saved to results.csv")

    # 在所有训练完成后输出PCA相关信息
    print(f"\nPCA Optimization Results:")
    print(f"Best number of components: {best_n_components}")
    print(f"Time taken to find optimal components: {search_time:.2f} seconds")

if __name__ == '__main__':
    app.run(main)

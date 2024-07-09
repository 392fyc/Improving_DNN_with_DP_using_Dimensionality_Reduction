import numpy as np
import pandas as pd
from absl import app, flags, logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate for training')
flags.DEFINE_integer('batch_size', 512, 'Batch size')
flags.DEFINE_integer('microbatches', 16, 'Number of microbatches (not directly used in this version)')
flags.DEFINE_integer('epochs', 400, 'Number of epochs')
flags.DEFINE_float('l2_norm_clip', 1.0, 'L2 norm clip for gradient (not directly used in this version)')
flags.DEFINE_integer('num_runs', 1, 'Number of training runs')


def load_data():
    print("Loading data...")
    data = pd.read_csv('data/preprocessed_data.csv')
    y = data['readmitted']
    X = data.drop('readmitted', axis=1)
    return X, y


def prepare_data(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return X_train_resampled, y_train_resampled, X_test_scaled, y_test


def create_model(input_shape):
    model = Sequential([
        Dense(1024, activation=LeakyReLU(alpha=0.01), input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(512, activation=LeakyReLU(alpha=0.01)),
        Dropout(0.3),
        Dense(3, activation=LeakyReLU(alpha=0.01))
    ])
    return model


def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = create_model(X_train.shape[1])
    optimizer = Adam(learning_rate=FLAGS.learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        epochs=FLAGS.epochs,
                        batch_size=FLAGS.batch_size,
                        validation_split=0.2,
                        verbose=1)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\nTest Accuracy:", accuracy)
    print("Test F1 Score:", f1)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, history, accuracy, f1, conf_matrix


def main(argv):
    del argv  # Unused

    X, y = load_data()
    print(f"Data loaded. Shape of data: {X.shape}")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")

    print("\nPreparing data...")
    X_train, y_train, X_test, y_test = prepare_data(X, y)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    all_results = []
    for run in range(FLAGS.num_runs):
        print(f"\nRun {run + 1}/{FLAGS.num_runs}")
        print("\nTraining and evaluating model...")
        model, history, accuracy, f1, conf_matrix = train_and_evaluate(X_train, y_train, X_test, y_test)

        results = {
            'run': run + 1,
            'test_acc': accuracy,
            'test_f1': f1,
            'confusion_matrix': conf_matrix.tolist()
        }
        all_results.append(results)

    # Print average results
    avg_accuracy = np.mean([r['test_acc'] for r in all_results])
    avg_f1 = np.mean([r['test_f1'] for r in all_results])
    print(f"\nAverage Test Accuracy: {avg_accuracy:.4f}")
    print(f"Average Test F1 Score: {avg_f1:.4f}")


if __name__ == '__main__':
    app.run(main)
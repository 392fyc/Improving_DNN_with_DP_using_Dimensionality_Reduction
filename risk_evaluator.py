import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

class MembershipInferenceAttack:
    def __init__(self, target_model, attack_model_epochs=10):
        self.target_model = target_model
        self.attack_model = None
        self.attack_model_epochs = attack_model_epochs

    def _create_attack_model(self, input_shape):
        inputs = Input(shape=(input_shape,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _prepare_attack_data(self, X, y):
        predictions = self.target_model.predict(X)
        print(f"Predictions shape: {np.shape(predictions)}")
        print(f"y shape: {np.shape(y)}")

        if isinstance(predictions, list):  # 处理多输出模型
            predictions = np.concatenate(predictions, axis=1)

        # Ensure y is 2D
        y = np.array(y).reshape(-1, 1)

        # Ensure predictions and y have the same number of samples
        min_samples = min(predictions.shape[0], y.shape[0])
        predictions = predictions[:min_samples]
        y = y[:min_samples]

        print(f"Processed predictions shape: {predictions.shape}")
        print(f"Processed y shape: {y.shape}")

        attack_data = np.concatenate([predictions, y], axis=1)
        print(f"Attack data shape: {attack_data.shape}")
        return attack_data

    def train_attack_model(self, X_train, y_train, X_test, y_test):
        try:
            train_data = self._prepare_attack_data(X_train, y_train)
            test_data = self._prepare_attack_data(X_test, y_test)

            train_labels = np.ones(len(X_train))
            test_labels = np.zeros(len(X_test))

            all_data = np.vstack([train_data, test_data])
            all_labels = np.concatenate([train_labels, test_labels])

            attack_X_train, attack_X_test, attack_y_train, attack_y_test = train_test_split(
                all_data, all_labels, test_size=0.3, random_state=42
            )

            self.attack_model = self._create_attack_model(attack_X_train.shape[1])
            self.attack_model.fit(
                attack_X_train, attack_y_train,
                epochs=self.attack_model_epochs,
                batch_size=32,
                validation_data=(attack_X_test, attack_y_test),
                verbose=0
            )

            attack_acc = self.attack_model.evaluate(attack_X_test, attack_y_test, verbose=0)[1]
            attack_auc = roc_auc_score(attack_y_test, self.attack_model.predict(attack_X_test))

            return attack_acc, attack_auc
        except Exception as e:
            print(f"Error in train_attack_model: {str(e)}")
            raise

    def evaluate_privacy(self, X_train, y_train, X_test, y_test):
        try:
            attack_acc, attack_auc = self.train_attack_model(X_train, y_train, X_test, y_test)
            privacy_score = 1 - attack_auc

            return {
                'attack_accuracy': attack_acc,
                'attack_auc': attack_auc,
                'privacy_score': privacy_score
            }
        except Exception as e:
            print(f"Error in evaluate_privacy: {str(e)}")
            raise

def evaluate_model_privacy(model, X_train, y_train, X_test, y_test):
    try:
        mia = MembershipInferenceAttack(model)
        privacy_results = mia.evaluate_privacy(X_train, y_train, X_test, y_test)
        return privacy_results
    except Exception as e:
        print(f"Error in evaluate_model_privacy: {str(e)}")
        raise
import tensorflow as tf

"""
    Compute epsilon for Laplace mechanism.

    Args:
    l1_sensitivity: L1 sensitivity of the query
    noise_scale: Scale of Laplace noise (b in Lap(b))
    num_iterations: Number of iterations or queries

    Returns:
    epsilon: Privacy loss
"""

def compute_epsilon_laplace(l1_sensitivity, noise_scale, num_iterations):
    epsilon_per_iteration = l1_sensitivity / noise_scale
    epsilon = epsilon_per_iteration * num_iterations
    return epsilon


def evaluate_epsilon_laplace(l1_norm_clip, noise_multiplier, batch_size, epochs, train_data_size):
    steps = epochs * (train_data_size // batch_size)
    epsilon = (2 * l1_norm_clip * steps) / (noise_multiplier * batch_size)
    return epsilon


class EpsilonCallback(tf.keras.callbacks.Callback):
    def __init__(self, epsilon_accountant, train_data_size, batch_size, l1_norm_clip, noise_multiplier):
        self.epsilon_accountant = epsilon_accountant
        self.train_data_size = train_data_size
        self.batch_size = batch_size
        self.l1_norm_clip = l1_norm_clip
        self.noise_multiplier = noise_multiplier
        self.global_epsilon = 0

    def on_epoch_end(self, epoch, logs=None):
        steps = (epoch + 1) * (self.train_data_size // self.batch_size)
        l1_sensitivity = 2 * self.l1_norm_clip
        noise_scale = self.l1_norm_clip * self.noise_multiplier
        self.global_epsilon = compute_epsilon_laplace(l1_sensitivity, noise_scale, steps)

        if self.global_epsilon >= self.epsilon_accountant:
            self.model.stop_training = True

        print(f"Current epsilon: {self.global_epsilon}")
import numpy as np
from privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent
import tensorflow as tf


def compute_epsilon(steps, noise_multiplier, batch_size, num_examples, target_delta=1e-5):
    """Computes epsilon value for given hyperparameters."""
    if noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = batch_size / num_examples
    rdp = compute_rdp(q=sampling_probability,
                      noise_multiplier=noise_multiplier,
                      steps=steps,
                      orders=orders)
    return get_privacy_spent(orders, rdp, target_delta=target_delta)[0]


class DynamicEpsilonPrintingCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, num_examples, initial_noise_multiplier, target_epsilon, target_delta, total_epochs):
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.initial_noise_multiplier = initial_noise_multiplier
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.total_epochs = total_epochs
        self.eps_history = []

    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model.optimizer, 'increment_epoch'):
            self.model.optimizer.increment_epoch()

    def on_epoch_end(self, epoch, logs=None):
        noise_multiplier = self.initial_noise_multiplier
        if hasattr(self.model.optimizer, '_noise_multiplier'):
            noise_multiplier = self.model.optimizer._noise_multiplier

        steps = (epoch + 1) * (self.num_examples // self.batch_size)
        eps = compute_epsilon(
            steps=steps,
            noise_multiplier=noise_multiplier,
            batch_size=self.batch_size,
            num_examples=self.num_examples,
            target_delta=self.target_delta
        )
        self.eps_history.append(eps)
        print(f', eps = {eps:.4f}')

    def calculate_next_noise_multiplier(self, next_epoch):
        # 这里使用二分查找来找到合适的noise_multiplier
        remaining_epochs = self.total_epochs - next_epoch
        remaining_epsilon = self.target_epsilon - self.eps_history[-1]

        def get_next_epsilon(noise_mult):
            next_steps = (next_epoch + 1) * (self.num_examples // self.batch_size)
            return compute_epsilon(next_steps, noise_mult, self.batch_size, self.num_examples, self.target_delta)

        low, high = 0.01, 1000.0
        while high - low > 1e-5:
            mid = (low + high) / 2
            next_eps = get_next_epsilon(mid)
            if next_eps > remaining_epsilon / remaining_epochs:
                low = mid
            else:
                high = mid
        return (low + high) / 2


def get_epsilon_noise_multiplier(target_epsilon, steps, batch_size, num_examples, target_delta=1e-5, tolerance=1e-7):
    """Binary search to find the initial noise multiplier for a target epsilon."""
    low, high = 0.1, 10000.0
    while high - low > tolerance:
        mid = (low + high) / 2
        eps = compute_epsilon(steps, mid, batch_size, num_examples, target_delta)
        if eps < target_epsilon:
            high = mid
        else:
            low = mid
    return (low + high) / 2
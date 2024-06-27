import tensorflow as tf
from tensorflow_privacy.privacy.analysis import privacy_ledger


class DPGradientDescentLaplaceOptimizer(tf.keras.optimizers.SGD):
    def __init__(self, l1_norm_clip, noise_multiplier, num_microbatches=None, learning_rate=0.01, *args, **kwargs):
        super(DPGradientDescentLaplaceOptimizer, self).__init__(learning_rate=learning_rate, *args, **kwargs)
        self.l1_norm_clip = l1_norm_clip
        self.noise_multiplier = noise_multiplier
        self._num_microbatches = num_microbatches

    def compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
        if tape is None:
            raise ValueError("A tape needs to be passed.")

        vector_loss = loss()
        if self._num_microbatches is None:
            self._num_microbatches = tf.shape(vector_loss)[0]

        def clip_gradients(gradients):
            return [tf.clip_by_norm(grad, self.l1_norm_clip, axes=[0], ord=1) for grad in gradients]

        def add_noise(gradients):
            return [grad + tf.random.laplace(tf.shape(grad), 0.0, self.noise_multiplier * self.l1_norm_clip)
                    for grad in gradients]

        microbatches_losses = tf.reshape(vector_loss, [self._num_microbatches, -1])
        final_grads = []

        for idx in range(self._num_microbatches):
            microbatch_loss = tf.reduce_mean(tf.gather(microbatches_losses, [idx]))
            grads = tape.gradient(microbatch_loss, var_list)
            clipped_grads = clip_gradients(grads)
            noised_grads = add_noise(clipped_grads)
            final_grads.append(noised_grads)

        average_grads = [tf.reduce_mean(tf.stack([grads[i] for grads in final_grads]), axis=0)
                         for i in range(len(var_list))]

        return list(zip(average_grads, var_list))

class LaplaceSumQuery:
    def __init__(self, l1_norm_clip, l1_norm_epsilon):
        self._l1_norm_clip = l1_norm_clip
        self._l1_norm_epsilon = l1_norm_epsilon

    def initial_global_state(self):
        return None

    def derive_sample_params(self, global_state):
        return None

    def initial_sample_state(self, templates):
        return tf.nest.map_structure(tf.zeros_like, templates)

    def accumulate_record(self, params, sample_state, record):
        clipped_record = tf.clip_by_norm(record, self._l1_norm_clip, axes=[0], ord=1)
        return tf.nest.map_structure(tf.add, sample_state, clipped_record)

    def get_noised_result(self, sample_state, global_state):
        def add_noise(v):
            noise_stddev = self._l1_norm_clip / self._l1_norm_epsilon
            noise = tf.random.laplace(tf.shape(v), 0.0, noise_stddev)
            return v + noise

        return tf.nest.map_structure(add_noise, sample_state), global_state

def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, dp_sum_query, num_microbatches=None, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)
            self._dp_sum_query = dp_sum_query
            self._num_microbatches = num_microbatches
            self._global_state = self._dp_sum_query.initial_global_state()

        def compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
            if tape is None:
                raise ValueError("A tape needs to be passed.")

            vector_loss = loss()
            if self._num_microbatches is None:
                self._num_microbatches = tf.shape(vector_loss)[0]
            sample_state = self._dp_sum_query.initial_sample_state(var_list)
            microbatches_losses = tf.reshape(vector_loss, [self._num_microbatches, -1])
            sample_params = self._dp_sum_query.derive_sample_params(self._global_state)

            for idx in range(self._num_microbatches):
                microbatch_loss = tf.reduce_mean(tf.gather(microbatches_losses, [idx]))
                grads = tape.gradient(microbatch_loss, var_list)
                sample_state = self._dp_sum_query.accumulate_record(sample_params, sample_state, grads)

            grad_sums, self._global_state = self._dp_sum_query.get_noised_result(sample_state, self._global_state)

            def normalize(v):
                return v / tf.cast(self._num_microbatches, tf.float32)

            final_grads = tf.nest.map_structure(normalize, grad_sums)

            return list(zip(final_grads, var_list))

    return DPOptimizerClass

def make_laplace_optimizer_class(cls):
    class DPLaplaceOptimizerClass(make_optimizer_class(cls)):
        def __init__(self, l1_norm_clip, noise_multiplier, num_microbatches=None, ledger=None, *args, **kwargs):
            dp_sum_query = LaplaceSumQuery(l1_norm_clip, l1_norm_clip * noise_multiplier)

            if ledger:
                dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query, ledger=ledger)

            super(DPLaplaceOptimizerClass, self).__init__(dp_sum_query, num_microbatches, *args, **kwargs)

        @property
        def ledger(self):
            return getattr(self._dp_sum_query, 'ledger', None)

    return DPLaplaceOptimizerClass

DPGradientDescentLaplaceOptimizer = make_laplace_optimizer_class(tf.keras.optimizers.SGD)

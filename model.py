import numpy as np
import tensorflow as tf


class PreferenceElicitationModel:
    def __init__(
            self,
            item_embeddings: np.ndarray,
            item_embedding_size: int = 20,
            episode_length: int = 50,
            state_size: int = 256):
        """

        :param item_embeddings:
        :param item_embedding_size:
        :param episode_length:
        :param state_size:
        """
        self.item_indices = tf.placeholder(dtype=tf.int32, shape=(None, None), name='item_indices')
        self.targets = tf.placeholder(dtype=tf.float32, shape=(None, None), name='targets')

        item_embeddings_fixed = tf.get_variable(
            'item_embeddings_fixed',
            shape=item_embeddings.shape,
            dtype=tf.float32,
            initializer=tf.constant_initializer(item_embeddings),
            trainable=False)

        num_users = item_embeddings.shape[0]

        item_embeddings_trainable = tf.get_variable(
            'item_embeddings_trainable',
            shape=(num_users, item_embedding_size),
            dtype=tf.float32,
            initializer=tf.initializers.glorot_uniform())

        items = tf.concat([
            tf.nn.embedding_lookup(item_embeddings_fixed, self.item_indices),
            tf.nn.embedding_lookup(item_embeddings_trainable, self.item_indices)], axis=-1)

        log_probability, reward = self.create_controller(items, self.targets, episode_length, state_size)

        self.loss = self.calculate_loss(log_probability, reward)

    @staticmethod
    def gather_subset(tensor: tf.Tensor, *dimension_indices):
        """

        :param tensor:
        :param dimension_indices:
        :return:
        """
        previous_lengths = []

        for dimension, indices in enumerate(dimension_indices):
            indices_shape = tf.unstack(tf.shape(indices))

            target_shape = [dimension for dimension in indices_shape[:-1]] + [1 for _ in range(dimension)] + [indices_shape[-1]]

            tiling = [1 for _ in range(len(indices.shape) - 1)] + previous_lengths + [1]

            indices = tf.tile(tf.reshape(indices, target_shape), tiling)

            tensor = tf.batch_gather(tensor, indices)

            previous_lengths.append(indices_shape[-1])

        return tensor

    @staticmethod
    def get_indices(mask: tf.Tensor, value: int):
        """

        :param mask:
        :param value:
        :return:
        """
        return tf.cast(tf.where(tf.equal(mask, value))[:, 1], dtype=tf.int32)

    @staticmethod
    def get_selections(tensor: tf.Tensor, indices: tf.Tensor):
        """

        :param tensor:
        :param indices:
        :return:
        """
        return tf.squeeze(tf.batch_gather(tensor, tf.expand_dims(indices, axis=-1)), axis=1)

    @staticmethod
    def generate_unknown_indices(known_indices, episode_length):
        """

        :param known_indices:
        :param episode_length:
        :return:
        """
        batch_size, num_known = tf.unstack(tf.shape(known_indices))

        offsets = tf.expand_dims(tf.range(batch_size * episode_length, delta=episode_length, dtype=tf.int32), axis=-1)

        known_indices_flat = tf.reshape(known_indices + offsets, (-1,))

        mask_values = tf.ones_like(known_indices_flat, dtype=tf.bool)

        mask = tf.logical_not(tf.scatter_nd(tf.expand_dims(known_indices_flat, axis=-1), mask_values, (batch_size * episode_length,)))

        indices = tf.tile(tf.range(episode_length, dtype=tf.int32), [batch_size])

        unknown_indices = tf.reshape(tf.boolean_mask(indices, mask), (batch_size, episode_length - num_known))

        return unknown_indices

    @staticmethod
    def calculate_cosine_similarities(items):
        """

        :param items:
        :return:
        """
        norms = tf.sqrt(tf.reduce_sum(tf.square(items), axis=2), name='item_norms')

        cosine_similarities = tf.matmul(items, items, transpose_b=True) / tf.expand_dims(tf.square(norms), 1)

        return cosine_similarities

    def select(self, items, unknown_indices, known_indices, state, cosine_similarities):
        """

        :param items:
        :param unknown_indices:
        :param known_indices:
        :param state:
        :param cosine_similarities:
        :return:
        """
        controller_item_weights = tf.get_variable(
            'controller_item_weights',
            (state.shape[-1], items.shape[-1]),
            tf.float32,
            initializer=tf.initializers.glorot_uniform())

        unknown_items = tf.batch_gather(items, unknown_indices)
        controller_similarities = unknown_items * tf.matmul(state, controller_item_weights)

        unknown_cosine_similarities = self.gather_subset(cosine_similarities, unknown_indices, unknown_indices)
        known_cosine_similarities = self.gather_subset(cosine_similarities, unknown_indices, known_indices)

        item_similarities = tf.stack(
            [
                tf.reduce_min(unknown_cosine_similarities, axis=-1),
                tf.reduce_mean(unknown_cosine_similarities, axis=-1),
                tf.reduce_max(unknown_cosine_similarities, axis=-1),
                tf.reduce_min(known_cosine_similarities, axis=-1),
                tf.reduce_mean(known_cosine_similarities, axis=-1),
                tf.reduce_max(known_cosine_similarities, axis=-1),
            ],
            axis=-1)

        similarities = tf.concat((controller_similarities, item_similarities), axis=-1)

        num_similarities = similarities.shape[-1]

        controller_similarity_weights = tf.get_variable(
            'controller_similarity_weights',
            (state.shape[-1], num_similarities),
            tf.float32,
            initializer=tf.initializers.glorot_uniform())

        feature_weights = tf.get_variable(
            'feature_weights',
            (num_similarities,),
            tf.float32,
            initializer=tf.initializers.glorot_uniform())

        gating_vector = tf.expand_dims(tf.sigmoid(tf.matmul(state, controller_similarity_weights)), axis=1)

        logits = tf.tensordot(similarities * gating_vector, feature_weights, axes=[2, 0])

        selected_index = tf.reshape(tf.multinomial(logits, 1, output_dtype=tf.int32), shape=(-1,))

        log_probabilities = tf.log(tf.nn.softmax(logits, axis=-1))

        log_probability = self.get_selections(log_probabilities, selected_index)

        return selected_index, log_probability

    @staticmethod
    def read(items, targets, state_size):
        """

        :param items:
        :param targets:
        :param state_size:
        :return:
        """
        controller_input_weights = tf.get_variable(
            'controller_input_weights',
            (items.shape[-1] + 1, state_size),
            tf.float32)

        controller_input = tf.matmul(tf.concat((items, tf.expand_dims(targets, -1)), axis=-1), controller_input_weights)

        return controller_input

    def predict(self, items, prediction_indices, known_indices, state, cosine_similarities, targets):
        """

        :param items:
        :param prediction_indices:
        :param known_indices:
        :param state:
        :param cosine_similarities:
        :param targets:
        :return:
        """
        sharpening_weights = tf.get_variable(
            'sharpening_weights',
            (state.shape[-1], items.shape[-1]),
            tf.float32)

        unknown_items = tf.batch_gather(items, prediction_indices)
        sharpening_values = tf.exp(tf.reduce_sum(unknown_items * tf.expand_dims(tf.matmul(state, sharpening_weights), axis=1), axis=-1))

        known_cosine_similarities = self.gather_subset(cosine_similarities, prediction_indices, known_indices)
        attention_weights = tf.nn.softmax(known_cosine_similarities / tf.expand_dims(sharpening_values, axis=-1), axis=-1)

        predictions = tf.reduce_sum(attention_weights * tf.expand_dims(tf.batch_gather(targets, known_indices), axis=1), axis=-1)

        return predictions

    @staticmethod
    def calculate_rewards(predictions, targets):
        """

        :param predictions:
        :param targets:
        :return:
        """
        reward = -tf.reduce_sum(tf.square(predictions - targets), axis=-1)

        return reward

    def create_controller(self, items, targets, episode_length=50, state_size=256):
        """

        :param items:
        :param targets:
        :param episode_length:
        :param state_size:
        :return:
        """
        batch_size, num_items, _ = tf.unstack(tf.shape(items))

        lstm = tf.nn.rnn_cell.LSTMCell(state_size)

        cosine_similarities = self.calculate_cosine_similarities(items)

        def body(step, controller_state, known_indices_ta, log_probability, reward):
            known_indices = tf.transpose(known_indices_ta.stack())
            unknown_indices = self.generate_unknown_indices(known_indices, episode_length)

            selected_indices, selected_log_probability = self.select(
                items,
                unknown_indices,
                known_indices,
                controller_state.h,
                cosine_similarities)

            controller_input = self.read(
                self.get_selections(items, selected_indices),
                self.get_selections(targets, selected_indices),
                state_size)

            _, controller_state = lstm(controller_input, controller_state)

            new_known_indices = tf.concat((known_indices, tf.expand_dims(selected_indices, axis=-1)), axis=1)
            new_unknown_indices = self.generate_unknown_indices(new_known_indices, episode_length)

            predictions = self.predict(
                tf.stop_gradient(items),
                new_known_indices,
                new_unknown_indices,
                tf.stop_gradient(controller_state.h),
                cosine_similarities,
                targets)

            return (
                step + 1,
                controller_state,
                known_indices_ta.write(step, selected_indices),
                log_probability + selected_log_probability,
                reward + self.calculate_rewards(predictions, tf.batch_gather(targets, new_known_indices)))

        _, controller_state, selected_indices, log_probability, reward = tf.while_loop(
            lambda step, *_: step < episode_length,
            body,
            (
                0,
                lstm.zero_state(batch_size, dtype=tf.float32),
                tf.TensorArray(dtype=tf.int32, size=episode_length, element_shape=(None,), clear_after_read=False),
                tf.zeros((batch_size,), dtype=tf.float32),
                tf.zeros((batch_size,), dtype=tf.float32),
            ),
            name='controller')

        known_indices = tf.transpose(selected_indices.stack())

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            evaluation_predictions = self.predict(
                tf.stop_gradient(items),
                tf.tile(tf.expand_dims(tf.range(episode_length, num_items), axis=0), (batch_size, 1)),
                known_indices,
                tf.stop_gradient(controller_state.h),
                cosine_similarities,
                targets)

        reward += self.calculate_rewards(evaluation_predictions, targets[:, episode_length:])

        return log_probability, reward

    @staticmethod
    def calculate_loss(log_probability: tf.Tensor, reward: tf.Tensor) -> tf.Tensor:
        """

        :param log_probability:
        :param reward:
        :return:
        """
        loss = tf.reduce_mean(log_probability * tf.stop_gradient(reward) + reward)

        return loss

    def create_optimizer(self, learning_rate: float, momentum: float):
        """

        :param learning_rate:
        :param momentum:
        :return:
        """
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(self.loss)

        return optimizer

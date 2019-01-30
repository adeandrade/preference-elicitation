import numpy as np
import tensorflow as tf

from model import PreferenceElicitationModel


class TestGaussianProcess(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        item_embeddings = np.zeros((2, 2))

        cls.model = PreferenceElicitationModel(item_embeddings)

    def test_gather_subset(self):
        items = tf.constant([
            [
                [0.1, 0.0, 0.0, 0.5, 0.7],
                [0.0, 0.1, 0.0, 0.5, 0.7],
                [0.0, 0.0, 0.1, 0.5, 0.7],
            ],
            [
                [0.0, 0.0, 0.1, 0.5, 0.7],
                [0.0, 0.1, 0.0, 0.5, 0.7],
                [0.1, 0.0, 0.0, 0.5, 0.7],
            ]], dtype=tf.float32)

        unknown_indices = tf.constant([[0, 2], [1, 2]], dtype=tf.int32)
        known_indices = tf.constant([[1], [0]], dtype=tf.int32)

        subset = self.model.gather_subset(items, unknown_indices, known_indices)

        expected_subset = tf.constant([
            [
                [0.0],
                [0.0],
            ],
            [
                [0.0],
                [0.1],
            ],
        ])

        self.assertAllEqual(subset, expected_subset)

    def test_calculate_cosine_similarities(self):
        items = tf.constant([
            [
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ],
            [
                [0.0, 0.0, 0.1],
                [0.0, 0.1, 0.0],
                [0.1, 0.0, 0.0],
            ]], dtype=tf.float32)

        cosine_similarities = self.model.calculate_cosine_similarities(items)

        expected = tf.constant([
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]], dtype=tf.float32)

        self.assertAllEqual(cosine_similarities, expected)

    def test_get_indices(self):
        mask = tf.constant(
            [
                [1, -1, 0],
                [0, 1, -1],
                [-1, 0, 1],
                [-1, 1, 0],
            ],
            dtype=tf.int32)

        indices = self.model.get_indices(mask, 1)

        self.assertAllEqual(indices, tf.constant([0, 1, 2, 1]))

    def test_generate_unknown_indices(self):
        known_indices = tf.constant(
            [
                [0, 2],
                [1, 4],
            ],
            dtype=tf.int32)

        unknown_indices = self.model.generate_unknown_indices(known_indices, 5)

        expected_indices = tf.constant(
            [
                [1, 3, 4],
                [0, 2, 3],
            ],
            dtype=tf.int32)

        self.assertAllEqual(unknown_indices, expected_indices)

    def test_selection(self):
        items = tf.constant([
            [
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ],
            [
                [0.0, 0.0, 0.1],
                [0.0, 0.1, 0.0],
                [0.1, 0.0, 0.0],
            ]], dtype=tf.float32)

        unknown_indices = tf.constant([[0, 2], [1, 2]], dtype=tf.int32)
        known_indices = tf.constant([[1], [0]], dtype=tf.int32)
        state = tf.constant([[0.50, 0.50, 0.50, 0.50], [0.25, 0.25, 0.25, 0.50]], dtype=tf.float32)

        cosine_similarities = self.model.calculate_cosine_similarities(items)
        selected_index, log_probability = self.model.select(items, unknown_indices, known_indices, state, cosine_similarities)

        self.assertAllEqual(selected_index.shape, tf.constant([2]))
        self.assertAllEqual(log_probability.shape, tf.constant([2]))

    def test_read(self):
        items = tf.constant(
            [
                [0.1, 0.0, 0.0, 5.0],
                [0.0, 0.1, 0.0, 2.0],
                [0.0, 0.0, 0.1, 3.0],
            ],
            dtype=tf.float32)

        targets = tf.constant([0, 1, -1], dtype=tf.float32)

        controller_input = self.model.read(items, targets, state_size=100)

        self.assertAllEqual(controller_input.shape, tf.constant([3, 100]))

    def test_prediction(self):
        items = tf.constant([
            [
                [0.1, 0.0, 0.0, 0.5, 0.7],
                [0.0, 0.1, 0.0, 0.5, 0.7],
                [0.0, 0.0, 0.1, 0.5, 0.7],
            ],
            [
                [0.0, 0.0, 0.1, 0.5, 0.7],
                [0.0, 0.1, 0.0, 0.5, 0.7],
                [0.1, 0.0, 0.0, 0.5, 0.7],
            ]], dtype=tf.float32)

        unknown_indices = tf.constant([[0, 2], [1, 2]], dtype=tf.int32)
        known_indices = tf.constant([[1], [0]], dtype=tf.int32)
        state = tf.constant([[0.50, 0.50, 0.50, 0.50], [0.25, 0.25, 0.25, 0.50]], dtype=tf.float32)
        targets = tf.constant(
            [
                [1, 0, -1],
                [0, -1, 1],
            ],
            dtype=tf.float32)

        cosine_similarities = self.model.calculate_cosine_similarities(items)

        predictions = self.model.predict(items, unknown_indices, known_indices, state, cosine_similarities, targets)

        self.assertAllEqual(predictions.shape, tf.constant([2, 2]))

    def test_create_controller(self):
        items = tf.constant([
            [
                [0.1, 0.0, 0.0, 0.5, 0.7],
                [0.0, 0.1, 0.0, 0.5, 0.7],
                [0.0, 0.0, 0.1, 0.5, 0.7],
            ],
            [
                [0.0, 0.0, 0.1, 0.5, 0.7],
                [0.0, 0.1, 0.0, 0.5, 0.7],
                [0.1, 0.0, 0.0, 0.5, 0.7],
            ]], dtype=tf.float32)

        targets = tf.constant(
            [
                [1, 0, -1],
                [0, -1, 1],
            ],
            dtype=tf.float32)

        log_probability, reward = self.model.create_controller(items, targets, episode_length=2)

        self.assertAllEqual(log_probability.shape, tf.constant([2]))
        self.assertAllEqual(reward.shape, tf.constant([2]))

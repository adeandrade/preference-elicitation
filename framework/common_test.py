from unittest import TestCase

import tensorflow as tf

from .common import Common


class CommonTest(TestCase):
    def test_modal_wise_dropout(self):
        modes = [tf.random_uniform([10, 5]), tf.random_uniform([10, 10]), tf.random_uniform([10, 15])]

        dropped_modes = Common.modal_wise_dropout(modes, tf.constant(0.0))

        all_zeros = [tf.cast(tf.equal(tf.reduce_sum(dropped_mode), 0.0), tf.int32) for dropped_mode in dropped_modes]

        num_all_zeros = tf.reduce_sum(tf.stack(all_zeros))

        with tf.Session() as session:
            dropped_modes = set()

            for _ in range(100):
                all_zeros_, num_all_zeros_ = session.run([all_zeros, num_all_zeros])

                self.assertEqual(num_all_zeros_, 2)

                for index, is_all_zero in enumerate(all_zeros_):
                    if is_all_zero == 1:
                        dropped_modes.add(index)

        self.assertEqual(len(dropped_modes), 3)

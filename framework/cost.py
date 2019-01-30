from enum import Enum
from typing import Mapping

import numpy as np
import tensorflow as tf

from .base_cost import BaseCost
from .common import Common


class GenericCostType(Enum):
    SQUARED_DIFFERENCE_MEAN = 1
    SQUARED_EUCLIDEAN = 2
    SOFTMAX_CROSS_ENTROPY = 3
    SIGMOID_CROSS_ENTROPY = 4
    COSINE_DISTANCE = 5


class GenericCost(BaseCost):
    def __init__(self, predictions: tf.Tensor, cost_type: GenericCostType, label: str) -> None:
        """
        Creates a placeholder for the prediction targets and a cost function according to `cost_type`.
        The name of the placeholder is the same as `predictions` with the "_targets" suffix.
        Tensors are available as properties.
        :param predictions: A 2D tensor of dimensions (batch_size, prediction_output) with the model predictions.
        :param cost_type: A CostType value specifying the cost function to use.
        :param label: The label to be used as prefix for the targets placeholder.
        """
        targets_name = f'{label}_targets'

        self._targets = tf.placeholder(predictions.dtype, predictions.shape, targets_name)

        self._cost = self.build_cost(predictions, self._targets, cost_type)

    @staticmethod
    def build_cost(predictions: tf.Tensor, targets: tf.Tensor, cost_type: GenericCostType) -> tf.Tensor:
        """
        Creates a cost-function tensor.
        Does not aggregate over the batch dimension.
        :param predictions: A 2D tensor of dimensions (batch_size, prediction_output) with the model predictions.
        :param targets: A 2D tensor of dimensions (batch_size, prediction_output) with the train/test targets.
        :param cost_type: The type of cost function to use.
        :return: A 1D tensor of dimensions (batch_size) specifying the cost for each example.
        """
        if cost_type == GenericCostType.SQUARED_EUCLIDEAN:
            return tf.reduce_sum(
                input_tensor=tf.squared_difference(predictions, targets),
                axis=-1)

        elif cost_type == GenericCostType.SQUARED_DIFFERENCE_MEAN:
            return tf.reduce_mean(
                input_tensor=tf.squared_difference(predictions, targets),
                axis=-1)

        elif cost_type == GenericCostType.COSINE_DISTANCE:
            normalized_predictions = tf.nn.l2_normalize(predictions, axis=1)
            normalized_targets = tf.nn.l2_normalize(targets, axis=1)

            cost = 1 - tf.reduce_sum(normalized_predictions * normalized_targets, axis=-1)

            return cost

        elif cost_type == GenericCostType.SOFTMAX_CROSS_ENTROPY:
            return tf.losses.softmax_cross_entropy(
                onehot_labels=targets,
                logits=predictions,
                reduction=tf.losses.Reduction.NONE)

        elif cost_type == GenericCostType.SIGMOID_CROSS_ENTROPY:
            return tf.losses.sigmoid_cross_entropy(
                multi_class_labels=targets,
                logits=predictions,
                reduction=tf.losses.Reduction.NONE)

    @property
    def targets(self) -> Mapping[str, tf.Tensor]:
        return {Common.build_tensor_name(self._targets): self._targets}

    @property
    def cost(self) -> tf.Tensor:
        return self._cost


class SparseSigmoidCost(BaseCost):
    def __init__(self, predictions: tf.Tensor, label: str) -> None:
        """
        Creates a placeholder for sparse multi-class targets and computes sigmoid entropy.
        The name of the placeholder is the same as `predictions` with the "_targets" suffix.
        Tensors are available as properties. The final cost is the mean of all sigmoid entropies.
        :param predictions: A 2D tensor of dimensions (batch_size, prediction_output) with the model predictions.
        :param label: The label to be used as prefix for the targets placeholder.
        """
        targets_name = f'{label}_targets'

        self._targets = tf.placeholder(tf.int32, [None, 2], targets_name)

        updates = tf.ones(tf.shape(self._targets)[0], dtype=tf.float32)

        targets_dense = tf.scatter_nd(self._targets, updates, tf.shape(predictions))

        costs = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=targets_dense,
            logits=predictions,
            reduction=tf.losses.Reduction.NONE)

        self._cost = tf.reduce_mean(costs, axis=1)

    @property
    def targets(self) -> Mapping[str, tf.Tensor]:
        return {Common.build_tensor_name(self._targets): self._targets}

    @property
    def cost(self) -> tf.Tensor:
        return self._cost


class SparseSoftmaxCost(BaseCost):
    def __init__(self, predictions: tf.Tensor, label: str) -> None:
        """
        Creates a placeholder for sparse class targets and computes softmax entropy.
        The name of the placeholder is the same as `predictions` with the "_targets" suffix.
        Tensors are available as properties.
        :param predictions: A 2D tensor of dimensions (batch_size, prediction_output) with the model predictions.
        :param label: The label to be used as prefix for the targets placeholder.
        """
        targets_name = f'{label}_targets'

        self._targets = tf.placeholder(tf.int32, [None], targets_name)

        self._cost = tf.losses.sparse_softmax_cross_entropy(
            labels=self._targets,
            logits=predictions,
            reduction=tf.losses.Reduction.NONE)

    @property
    def targets(self) -> Mapping[str, tf.Tensor]:
        return {Common.build_tensor_name(self._targets): self._targets}

    @property
    def cost(self) -> tf.Tensor:
        return self._cost


class WassersteinReconstructionCost(BaseCost):
    def __init__(
            self,
            original: tf.Tensor,
            reconstruction: tf.Tensor,
            compression: tf.Tensor,
            prior_sample: tf.Tensor,
            latent_dim: int,
            wae_lambda: int,
            z_num_units: int = 1024,
            z_num_layers: int = 4) -> None:
        """
        Computes a Wasserstein reconstruction loss between the original and reconstructed tensors.
        Adapted from https://github.com/tolstikhin/wae
        :param original: The original tensor to compress, where the first dimension matches batch_size.
        :param reconstruction: The tensor reconstructed from the compression.
        :param compression: The encoded tensor produced by the encoder.
        :param prior_sample: A tensor representing samples produced from the prior.
        :param latent_dim: The number of latent variables to consider.
        :param wae_lambda: Lambda for gradient penalty.
        :param z_num_units: Size of the z_adversary network
        :param z_num_layers: Number of layers of the z_adversary network
        """
        self._original = original
        self._reconstruction = reconstruction
        self._compression = compression
        self._prior_sample = prior_sample
        self._latent_dim = latent_dim
        self._wae_lambda = wae_lambda
        self._z_num_units = z_num_units
        self._z_num_layers = z_num_layers

    def reconstruction_loss(self):
        loss = tf.reduce_sum(
            tf.square(self._original - self._reconstruction), axis=range(1, tf.shape(self._original).length)
        )
        return 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))

    def gan_penalty(self):
        logits_qz = self.z_adversary(self._compression + self._prior_sample, reuse=True)
        # Non-saturating loss trick
        loss_qz_trick = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_qz, labels=tf.ones_like(logits_qz)))
        return loss_qz_trick

    def z_adversary(self, inputs, reuse=False):
        nowozin_trick = True
        # No convolutions as GAN happens in the latent space
        with tf.variable_scope('z_adversary', reuse=reuse):
            hi = inputs
            for i in range(self._z_num_layers):
                hi = self.linear(hi, self._z_num_units, scope='h%d_lin' % (i + 1))
                hi = tf.nn.relu(hi)
            hi = self.linear(hi, 1, scope='hfinal_lin')
            if nowozin_trick:
                # We are doing GAN between our model Qz and the true Pz.
                # Imagine we know analytical form of the true Pz.
                # The optimal discriminator for D_JS(Pz, Qz) is given by:
                # Dopt(x) = log dPz(x) - log dQz(x)
                # And we know exactly dPz(x). So add log dPz(x) explicitly
                # to the discriminator and let it learn only the remaining
                # dQz(x) term. This appeared in the AVB paper.
                sigma2_p = float(1)
                normsq = tf.reduce_sum(tf.square(inputs), 1)
                hi = hi - normsq / 2. / sigma2_p \
                     - 0.5 * tf.log(2. * np.pi) \
                     - 0.5 * self._latent_dim * np.log(sigma2_p)
        return hi

    @staticmethod
    def linear(input_, output_dim, scope=None, init='normal', reuse=None):
        stddev = 0.0099999
        bias_start = 0.0
        shape = input_.get_shape().as_list()

        in_shape = shape[1]
        if len(shape) > 2:
            # This means points contained in input_ have more than one
            # dimensions. In this case we first stretch them in one
            # dimensional vectors
            input_ = tf.reshape(input_, [-1, np.prod(shape[1:])])
            in_shape = np.prod(shape[1:])

        with tf.variable_scope(scope or "lin", reuse=reuse):
            if init == 'normal':
                matrix = tf.get_variable(
                    "W", [in_shape, output_dim], tf.float32,
                    tf.random_normal_initializer(stddev=stddev))
            else:
                matrix = tf.get_variable(
                    "W", [in_shape, output_dim], tf.float32,
                    tf.constant_initializer(np.identity(in_shape)))
            bias = tf.get_variable(
                "b", [output_dim],
                initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias

    @property
    def targets(self) -> Mapping[str, tf.Tensor]:
        return {Common.build_tensor_name(self._original): self._original}

    @property
    def cost(self) -> tf.Tensor:
        return self.reconstruction_loss() + self._wae_lambda * self.gan_penalty()


class NoisyBinarySigmoidCost(BaseCost):
    def __init__(self, logits: tf.Tensor, label: str, beta: float = 0.8) -> None:
        """
        Creates a placeholder for binary targets and computes sigmoid entropy.
        The name of the placeholder is the same as `predictions` with the "_targets" suffix.
        Tensors are available as properties. The final cost is the mean of all sigmoid entropies.
        :param logits: A 2D tensor of dimensions (batch_size, prediction_output) with the output logits.
        :param label: The label to be used as prefix for the targets placeholder.
        :param beta: The value to use for hard target regularization.
        """
        self._targets = tf.placeholder(tf.float32, [None, None], f'{label}_targets')

        regularized_targets = beta * self._targets + (1.0 - beta) * tf.round(tf.sigmoid(logits))

        costs = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=regularized_targets,
            logits=logits,
            reduction=tf.losses.Reduction.NONE)

        self._cost = tf.reduce_sum(costs, axis=-1)

    @property
    def targets(self) -> Mapping[str, tf.Tensor]:
        return {
            Common.build_tensor_name(self._targets): self._targets,
        }

    @property
    def cost(self) -> tf.Tensor:
        return self._cost


class NoisyUnbalancedSparseSigmoidCost(BaseCost):
    def __init__(self, logits: tf.Tensor, label: str, beta: float = 0.8) -> None:
        """
        Creates a placeholder for sparse multi-class targets and computes sigmoid entropy.
        The name of the placeholder is the same as `predictions` with the "_targets" suffix.
        Tensors are available as properties. The final cost is the mean of all sigmoid entropies.
        :param logits: A 2D tensor of dimensions (batch_size, prediction_output) with the output logits.
        :param label: The label to be used as prefix for the targets placeholder.
        :param beta:
        """
        num_tags = logits.shape[1]

        self._indices = tf.placeholder(tf.int32, [None, 2], f'{label}_indices')
        self._target_weights = tf.placeholder(tf.float32, [None, num_tags], f'{label}_target_weights')

        updates = tf.ones(tf.shape(self._indices)[0], dtype=tf.float32)

        targets_dense = tf.scatter_nd(self._indices, updates, tf.shape(logits))

        regularized_targets = beta * targets_dense + (1.0 - beta) * tf.round(tf.sigmoid(logits))

        costs = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=regularized_targets,
            logits=logits,
            weights=self._target_weights,
            reduction=tf.losses.Reduction.NONE)

        self._cost = tf.reduce_sum(costs, axis=-1)

    @property
    def targets(self) -> Mapping[str, tf.Tensor]:
        return {
            Common.build_tensor_name(self._indices): self._indices,
            Common.build_tensor_name(self._target_weights): self._target_weights
        }

    @property
    def cost(self) -> tf.Tensor:
        return self._cost

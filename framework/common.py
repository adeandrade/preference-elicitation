from typing import Sequence, Optional, Mapping, Any, Iterable, Union

import numpy as np
import tensorflow as tf

from .parameters import LayerParameters, ConvolutionParameters, DeconvolutionParameters, DistributionType, \
    RNNLayerParameters


class Common:
    @staticmethod
    def add_layers(tensor: tf.Tensor, label: str, layer_configs: Sequence[LayerParameters]) -> tf.Tensor:
        """
        Passes a tensor through a sequence of dense layers.
        :param tensor: A two-dimensional tensor of size (batch_size, features).
        :param label: The label to be used as prefix for the variables in all layers.
        :param layer_configs: A sequence of `LayerParameters` objects representing the configuration for each layer in order.
        :return: A two-dimensional tensor with the output of the last layer.
        """
        for i, config in enumerate(layer_configs):
            tensor = tf.layers.dense(
                tensor,
                units=config.output_dim,
                use_bias=config.use_bias,
                activation=config.activation_type,
                kernel_initializer=config.initializer(),
                name=f'{label}_{i}')

            if config.batch_normalization is not None:
                tensor = tf.layers.batch_normalization(tensor, training=config.batch_normalization)

            tensor = tf.nn.dropout(tensor, config.keep_prob)

        return tensor

    @staticmethod
    def apply_layers(tensor: tf.Tensor, layers: Sequence[tf.keras.Sequential]) -> tf.Tensor:
        """
        Passes a tensor through a sequence of dense layers.
        :param tensor: A two-dimensional tensor of size (batch_size, features).
        :param layers: A sequence of `Layer` objects.
        :return: A two-dimensional tensor with the output of the last layer.
        """
        for layer in layers:
            tensor = layer(tensor)

        return tensor

    @staticmethod
    def create_layers(label: str, layer_configs: Sequence[LayerParameters]) -> Sequence[tf.keras.Sequential]:
        """
        Create a sequence of dense layers given the configuration.
        :param label: The label to be used as prefix for the variables in all layers.
        :param layer_configs: A sequence of `LayerParameters` objects representing the configuration for each layer in order.
        :return: A sequence of `Layer` objects.
        """
        layers = []

        for i, config in enumerate(layer_configs):
            pipeline = [tf.layers.Dense(
                units=config.output_dim,
                use_bias=config.use_bias,
                activation=config.activation_type,
                kernel_initializer=config.initializer(),
                name=f'{label}_{i}')]

            if config.batch_normalization is not None:
                pipeline.append(tf.layers.BatchNormalization(training=config.batch_normalization))

            pipeline.append(tf.layers.Dropout(1.0 - config.keep_prob))

            layers.append(tf.keras.Sequential(pipeline))

        return layers

    @staticmethod
    def combine_losses(tensors: Iterable[tf.Tensor], weights: Optional[np.ndarray] = None) -> tf.Tensor:
        """
        Combines a sequence of cost tensors optionally using a weighted sum.
        Takes the average across the batch dimension.
        :param tensors: A sequence of `Tensor` vectors.
        :param weights: An optional float sequence specifying the weight for each cost tensor.
        :return: A vector of costs for each batch item.
        """
        costs = tf.stack(tensors, axis=-1)

        weighted_costs = costs * weights if weights is not None else costs

        return tf.reduce_mean(tf.reduce_sum(weighted_costs, axis=1))

    @staticmethod
    def build_tensor_name(tensor: tf.Tensor, *args: str) -> str:
        """
        Builds a tensor name using an existing tensor as prefix.
        Removes invalid tensor information from the name of the existing tensor.
        :param tensor: The existing tensor.
        :return: A string with a cleaned tensor name and all the provided strings concatenated by "_".
        """
        return '_'.join(tensor.name.split(':')[:1] + list(args))

    @staticmethod
    def align_tensors(tensors: Sequence[tf.Tensor], axis: int = 0) -> Sequence[tf.Tensor]:
        """
        Pads tensors in the selected dimension so they have the same size.
        Padding is only added to the end of the tensor.
        :param tensors: A collection of tensors.
        :param axis: The dimension to align for.
        :return: A sequence of padded tensors in the same order.
        """
        sizes = [tf.shape(tensor)[axis] for tensor in tensors]

        max_size = tf.reduce_max(tf.stack(sizes))

        padded_tensors = []

        for tensor, batch_size in zip(tensors, sizes):
            num_dimensions = len(tensor.shape)

            paddings = [[0, 0] for _ in range(num_dimensions)]
            paddings[axis][1] = max_size - batch_size

            padded_tensor = tf.pad(tensor, paddings)

            padded_tensors.append(padded_tensor)

        return padded_tensors

    @staticmethod
    def pad_to(tensor: tf.Tensor, size: Union[int, tf.Tensor], axis: int = 0) -> tf.Tensor:
        """
        Pads a tensor to `size`. Padding is only added to the end of the tensor.
        :param tensor: The tensor to pad.
        :param size: The tensor to align to.
        :param axis: The dimension to align for.
        :return: A padded version of `tensor`.
        """
        num_dimensions = len(tensor.shape)

        paddings = [[0, 0] for _ in range(num_dimensions)]
        paddings[axis][1] = tf.maximum(size - tf.shape(tensor)[axis], 0)

        padded_tensor = tf.pad(tensor, paddings)

        return padded_tensor

    @staticmethod
    def build_2d_depthwise_convolution_layers(
            label: str,
            cnn_input: tf.Tensor,
            config: Sequence[ConvolutionParameters]) -> tf.Tensor:
        """
        Build a series of two-dimensional convolutional depthwise separable layers with pooling.
        Loops through a sequence of convolution configurations and returns the final tensor.
        :param label: A label used as prefix for variable names.
        :param cnn_input: The input tensor to the current convolution layer.
        :param config: A sequence of `ConvolutionParameters` objects.
        :return: The final tensor after all convolution operations have been applied.
        """
        for index, params in enumerate(config):
            num_channels = cnn_input.shape[-1]

            depthwise_filter = tf.get_variable(
                f'{label}_depthwise_filter_{index}',
                shape=[params.filter_size, params.filter_size, num_channels, params.channel_multiplier],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer())

            pointwise_filter = tf.get_variable(
                f'{label}_pointwise_filter_{index}',
                shape=[1, 1, num_channels * params.channel_multiplier, params.num_filters],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer())

            biases = tf.get_variable(
                f'{label}_biases_{index}',
                shape=[1, 1, 1, params.num_filters],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())

            convolution = tf.nn.relu(tf.nn.separable_conv2d(
                cnn_input,
                depthwise_filter,
                pointwise_filter,
                strides=[1, params.conv_stride, params.conv_stride, 1],
                padding=params.conv_padding) + biases)

            if params.batch_normalization is not None:
                convolution = tf.layers.batch_normalization(convolution, training=params.batch_normalization)

            cnn_input = tf.nn.max_pool(
                convolution,
                ksize=[1, params.pool_size, params.pool_size, 1],
                strides=[1, params.pool_stride, params.pool_stride, 1],
                padding='VALID')

        return cnn_input

    @staticmethod
    def build_2d_convolution_layers(
            label: str,
            cnn_input: tf.Tensor,
            config: Sequence[ConvolutionParameters]) -> tf.Tensor:
        """
        Build a series of two-dimensional convolutional layers with pooling.
        Loops through a sequence of convolution configurations and returns the final tensor.
        :param label: A label used as prefix for variable names.
        :param cnn_input: The input tensor to the current convolution layer.
        :param config: A sequence of `ConvolutionParameters` objects.
        :return: The final tensor after all convolution operations have been applied.
        """
        for index, params in enumerate(config):
            convolution = tf.layers.conv2d(
                cnn_input,
                params.num_filters,
                params.filter_size,
                params.conv_stride,
                params.conv_padding,
                activation=tf.nn.relu,
                name=f'{label}_convolution_{index}')

            if params.batch_normalization is not None:
                convolution = tf.layers.batch_normalization(convolution, training=params.batch_normalization)

            cnn_input = tf.layers.max_pooling2d(
                convolution,
                pool_size=params.pool_size,
                strides=params.pool_stride)

        return cnn_input

    @staticmethod
    def build_2d_deconvolution_layers(
            label: str,
            cnn_input: tf.Tensor,
            config: Sequence[DeconvolutionParameters]) -> tf.Tensor:
        """
        Build a series of two-dimensional convolutional depthwise separable layers with upsample.
        Loops through a sequence of convolution configurations and returns the final tensor.
        :param label: A label used as prefix for variable names.
        :param cnn_input: The input tensor to the current convolution layer.
        :param config: A sequence of `ConvolutionParameters` objects.
        :return: The final tensor after all convolution operations have been applied.
        """
        for index, params in enumerate(config):
            num_channels = cnn_input.shape[-1]

            upsample = tf.image.resize_images(
                cnn_input,
                size=params.upsample_shape,
                method=params.resize_method)

            depthwise_filter = tf.get_variable(
                f'{label}_depthwise_filter_{index}',
                shape=[params.filter_size, params.filter_size, num_channels, params.channel_multiplier],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer())

            pointwise_filter = tf.get_variable(
                f'{label}_pointwise_filter_{index}',
                shape=[1, 1, num_channels * params.channel_multiplier, params.num_filters],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer())

            biases = tf.get_variable(
                f'{label}_biases_{index}',
                shape=[1, 1, 1, params.num_filters],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())

            cnn_input = tf.nn.relu(tf.nn.separable_conv2d(
                upsample,
                depthwise_filter,
                pointwise_filter,
                strides=[1, params.conv_stride, params.conv_stride, 1],
                padding=params.conv_padding) + biases)

        return cnn_input

    @staticmethod
    def build_1d_convolution_layers(
            label: str,
            cnn_input: tf.Tensor,
            config: Sequence[ConvolutionParameters]) -> tf.Tensor:
        """
        Build a series of one-dimensional convolutional depthwise separable layers with pooling.
        Loops through a sequence of convolution configurations and returns the final tensor.
        :param label: A label used as prefix for variable names.
        :param cnn_input: The input tensor to the current convolution layer.
        :param config: A sequence of `ConvolutionParameters` objects.
        :return: The final tensor after all convolution operations have been applied.
        """
        for index, params in enumerate(config):
            num_channels = cnn_input.shape[-1]

            depthwise_filter = tf.get_variable(
                f'{label}_depthwise_filter_{index}',
                shape=[1, params.filter_size, num_channels, params.channel_multiplier],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer())

            pointwise_filter = tf.get_variable(
                f'{label}_pointwise_filter_{index}',
                shape=[1, 1, num_channels * params.channel_multiplier, params.num_filters],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer())

            biases = tf.get_variable(
                f'{label}_biases_{index}',
                shape=[1, 1, 1, params.num_filters],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())

            convolution = tf.nn.relu(tf.nn.separable_conv2d(
                tf.expand_dims(cnn_input, 1),
                depthwise_filter,
                pointwise_filter,
                strides=[1, params.conv_stride, params.conv_stride, 1],
                padding=params.conv_padding) + biases)

            if params.batch_normalization is not None:
                convolution = tf.layers.batch_normalization(convolution, training=params.batch_normalization)

            pooling = tf.nn.max_pool(
                convolution,
                ksize=[1, 1, params.pool_size, 1],
                strides=[1, 1, params.pool_stride, 1],
                padding='VALID')

            cnn_input = tf.squeeze(pooling, axis=1)

        return cnn_input

    @staticmethod
    def modal_wise_dropout(tensors: Sequence[tf.Tensor], keep_prob: float) -> Sequence[tf.Tensor]:
        """
        Performs dropout on one entire mode with `keep_prob`, selected uniformly.
        :param tensors: A sequence of tensors.
        :param keep_prob: The probability with which any tensor is dropped.
        :return: A sequence of tensors with one of them potentially zeros.
        """
        if len(tensors) == 1:
            return tensors

        num_modes = len(tensors)

        def keep():
            return tensors

        def drop():
            drop_idx = tf.random_uniform(shape=(), maxval=num_modes, dtype=tf.int32)

            drop_idx_mask = [
                tensors[index] * tf.cast(tf.equal(tf.convert_to_tensor(index), drop_idx), tf.float32) / num_modes
                for index in range(num_modes)]

            return drop_idx_mask

        is_dropout = tf.greater(tf.random_uniform(shape=()), keep_prob)

        return tf.cond(is_dropout, drop, keep)

    @staticmethod
    def sample_noise(num: int, zdim: int, distribution: DistributionType,
                     options: Mapping[str, Any], scale: Optional[float] = 1.0) -> np.array:
        """
        Sampling function, sample of uniform, multivariate (optional spherical)
        :param num: total number of samples
        :param zdim:  number of multivariate distributions
        :param distribution: distribution type
        :param scale: Optional, scalling parameter
        :return: 
        """
        noise = None
        if distribution == DistributionType.UNIFORM:
            noise = np.random.uniform(options["minval"], options["maxval"], (num, zdim)).astype(np.float32)
        elif distribution in (DistributionType.MULTIVARIATE_NORMAL, DistributionType.SPHERE):
            # mean = np.zeros(zdim)
            # cov = np.identity(zdim)
            noise = np.random.multivariate_normal(options["mean"], options["cov"], num).astype(np.float32)
            if distribution == DistributionType.SPHERE:
                noise = noise / np.sqrt(np.sum(noise * noise, axis=1))[:, np.newaxis]
        elif distribution == DistributionType.NORMAL:
            noise = np.random.normal(loc=options["mean"], scale=options["stddev"])
        return scale * noise

    @staticmethod
    def tf_sample_noise(num: int, zdim: int, distribution: DistributionType,
                        options: Mapping[str, Any], scale: Optional[float] = 1.0) -> tf.Tensor:
        """
        Sampling function, sample of uniform, multivariate (optional spherical)
        :param num: total number of samples
        :param zdim:  number of multivariate distributions
        :param distribution: distribution type
        :param scale: Optional, scaling parameter
        :return: 
        """
        noise = None
        if distribution == DistributionType.UNIFORM:
            noise = tf.random_uniform(shape=(num, zdim), minval=options["minval"], maxval=options["maxval"])
        elif distribution in (DistributionType.MULTIVARIATE_NORMAL, DistributionType.SPHERE):
            # mean = tf.zeros((zdim))
            # cov = tf.eye(zdim)
            noise = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=options["mean"],
                                                                              covariance_matrix=options["cov"]).sample(
                num)
            if distribution == DistributionType.SPHERE:
                noise = noise / tf.sqrt(tf.reduce_sum(noise * noise, axis=1))
        elif distribution == DistributionType.NORMAL:
            noise = tf.random_normal(shape=(num, zdim), mean=options["mean"],
                                     stddev=options["stddev"])
        return scale * noise

    @staticmethod
    def add_standard_normal_noise(data_mean: tf.Tensor, data_sigmas: tf.Tensor) -> tf.Tensor:
        """
        Add standard normal noise to data sample
        :param data_mean: the data tensor to add noise on
        :param data_sigmas: the amount of variances add to data_mean, noise will apply to this tensor
        :return: 
        """
        shapes = data_sigmas.get_shape().as_list()
        esp = Common.tf_sample_noise(shapes[0], shapes[1], DistributionType.NORMAL,
                                     options={"mean": 0, "stddev": 1})
        return data_mean + tf.multiply(esp, tf.sqrt(1e-8 + tf.exp(data_sigmas)))

    @staticmethod
    def add_static_rnn(tensor: tf.Tensor,
                       layer_configs: Sequence[RNNLayerParameters]):
        """
        Light wrapper to add static RNN based on config specified in RNNLayerParameters
        :param tensor: 
        :param layer_configs: 
        :return: 
        """
        inputs = tensor
        for i, config in enumerate(layer_configs):
            cell = config.rnn_cell
            if config.bidirection:
                initial_state_fw = cell[0].zero_state(tensor.get_shape().as_list()[0], dtype=tf.float32)
                initial_state_bw = cell[1].zero_state(tensor.get_shape().as_list()[0], dtype=tf.float32)

                outputs, states = tf.nn.static_bidirectional_rnn(cell_fw=cell[0], cell_bw=cell[1],
                                                                 inputs=inputs, initial_state_fw=initial_state_fw,
                                                                 initial_state_bw=initial_state_bw)


            else:
                initial_state = cell.zero_state(tensor.get_shape().as_list()[0], dtype=tf.float32)
                outputs, states = tf.nn.static_rnn(cell, inputs, initial_state=initial_state)
            inputs = tf.concat(outputs, 2)

        return inputs

    @staticmethod
    def add_dynamic_rnn(tensor: tf.Tensor, sequence_lengths: tf.Tensor,
                        layer_configs: Sequence[RNNLayerParameters]):
        """
        Light wrapper to add dynamic RNN based on config specified in RNNLayerParameters
        :param tensor: 
        :param sequence_lengths: 
        :param layer_configs: 
        :return: 
        """
        inputs = tensor
        for i, config in enumerate(layer_configs):
            cell = config.rnn_cell
            if config.bidirection:
                initial_state_fw = cell[0].zero_state(tensor.shape[0], dtype=tf.float32)
                initial_state_bw = cell[1].zero_state(tensor.shape[0], dtype=tf.float32)

                outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell[0], cell_bw=cell[1],
                                                                  sequence_length=sequence_lengths,
                                                                  inputs=inputs, initial_state_fw=initial_state_fw,
                                                                  initial_state_bw=initial_state_bw)
            else:
                initial_state = cell.zero_state(tensor.shape[0], dtype=tf.float32)
                outputs, states = tf.nn.dynamic_rnn(cell, inputs, sequence_length=sequence_lengths,
                                                    initial_state=initial_state)
            inputs = tf.concat(outputs, axis=2)

        return inputs

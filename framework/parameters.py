from enum import Enum
from typing import Callable, Tuple, Union, Optional

import tensorflow as tf


class ActivationType(Enum):
    RELU = 1
    SIGMOID = 2
    TANH = 3
    SOFTMAX = 4
    LINEAR = 5


class InitializerType(Enum):
    GLOROT_UNIFORM = 1
    GLOROT_NORMAL = 2
    RANDOM_UNIFORM = 3
    RANDOM_NORMAL = 4
    ZEROS = 5


class DistributionType(Enum):
    UNIFORM = 1
    MULTIVARIATE_NORMAL = 2
    SPHERE = 3
    NORMAL = 4


class RNNCellType(Enum):
    BASE_RNN = 1
    GRU = 2
    LSTM = 3


class ModalType(Enum):
    TEXT = 1
    COVER = 2
    INTERACTIONS = 3


class RNNType(Enum):
    STATIC = 1
    BI_STATIC = 2
    DYNAMIC = 3
    BI_DYNAMIC = 4


class ConvolutionParameters:
    def __init__(
            self,
            filter_size: int,
            num_filters: int,
            channel_multiplier: int,
            conv_stride: int,
            conv_padding: str,
            pool_size: int,
            pool_stride: int,
            batch_normalization: Optional[Union[bool, tf.Tensor]] = None):
        """
        Parameters for a standard convolution layer with pooling.
        :param filter_size: A squared filter size.
        :param num_filters: The number of feature detectors to use.
        :param channel_multiplier: The number of channels for the output of a depthwise convolution.
        :param conv_stride: The square stride of of the convolution.
        :param conv_padding: The type of Tensorflow padding to be applied ('VALID' or 'SAME').
        :param pool_size: The size of the pooled region.
        :param pool_stride: The square stride of of the pooling operation.
        :param batch_normalization: An optional boolean value. Enables batch normalization if present and expresses if
        the operation is in training mode or not.
        """
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.channel_multiplier = channel_multiplier
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.batch_normalization = batch_normalization


class DeconvolutionParameters:
    def __init__(
            self,
            filter_size: int,
            num_filters: int,
            channel_multiplier: int,
            conv_stride: int,
            conv_padding: str,
            upsample_shape: Tuple[int, int],
            resize_method: tf.image.ResizeMethod):
        """
        Parameters for a deconvolution layer with upsampling and a transposed convolution.
        :param filter_size: A squared filter size.
        :param num_filters: The number of feature detectors to use.
        :param channel_multiplier: The number of channels for the output of a depthwise convolution.
        :param conv_stride: The square stride of of the convolution.
        :param conv_padding: The type of Tensorflow padding to be applied ('VALID' or 'SAME').
        :param upsample_shape: Upsample shape.
        :param resize_method: The algorithm to use for resizing images.
        """
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.channel_multiplier = channel_multiplier
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.upsample_shape = upsample_shape
        self.resize_method = resize_method


class LayerParameters:
    activation_map = {
        ActivationType.RELU: tf.nn.relu,
        ActivationType.SIGMOID: tf.nn.sigmoid,
        ActivationType.TANH: tf.nn.tanh,
        ActivationType.SOFTMAX: tf.nn.softmax,
        ActivationType.LINEAR: lambda x: x,
    }

    initializer_map = {
        InitializerType.GLOROT_UNIFORM: tf.glorot_uniform_initializer,
        InitializerType.GLOROT_NORMAL: tf.glorot_normal_initializer,
        InitializerType.RANDOM_UNIFORM: tf.random_uniform_initializer,
        InitializerType.RANDOM_NORMAL: tf.random_normal_initializer,
        InitializerType.ZEROS: tf.zeros_initializer
    }

    def __init__(
            self,
            output_dim: int,
            use_bias: bool = True,
            activation_type: ActivationType = ActivationType.RELU,
            initializer_type: InitializerType = InitializerType.GLOROT_UNIFORM,
            keep_prob: Union[float, tf.Tensor] = 1.0,
            batch_normalization: Optional[Union[bool, tf.Tensor]] = None):
        """
        Parameters for a standard dense layer.
        :param output_dim: The number of dimensions of the output layer.
        :param use_bias: Whether to use bias units or not.
        :param activation_type: A `ActivationType` value specifying the type of activation function.
        :param initializer_type: A `InitializerType` value specifying the type of initializer for the weights.
        :param keep_prob: The dropout rate for the layer, default is 1.0 (no dropout).
        :param batch_normalization: An optional boolean value. Enables batch normalization if present and expresses if
        the operation is in training mode or not.
        """
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.activation_type = self.get_activation(activation_type)
        self.initializer = self.get_initializer(initializer_type)
        self.keep_prob = keep_prob
        self.batch_normalization = batch_normalization

    @classmethod
    def get_activation(cls, activation_type: ActivationType) -> Callable[[tf.Tensor], tf.Tensor]:
        """
        Get the activation function for a specified type.
        :param activation_type: A `ActivationType` value specifying the type of activation function.
        :return: A function that transforms a tensor into another tensor.
        """
        activation_function = cls.activation_map.get(activation_type)

        if activation_function is None:
            existing_keys = ' '.join(key.name for key in cls.activation_map.keys())

            raise Exception(f"""
            Activation type: {activation_type} - Does not exist!
            Please add to the class definition.
            Current support: {existing_keys}""")

        return activation_function

    @classmethod
    def get_initializer(cls, initializer_type: InitializerType) -> tf.keras.initializers.Initializer:
        """
        Get the initialization function for a specified type.
        :param initializer_type: A `InitializerType` value specifying the type of initializer for the weights.
        :return: A function that transforms a tensor into another tensor.
        """
        initializer_function = cls.initializer_map.get(initializer_type)

        if initializer_function is None:
            existing_keys = ' '.join(key.name for key in cls.initializer_map.keys())

            raise Exception(f"""
                Activation type: {initializer_type} - Does not exist!
                Please add to the class definition.
                Current support: {existing_keys}""")

        return initializer_function


class RNNLayerParameters(LayerParameters):
    rnn_cell_map = {
        RNNCellType.BASE_RNN: tf.nn.rnn_cell.BasicRNNCell,
        RNNCellType.GRU: tf.nn.rnn_cell.GRUCell,
        RNNCellType.LSTM: tf.nn.rnn_cell.LSTMCell
    }

    def __init__(self,
                 num_units: int,
                 rnn_cell_type: RNNCellType = RNNCellType.BASE_RNN,
                 activation_type: ActivationType = ActivationType.TANH,
                 initializer_type: InitializerType = InitializerType.GLOROT_UNIFORM,
                 use_peepholes: bool = False,
                 forget_bias: float = 1.0,
                 bidirection: bool = False,
                 name: str = "rnn"
                 ):
        """
        Parameters for RNN layer. The RNN cell tensor will be created within this function
        :param num_units: number of hidden units in RNN
        :param rnn_cell_type: A `RNNCellType` value specifying the type of RNN cell type function.
        :param activation_type: A `ActivationType` value specifying the type of activation function.
        :param initializer_type: A `InitializerType` value specifying the type of initializer for the weights.
        :param use_peepholes: default False
        :param forget_bias: default 1.0
        :param bidirection: default False
        :param name: 
        """

        self.rnn_cell_type = rnn_cell_type
        self.num_units = num_units
        self.activation_type = self.get_activation(activation_type)
        self.initializer = self.get_initializer(initializer_type)
        self.use_peepholes = use_peepholes
        self.forget_bias = forget_bias
        self.bidirection = bidirection

        if self.bidirection:
            self.rnn_cell = (self.get_rnn_cell(name + "_fw_cell"),
                             self.get_rnn_cell(name + "_bw_cell"))
        else:
            self.rnn_cell = self.get_rnn_cell(name + "_rnn_cell")

    def get_rnn_cell(self, name: str) -> tf.nn.rnn_cell.RNNCell:
        """
        Create RNN cell based on type and config
        :param rnn_cell_type: 
        :param name: 
        :return: 
        """
        cell_function = self.get_rnn_cell_func(self.rnn_cell_type)

        if isinstance(cell_function, tf.nn.rnn_cell.GRUCell):
            return cell_function(num_units=self.num_units, activation=self.activation_type,
                                 kernel_initializer=self.initializer, name=name)
        elif isinstance(cell_function, tf.nn.rnn_cell.LSTMCell):
            return cell_function(num_units=self.num_units, activation=self.activation_type,
                                 use_peepholes=self.use_peepholes, forget_bias=self.forget_bias, name=name)

        return cell_function(num_units=self.num_units, activation=self.activation_type, name=name)

    @classmethod
    def get_rnn_cell_func(cls, rnn_cell_type: RNNCellType) -> tf.keras.initializers.Initializer:
        """
        Get the RNN cell function for a specified type.
        :param rnn_cell_type: A `RNNCellType` value specifying the type of RNN cell.
        :return: A function that transforms a tensor into another tensor.
        """
        cell_function = cls.rnn_cell_map.get(rnn_cell_type)

        if cell_function is None:
            existing_keys = ' '.join(key.name for key in cls.rnn_cell_map.keys())

            raise Exception(f"""
                    Activation type: {rnn_cell_type} - Does not exist!
                    Please add to the class definition.
                    Current support: {existing_keys}""")

        return cell_function

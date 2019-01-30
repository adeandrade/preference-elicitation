from typing import Sequence, Mapping, Optional, Iterable

import numpy as np
import tensorflow as tf

from .base_coders import Decoder, Encoder
from .common import Common
from .base_cost import BaseCost
from .parameters import LayerParameters


class CombinerEncoder(Encoder):
    def __init__(self, encoders: Sequence[Encoder], configs: Sequence[LayerParameters], label: str = 'combiner'):
        """
        Combines a sequence of encoder representations and runs them through a sequence of dense layers.
        :param encoders: A sequence of `Encoder` objects.
        :param configs: A sequence of `LayerParameters` objects specifying the configuration for each dense layer.
        :param label: The label to use as prefix for dense layer variables.
        :return: A tensor representing the final embedding.
        """
        self._encoders = encoders

        self._original = tf.concat([encoder.representation for encoder in encoders], axis=1)

        self._representation = Common.add_layers(self._original, label, configs)

    @property
    def inputs(self) -> Mapping[str, tf.Tensor]:
        return {key: value for encoder in self._encoders for key, value in encoder.inputs.items()}

    @property
    def original(self) -> tf.Tensor:
        return self._original

    @property
    def representation(self) -> tf.Tensor:
        return self._representation


class CombinerDecoder(Decoder):
    def __init__(self, encoding: tf.Tensor, configs: Sequence[LayerParameters], label: str = "decoder"):
        """
        Decodes the passed embedding back into its original concatenated representation.
        :param encoding: The encoded tensor to be decoded.
        :param configs: A sequence of `LayerParameters` objects specifying the configuration for each dense layer.
        """
        self._original = encoding
        self._representation = Common.add_layers(self._original, label, configs)

    @property
    def outputs(self) -> Mapping[str, tf.Tensor]:
        return {"decoded": self._representation}


class CombinerCost(BaseCost):
    def __init__(self, costs: Iterable[BaseCost], weights: Optional[np.ndarray] = None):
        """
        Combines costs using optional weights.
        :param costs: A sequence of `BaseCost`s.
        :param weights: A vector of weights for each cost.
        """
        self._targets = {name: targets for cost in costs for name, targets in cost.targets.items()}

        self._cost = Common.combine_losses([cost.cost for cost in costs], weights)

    @property
    def targets(self) -> Mapping[str, tf.Tensor]:
        return self._targets

    @property
    def cost(self) -> tf.Tensor:
        return self._cost

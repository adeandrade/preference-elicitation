from typing import Mapping

import tensorflow as tf


class Encoder:
    @property
    def inputs(self) -> Mapping[str, tf.Tensor]:
        """
        Returns the inputs of the encoder.
        :return: A map from input label to tensor representations.
        """
        raise NotImplementedError

    @property
    def representation(self) -> tf.Tensor:
        """
        Returns the final encoded representations.
        :return: A tensor object.
        """
        raise NotImplementedError


class Decoder:
    @property
    def outputs(self) -> Mapping[str, tf.Tensor]:
        """
        Returns the predictions of the decoder.
        :return: A map from target labels to tensor representations.
        """
        raise NotImplementedError

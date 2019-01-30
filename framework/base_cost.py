from typing import Mapping

import tensorflow as tf


class BaseCost:
    @property
    def targets(self) -> Mapping[str, tf.Tensor]:
        raise NotImplementedError

    @property
    def cost(self) -> tf.Tensor:
        raise NotImplementedError

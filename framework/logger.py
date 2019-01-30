from typing import Mapping, MutableMapping, Sequence, Union, Tuple, List, Optional

import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf


class TensorBoardLogger:
    def __init__(self, session: tf.Session, path: str, task_labels: Optional[Sequence[str]] = None) -> None:
        """
        A logger class that stores training steps data and flushes desired stats to a TensorBoard event file.
        :param session: A TensorFlow session object.
        :param path: The path to store the TensorBoard event file.
        :param task_labels: An optional list of task labels to keep track of.
        """
        self._task_labels = task_labels if task_labels else []

        self._summary_writer = tf.summary.FileWriter(path, session.graph)

        self._step = 0

        self._training_costs = []  # type: List[float]

        self._validation_costs = []  # type: List[float]
        self._validation_cost_terms = {}  # type: MutableMapping[str, List[np.ndarray]]
        self._predictions = {task_label: [] for task_label in self._task_labels}  # type: MutableMapping[str, List[np.ndarray]]
        self._targets = {task_label: [] for task_label in self._task_labels}  # type: MutableMapping[str, List[np.ndarray]]

    def __enter__(self) -> 'TensorBoardLogger':
        return self

    def __exit__(self, type, value, traceback):
        self._summary_writer.close()

    def add_training_metadata(self, cost: float) -> None:
        """
        Adds training metadata to this logger's state.
        :param cost: The value of the training cost function.
        """
        self._training_costs.append(cost)

    def add_validation_metadata(
            self,
            cost: float,
            cost_terms: Optional[Mapping[str, np.ndarray]] = None,
            predictions: Optional[Mapping[str, np.ndarray]] = None,
            targets: Optional[Mapping[str, np.ndarray]] = None) -> None:
        """
        Adds validation metadata to this logger's state.
        The keys for `predictions` and `targets` must match and must be provided in the constructor of this object.
        The values of the maps in `cost_terms`, `predictions`, and `targets` are expected to be numpy vectors.
        :param cost: The value of the validation cost function.
        :param cost_terms: An optional map from cost term labels to their values, in case the cost function has several terms.
        :param predictions: An optional map from prediction labels to their values.
        :param targets: An optional map from true target labels to their values.
        """
        self._validation_costs.append(cost)

        if cost_terms:
            for cost_label, costs in cost_terms.items():
                if cost_label in self._validation_cost_terms:
                    self._validation_cost_terms[cost_label].append(costs)
                else:
                    self._validation_cost_terms[cost_label] = [costs]

        if predictions:
            for task_label in self._task_labels:
                self._predictions[task_label].append(predictions[task_label])

        if targets:
            for task_label in self._task_labels:
                self._targets[task_label].append(targets[task_label])

    def get_training_cost(self) -> float:
        """
        Calculates the mean of all submitted and available training costs before flushing.
        :return: The mean training cost.
        """
        return TensorBoardLogger.calculate_mean(self._training_costs)

    def get_validation_cost(self) -> float:
        """
        Calculates the mean of all submitted and available validation costs before flushing.
        :return: The mean validation cost.
        """
        return TensorBoardLogger.calculate_mean(self._validation_costs)

    def get_summary(self) -> Mapping[str, float]:
        """
        Computes a summary of stats:
            - Mean of the training costs.
            - Mean of the validation costs.
            - Mean of each cost term.
            - Precision/Recall/F1 scores for each task.
        :return: A map from statistic label to scalar value.
        """
        costs = {'training_cost': self.get_training_cost(), 'validation_cost': self.get_validation_cost()}

        cost_terms = {
            label: TensorBoardLogger.calculate_mean(np.concatenate(arrays))
            for label, arrays in self._validation_cost_terms.items()}

        type_error_scores = {}

        for task_label in self._task_labels:
            precision, recall, f1 = TensorBoardLogger.calculate_type_error_scores(
                np.concatenate(self._predictions[task_label]),
                np.concatenate(self._targets[task_label]))

            type_error_scores[f'{task_label}_precision'] = precision
            type_error_scores[f'{task_label}_recall'] = recall
            type_error_scores[f'{task_label}_f1'] = f1

        return {**costs, **cost_terms, **type_error_scores}

    def clear(self) -> None:
        """
        Clears all state sent to this object.
        """
        self._step += 1

        self._training_costs = []

        self._validation_costs = []
        self._validation_cost_terms = {}
        self._predictions = {task_label: [] for task_label in self._task_labels}
        self._targets = {task_label: [] for task_label in self._task_labels}

    def flush(self) -> None:
        """
        Computes a summary of the existing state, writes it to the TensorBoard event file as one step, and clears the state.
        """
        summary = self.get_summary()

        for label, value in summary.items():
            summary = tf.Summary()
            summary.value.add(tag=label, simple_value=value)

            self._summary_writer.add_summary(summary, self._step)

        self._summary_writer.flush()

        self.clear()

    @staticmethod
    def calculate_mean(array: Union[Sequence[Union[int, float]], np.ndarray]) -> float:
        """
        Calculates the mean of a vector.
        :param array: A sequence of numbers or a Numpy vector.
        :return: The mean as a scalar.
        """
        return float(np.mean(array))

    @staticmethod
    def calculate_type_error_scores(targets: np.ndarray, predictions: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates the precision, recall and f1 scores given a vectors of targets and predictions.
        :param targets: A vector of true targets where values are considered class indices.
        :param predictions: A vector of predictions where values are considered class indices.
        :return: A tuple with precision, recall and f1 scores.
        """
        precision = metrics.precision_score(targets, predictions, average='weighted')

        recall = metrics.recall_score(targets, predictions, average='weighted')

        f1 = metrics.f1_score(targets, predictions, average='weighted')

        return precision, recall, f1

    @staticmethod
    def quantize(array: np.ndarray, maximum: int, minimum: int = 0) -> np.ndarray:
        """
        Quantizes values in an array by rounding them and applying minimums and maximums.
        :param array: A Numpy vector.
        :param maximum: Minimum value allowed.
        :param minimum: Maximum value allowed.
        :return: A Numpy vector of integers representing quantizations of the given vector.
        """
        rounded = np.round(array, decimals=0).astype(np.int32)

        capped = np.minimum(np.maximum(rounded, maximum), minimum)

        return capped

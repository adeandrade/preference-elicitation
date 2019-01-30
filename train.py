import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from dataset import UserInteractionDataset, load_dataset, load_item_embeddings
from framework.logger import TensorBoardLogger
from model import PreferenceElicitationModel


def create_feed_map(
        batch_item_indices: np.ndarray,
        batch_targets: np.ndarray,
        model: PreferenceElicitationModel):
    """

    :param batch_item_indices:
    :param batch_targets:
    :param model:
    :return:
    """
    return {
        model.item_indices: batch_item_indices,
        model.targets: batch_targets,
    }


def train(
        model: PreferenceElicitationModel,
        dataset: UserInteractionDataset,
        path: str,
        batch_size: int = 64,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        momentum: float = 0.99,
        early_stopping_threshold: int = 30) -> None:
    """

    :param model:
    :param dataset:
    :param path:
    :param batch_size:
    :param num_epochs:
    :param learning_rate:
    :param momentum:
    :param early_stopping_threshold:
    :return:
    """
    optimizer = model.create_optimizer(learning_rate, momentum)

    saver = tf.train.Saver()

    with tf.Session() as session, TensorBoardLogger(session, path) as logger:
        session.run(tf.global_variables_initializer())

        min_validation_loss, non_improvement_times = np.inf, 0

        for epoch in range(num_epochs):
            for inputs, targets in dataset.get_train_batches(batch_size):
                loss_, _ = session.run([model.loss, optimizer], feed_dict=create_feed_map(inputs, targets, model))

                print(f'\rEpoch: {epoch}, loss: {loss_}', sep=' ', end='')
                sys.stdout.flush()

                logger.add_training_metadata(loss_)

            for inputs, targets in dataset.get_validation_batches(batch_size):
                loss_ = session.run(model.loss, feed_dict=create_feed_map(inputs, targets, model))

                logger.add_validation_metadata(loss_)

            validation_loss = logger.get_validation_cost()

            logger.flush()

            print(f'\nValidation set epoch: {epoch}, loss: {validation_loss}')

            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                non_improvement_times = 0

                saver.save(session, os.path.join(path, 'preference_elicitation.ckpt'))

            elif non_improvement_times < early_stopping_threshold:
                non_improvement_times += 1

            else:
                print('Stopping after no improvement.')
                return


def main(path: str, train_ratio: float = 0.9):
    """

    :param path:
    :param train_ratio:
    :return:
    """
    dataset = load_dataset(path, train_ratio)

    item_embeddings = load_item_embeddings(path, dataset.num_items)

    model = PreferenceElicitationModel(item_embeddings, episode_length=dataset.num_items_per_user)

    train(model, dataset, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str)

    args = parser.parse_args()

    main(args.path)

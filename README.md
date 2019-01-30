# Preference Elicitation Through Active Learning and Meta-Learning

This is a work in progress TensorFlow implementation of:

Bachman, P., Sordoni, A., and Trischler, A. Learning algorithms for active learning. In Precup, D. and Teh,
Y. W. (eds.), Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 301–310, International Convention Centre, Sydney, Australia, 06–
11 Aug 2017. PMLR. URL: http://proceedings.mlr.press/v70/bachman17a.html

It ignores the BiLSTM contextual encodings and uses the fast-predictor to compute the rewards of the held-out evaluation set.

It is currently missing the Generalized Advantage Estimation component. Instead it optimizes the policy gradients with just a non-discounted sum of all rewards.

The `UserInteractionDataset` class in `dataset.py` creates batches of interactions between users and items, suitable for user preference elicitation with an exploration-exploitation trade-off.
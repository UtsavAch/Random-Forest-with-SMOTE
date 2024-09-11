# coding:utf-8
import numpy as np

from base import BaseEstimator
from base_tree import information_gain, mse_criterion
from tree import Tree
import SMOTE
import torch
import pandas as pd


class RandomForest(BaseEstimator):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=None, criterion=None, smote=None, smote_type=None):
        """Base class for RandomForest.

        Parameters
        ----------
        n_estimators : int
            The number of decision tree.
        max_features : int
            The number of features to consider when looking for the best split.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        max_depth : int
            Maximum depth of the tree.
        criterion : str
            The function to measure the quality of a split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []
        self.smote=smote
        self.smote_type=smote_type

    def fit(self, X, y):
        self._setup_input(X, y)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            assert X.shape[1] > self.max_features
        self._train()

    def _train(self):
        if(self.smote_type=="binary"):
            percentages=[i%2 for i in range(self.n_estimators)]
            count=0
        elif(self.smote_type=="gradient" and self.n_estimators==10):
            percentages=[0,0.2,0.2,0.4,0.4,0.6,0.6,0.8,0.8,1]
        elif(self.smote_type=="gradient" and self.n_estimators!=10):
            print("The number of estimator is not compatible with the smote type gradient, for this type of smote the number of estimators must be 10")
        count=0
        for tree in self.trees:
            if self.smote is not None:
                X=self.X
                y=self.y
                #print(type(self.X))
                X= torch.tensor(X)
                y= torch.tensor(y)
                X, y = self.smote.fit_generate(X, y, percentage=percentages[count])
                X=X.numpy()
                y=y.numpy()

                print(f"dataset apos smote:{X.shape}")
                tree.train(
                X,
                y,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )
                count+=1
            else:
                tree.train(
                self.X,
                self.y,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )

    def _predict(self, X=None):
        raise NotImplementedError()


class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=None, criterion="entropy", smote = None, smote_type=None):
        super(RandomForestClassifier, self).__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            criterion=criterion,
        )

        if criterion == "entropy":
            self.criterion = information_gain
        else:
            raise ValueError()

        self.smote = smote
        self.smote_type=smote_type
        # Initialize empty trees
        for _ in range(self.n_estimators):
            self.trees.append(Tree(criterion=self.criterion))

    def _predict(self, X=None):
        y_shape = np.unique(self.y).shape[0]
        predictions = np.zeros((X.shape[0], y_shape))

        for i in range(X.shape[0]):
            row_pred = np.zeros(y_shape)
            for tree in self.trees:
                row_pred += tree.predict_row(X[i, :])

            row_pred /= self.n_estimators
            predictions[i, :] = row_pred
        return predictions

    def fit(self, X, y):

        super(RandomForestClassifier, self).fit(X, y)

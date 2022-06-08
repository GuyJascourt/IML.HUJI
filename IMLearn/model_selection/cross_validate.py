from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator

def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    trainScore = 0
    validationScore = 0
    XFolded = np.array_split(X,cv)
    YFolded = np.array_split(y,cv)
    for i in range(cv):
        trainX = np.concatenate(XFolded[:i] + XFolded[i+1:])
        trainY = np.concatenate(YFolded[:i] + YFolded[i+1:])
        validateX = XFolded[i]
        validateY = YFolded[i]
        estimator.fit(trainX,trainY)
        trainScore += scoring(trainY, estimator.predict(trainX))
        validationScore += scoring(validateY, estimator.predict(validateX))
    return trainScore / cv, validationScore / cv

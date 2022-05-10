from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True) # Stated in forum we may assume no classes are missing
        self.pi_ = counts / y.shape[0]
        self.mu_ = np.array([X[y==i].mean(axis=0) for i in self.classes_])
        insideSum = []
        for i,c in enumerate(self.classes_):
            xi_mu = X[y==c] - self.mu_[i]
            insideSum.append(xi_mu.T @ xi_mu )
        self.cov_ = sum(insideSum)/(y.shape[0]-self.classes_.shape[0]) # forum stated to use unbiased
        self._cov_inv = inv(self.cov_)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        indexesOfMax = self.likelihood(X).argmax(axis=1)
        return self.classes_[indexesOfMax]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        cov_det = np.linalg.det(self.cov_)
        cov_inv = self._cov_inv

        def normal_pdf_multi(X_i, mu_i):
            coefficient = 1 / np.sqrt(np.power(2 * np.pi, len(X_i)) * cov_det)
            exp = np.exp((-1 / 2) * np.transpose(X_i - mu_i) @ cov_inv @ (X_i - mu_i))
            return coefficient * exp

        likelihoodMatrix = np.zeros((X.shape[0],self.classes_.size))
        for i, sample in enumerate(X):
            for k, class_name in enumerate(self.classes_):
                likelihoodMatrix[i][k] = normal_pdf_multi(sample,self.mu_[k]) * self.pi_[k]
        return likelihoodMatrix




    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y,self.predict(X))

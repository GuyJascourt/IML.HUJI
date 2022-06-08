from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2,2,n_samples)
    f = lambda x: (x+3) * (x+2) * (x+1) * (x-1) * (x-2)
    y = f(X) + np.random.normal(0, noise, n_samples)
    trainX ,trainY, testX, testY = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion=2/3)
    trainX = np.array(trainX).flatten()
    trainY = np.array(trainY).flatten()
    testX = np.array(testX).flatten()
    testY = np.array(testY).flatten()
    go.Figure([go.Scatter(x=X, y=f(X), mode='markers+lines', name='True Without Noise',marker_color='black'),
              go.Scatter(x=trainX, y=trainY, mode='markers', name='Train Set', marker_color='blue'),
              go.Scatter(x=testX, y=testY, mode='markers', name='Test Set', marker_color='red')
              ]).update_layout(title=f'Using {n_samples} Samples, with noise: {noise}',xaxis_title="x", yaxis_title='f(X)').show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    trainError, validationError = [], []
    for k in range(11):
        estimator = PolynomialFitting(k)
        currentTrainError, currentValidationError = cross_validate(estimator, trainX, trainY, mean_square_error)
        trainError.append(currentTrainError)
        validationError.append(currentValidationError)
    go.Figure([go.Scatter(x=np.arange(11), y=trainError, mode='markers+lines', name='Train Errors'),
               go.Scatter(x=np.arange(11), y=validationError, mode='markers+lines', name='Validation Errors')]).\
        update_layout(title='Polynomial Validation and Train Error using Cross-Validation',
                      xaxis_title='k', yaxis_title='Error').show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error

    kStar = np.argmin(validationError)
    trainedPloynomialModel = PolynomialFitting(kStar).fit(trainX,trainY)
    testError = round(mean_square_error(testY, trainedPloynomialModel.predict(testX)),2)
    print(f"{n_samples} Samples, with noise of {noise}")
    print(f"K*: {kStar}")
    print(f"Test Error = {testError}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    trainX, trainY, testX, testY = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0, 3, n_evaluations)

    ridgeTrainErrors, ridgeValidationErrors = [], []
    lassoTrainErrors, lassoValidationErrors = [], []
    for lam in lambdas:
        ridgeEstimator = RidgeRegression(lam)
        lassoEstimator = Lasso(lam)
        currentTrainError, currentValidationError = cross_validate(ridgeEstimator, trainX, trainY, mean_square_error)
        ridgeTrainErrors.append(currentTrainError)
        ridgeValidationErrors.append(currentValidationError)
        currentTrainError, currentValidationError = cross_validate(lassoEstimator, trainX, trainY, mean_square_error)
        lassoTrainErrors.append(currentTrainError)
        lassoValidationErrors.append(currentValidationError)
    go.Figure([go.Scatter(x=lambdas, y=ridgeTrainErrors, mode='markers+lines', name='Ridge Train Error'),
               go.Scatter(x=lambdas, y=ridgeValidationErrors,mode='markers+lines', name='Ridge Validation Error')]).\
        update_layout(title=f'Ridge Train and Validation Errors for {n_samples} Samples',
                      xaxis_title='lambda', yaxis_title='Error').show()

    go.Figure([go.Scatter(x=lambdas, y=lassoTrainErrors, mode='markers+lines', name='Lasso Train Error'),
               go.Scatter(x=lambdas, y=lassoValidationErrors, mode='markers+lines', name='Lasso Validation Error')]).\
        update_layout(title=f'Lasso Train and Validation Errors for {n_samples} Samples',
                      xaxis_title='lambda', yaxis_title='Error').show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    bestRidgeLambdaIndex = np.argmin(np.array(ridgeValidationErrors))
    bestLassoLambdaIndex = np.argmin(np.array(lassoValidationErrors))
    bestRidgeLambda = lambdas[bestRidgeLambdaIndex]
    bestLassoLambda = lambdas[bestLassoLambdaIndex]
    fittedBestRidge = RidgeRegression(bestRidgeLambda).fit(trainX,trainY)
    fittedBestLasso = Lasso(bestLassoLambda).fit(trainX,trainY)
    fittedLinear = LinearRegression().fit(trainX,trainY)
    ridgeLoss = mean_square_error(testY, fittedBestRidge.predict(testX))
    lassoLoss = mean_square_error(testY, fittedBestLasso.predict(testX))
    linearLoss = fittedLinear.loss(testX,testY)
    print(f"Best Lambda for Ridge Estimator: {bestRidgeLambda}")
    print(f"Best Lambda for Lasso Estimator: {bestLassoLambda}")
    print(f"Ridge Error: {ridgeLoss}")
    print(f"Lasso Error: {lassoLoss}")
    print(f"Linear Error: {linearLoss}")

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()

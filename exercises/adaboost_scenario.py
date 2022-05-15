import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    myAda = AdaBoost(lambda: DecisionStump(), n_learners)
    myAda.fit(train_X, train_y)

    trainingResults = []
    testResults = []
    for i in range(1, n_learners + 1):
        trainingResults.append(myAda.partial_loss(train_X, train_y, i))
        testResults.append(myAda.partial_loss(test_X, test_y, i))
    numberOfLearnersArray = np.arange(n_learners)
    Q1 = px.line(title="Training and Test Errors as a Function of Number of Fitted Learners") \
        .add_scatter(x=numberOfLearnersArray, y=trainingResults, name="Training") \
        .add_scatter(x=numberOfLearnersArray, y=testResults, name="Test").show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    subPlotsTitles = [f"AdaBoost Trained Up To Iteration: {t}" for t in T]
    Q2 = make_subplots(2, 2, subplot_titles=subPlotsTitles)
    for idx, iterationNumber in enumerate(T):
        trace = [
            decision_surface(lambda X: myAda.partial_predict(X, iterationNumber),
                             lims[0], lims[1], showscale=False),
            go.Scatter(
                x=test_X[:, 0],
                y=test_X[:, 1],
                mode="markers",
                marker=dict(color=test_y),
                showlegend=False)]
        Q2.add_traces(trace, rows=int(idx / 2) + 1, cols=(idx % 2) + 1)
    Q2.update_layout(title="Subplots of Decision surfaces for different number of iterations").show()

    # Question 3: Decision surface of best performing ensemble
    minTestIndex = np.argmin(testResults)
    accuracy = 1 -testResults[minTestIndex]
    Q3 = go.Figure()
    trace = [decision_surface(lambda X: myAda.partial_predict(X, minTestIndex),
                         lims[0], lims[1], showscale=False),
        go.Scatter(
            x=test_X[:, 0],
            y=test_X[:, 1],
            mode="markers",
            marker=dict(color=test_y),
            showlegend=False)]
    Q3.add_traces(trace).update_layout(title=f"Decision Boundry of Ensemble Size: {minTestIndex + 1} With Accuracy: {accuracy}").show()


    # Question 4: Decision surface with weighted samples
    D = myAda.D_
    D = (D/np.max(D)) * 20
    Q4 = go.Figure().add_traces(
        [decision_surface(lambda X: myAda.partial_predict(X, minTestIndex), lims[0], lims[1], showscale=False),
            go.Scatter(
                x=train_X[:, 0],
                y=train_X[:, 1],
                mode="markers",
                showlegend=False,
                marker=dict(color=train_y.astype(int),
                            colorscale=[custom[0], custom[-1]],
                            line=dict(color="black", width=1),
                            size=D))])

    Q4.update_layout(title=f"Decision surface with weighted samples").show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)

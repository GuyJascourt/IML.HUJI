from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "../datasets/linearly_separable.npy"), ("Linearly Inseparable", "../datasets/linearly_inseparable.npy")]:
        # Load dataset
        db, response = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(callback=lambda x,y,z:losses.append(x.loss(db,response)))
        perceptron.fit(db,response)

        # Plot figure of loss as function of fitting iteration
        numberOfIterations = np.arange(1,len(losses)+1)
        px.line(x=numberOfIterations, y=np.array(losses),
                title="Perceptron's loss as a function of iteration number",
                labels={"x": "Iteration Number", "y": "Perceptron's Loss"}).show()



def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    for f in ["../datasets/gaussian1.npy", "../datasets/gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes()
        lda = LDA()
        lda.fit(X,y)
        gnb.fit(X,y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots

        # Used code example for subplots
        from IMLearn.metrics import accuracy
        ldaScore = accuracy(y, lda.predict(X))
        gnbScore = round(accuracy(y,gnb.predict(X)),4)

        modelsNames = ["GNB, Accuracy: " + str(gnbScore),"LDA, Accuracy: " + str(ldaScore)]
        models = [gnb,lda]
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        fig = make_subplots(rows=1, cols=len(models), subplot_titles=modelsNames, horizontal_spacing=0.1)
        for i, model in enumerate(models):

            fig.add_traces([decision_surface(model.predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=y, symbol=class_symbols[y], colorscale=class_colors(3),
                                               line=dict(color="black", width=1))),
                            go.Scatter(x=model.mu_[:,0],y=model.mu_[:,1],mode="markers",
                                       marker=dict(color="black", symbol="x"),showlegend=False)],
                       rows=1, cols=i + 1)
            for j in range(len(model.classes_)):
                if i == 0: #GNB
                    fig.add_traces(get_ellipse(model.mu_[j],np.diag(model.vars_[j])),rows=1, cols=i + 1)
                else: #LDA
                    fig.add_traces(get_ellipse(model.mu_[j], model.cov_),rows=1, cols=i + 1)
        fig.update_layout(title="GNB and LDA prediction over: " + f, margin=dict(t=100))
        fig.show()
#
# if __name__ == '__main__':
#     np.random.seed(0)
#     gnb = GaussianNaiveBayes()
#     #Q1
#     S = np.array([0,0,1,0,2,1,3,1,4,1,5,1,6,2,7,2]).reshape((8,2))
#     gnb.fit(S[:,0],S[:,1])
#     Q1resa = gnb.pi_
#     Q1resb = gnb.mu_
#
#     #Q2
#     S = np.array([1,1,1,2,2,3,2,4,3,3,3,4]).reshape((6,2))
#     B = np.array([0,0,1,1,1,1])
#     gnb.fit(S,B)
#     Q2res = gnb.vars_
#
#     #Q3
#     run_perceptron()
#
#
#     a = 0
#
#

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    housesDF = pd.read_csv(filename).dropna().drop_duplicates()
    housesDF = housesDF.loc[~(housesDF == 0).all(axis=1)]
    #features that need to be positive
    for feature in ["price", "condition", "floors", "grade", "sqft_lot15", "sqft_living15"]:
        housesDF = housesDF[housesDF[feature] > 0]
    #features that have a bound
    housesDF = housesDF[(housesDF["bedrooms"] < 15) & (housesDF["sqft_lot"] < 1500000) &
                        (housesDF["sqft_lot15"] < 500000)]
    prices = housesDF.price
    housesDF = housesDF.drop(["price", "id", "lat", "long", "date"], 1)
    return (housesDF, prices)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        cov = np.cov(X[feature], y)[0, 1]
        stdFeature = np.std(X[feature])
        stdY = np.std(y)
        corr = cov / (stdY * stdFeature)
        featurePlot = px.scatter(x=X[feature], y=y, title=f"Feature: '{feature}'<br>Pearson Correlation: {corr}",
                                 labels={"x": f"'{feature}' Feature Values", "y": "Response values"})
        featurePlot.write_image(output_path + "%s.png" % feature)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    db, response = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(db, response)

    # Question 3 - Split samples into training- and testing sets.
    trainingX, trainingY, textX, testY = split_train_test(db, response, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    x = np.arange(10, 101)
    lossArray = []
    varianceArray = []
    for p in x:
        pLosses = []
        for i in range(10):
            pTrainingX = trainingX.sample(frac=p * 0.01)
            pTrainingY = trainingY.loc[pTrainingX.index]
            lin = LinearRegression()
            lin.fit(pTrainingX, pTrainingY)
            pLosses.append(lin.loss(textX, testY.to_numpy()))
        lossArray.append(np.mean(pLosses))
        varianceArray.append(np.std(pLosses))
    npLoss = np.array(lossArray)
    npVar = np.array(varianceArray)
    go.Figure([go.Scatter(x=x, y=npLoss - 2 * npVar, fill=None, mode="lines", line=dict(color="lightgrey"),
                          showlegend=False),
               go.Scatter(x=x, y=npLoss + 2 * npVar, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                          showlegend=False),
               go.Scatter(x=x, y=lossArray, mode="markers+lines", marker=dict(color="black", size=1),
                          showlegend=False)],
              layout=go.Layout(
                  title="Mean loss as a function of %p of sample with confidence interval",
                  height=600)).show()

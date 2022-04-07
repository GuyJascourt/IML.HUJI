import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    tempDF = pd.read_csv(filename,parse_dates=['Date']).dropna().drop_duplicates()
    tempDF['DayOfYear'] = tempDF['Date'].dt.dayofyear
    tempDF = tempDF[tempDF["Temp"]>-60]
    temperatures = tempDF.Temp
    tempDF = tempDF.drop(["Temp"],1)
    return tempDF, temperatures



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    db, response = load_data("../datasets/City_Temperature.csv")

    # # Question 2 - Exploring data for specific country
    dbIsrael = db.loc[db.Country == "Israel"]
    responseIsrael = response.loc[dbIsrael.index]
    dbIsrael["Year"] = dbIsrael["Year"].astype(str)
    px.scatter(x=dbIsrael['DayOfYear'], y=responseIsrael, title="Average daily temperature as function of day of year",
               color=dbIsrael['Year'],
                             labels={"x": "Day Of The Year", "y": "Average Daily Temperature"}).show()
    fulldbIsrael = dbIsrael
    fulldbIsrael["Temp"] = responseIsrael
    dbIsraelTempStd = fulldbIsrael.groupby(['Month'], as_index=False)["Temp"].std()
    px.bar(dbIsraelTempStd, x=dbIsraelTempStd['Month'], y=dbIsraelTempStd["Temp"],
           title="Standard Deviation of daily temperature by month",
           labels={"x":"Month", "y":"STD of Daily Temperature"}).show()

    # Question 3 - Exploring differences between countries
    fulldb = db
    fulldb["Temp"] = response
    dbQ3 = db.groupby(['Country','Month'], as_index=False)
    stdOfCountriesByMonth = dbQ3["Temp"].std()
    meanOfCountriesByMonth = dbQ3["Temp"].mean()
    px.line(meanOfCountriesByMonth,x='Month', y="Temp", color='Country', error_y=stdOfCountriesByMonth["Temp"],
            title="Average Monthly Temperature for Each Country",
            labels={"x":"Month", "y":"Average Temperature"}).show()

    # Question 4 - Fitting model for different values of `k`
    kRange = np.arange(1,11)
    trainingX, trainingY, textX, testY = split_train_test(dbIsrael, responseIsrael,0.75)
    polyLosses = []
    for k in range(1,11):
        poly = PolynomialFitting(k)
        poly.fit(trainingX.DayOfYear,trainingY)
        loss = np.round(poly.loss(textX.DayOfYear,testY),2)
        print(loss)
        polyLosses.append(loss)
    lossesnp = np.array(polyLosses)
    px.bar(x=kRange, y=lossesnp,title="Loss Of Model For Each Polynomial Degree k",
           labels={"x":"Degree of Model","y":"Loss of Model"}).show()

    # Question 5 - Evaluating fitted model on different countries
    poly = PolynomialFitting(5)
    poly.fit(dbIsrael.DayOfYear,responseIsrael)
    dbNoIsrael = db.loc[db.Country != "Israel"]
    dbNoIsraelGrouped = dbNoIsrael.groupby(['Country'], as_index=False)
    countriesLosses = []
    countries = []
    for country in dbNoIsraelGrouped.Country:
        countries.append(country[0])
        onlyCountry = db.loc[db.Country == country[0]]
        countriesLosses.append(poly.loss(onlyCountry.DayOfYear,onlyCountry.Temp))
    px.bar(x=np.array(countries), y=countriesLosses,
           title="Loss of model fitted by Israel for each Country",
           labels={"x":"Country","y":"Models Error"}).show()

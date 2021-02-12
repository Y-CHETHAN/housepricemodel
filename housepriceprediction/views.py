from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import pickle

# Create your views here.


def HouseModelTraining(request):
    context = {}
    data= pd.read_csv("House_data_preprocessed.csv")
    context["samples"] = data.shape[0]

    if request.method == 'GET':
        context["score"] = "-"

    if request.method == 'POST':
        y = data["price"]
        x = data.drop("price", axis="columns")
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2)
        house_model = LinearRegression()
        house_model.fit(x_train, y_train)
        score = house_model.score(x_test, y_test)
        context["score"] = score
        with open('house_model.pickle', 'wb') as f:
            pickle.dump(house_model, f)
    return render(request, 'housepriceprediction/HouseModelTraining.html', context)


def HouseModelPrediction(request):
    context = {}
    data = pd.read_csv("House_data_preprocessed.csv")
    context["locations"] = data.columns[4:]

    if request.method == 'GET':
        context["area"] = '1500'
        context["bathrooms"] = '2'
        context["bhk"] = '3'
        context["location"] = ''
        context["price"] = '-'

    if request.method == 'POST':
        area = int(request.POST.get('area', 0))
        bathrooms = int(request.POST.get('bathrooms', 0))
        bhk = int(request.POST.get('bhk', 0))
        location = request.POST.get('location', '-')

        context['area'] = area
        context['bathrooms'] = bathrooms
        context['bhk'] = bhk
        context['location'] = location

        y = data["price"]
        x = data.drop("price", axis="columns")

        with open('house_model.pickle', 'rb') as f:
            house_model = pickle.load(f)

        loc_index = np.where(x.columns == location)[0][0]

        input = np.zeros(len(x.columns))
        input[0] = area
        input[1] = bathrooms
        input[2] = bhk
        if loc_index >= 0:
            input[loc_index] = 1
        price = house_model.predict([input])
        context['price'] = "{0:.2f}".format(price[0])
    return render(request, 'housepriceprediction/HouseModelPrediction.html', context)

# House Price Prediction
## AIM:
To design a website to train the house price model and to predict the price.

## DESIGN STEPS:
### Step 1: 
Requirement collection.
### Step 2:
Creating the layout using HTML and CSS.
### Step 3:
Train the model using the given data set
### Step 4:
Save the trained model using pickle
### Step 5:
Get the input from the user
### Step 6:
Load the trained model using pickle
### Step 7:
Apply the given data to the model
### Step 8:
Display the result
### Step 9:
Publish the website in the given URL.

## PROGRAM:
### HouseModelTraining.html
```
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1,shrink-to-fit=no">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <title>House Price Prediction</title>
</head>

<body>
    <div class="jumbotron jumbotron-fluid" style="background-color:grey;">
        <div class="container text-center">
            <h1>Price Model Training</h1>
            <p class="lead">You can use this website to calculate your house price</p>
        </div>
    </div>
    <div class="container">
        <form action="/HouseModelTraining/" method="POST">
            {% csrf_token %}
            <div class="form-group row">
                <label for="samples" class="col-md-2 col-form-label">Total Samples</label>
                <div class="col-md-10">
                    <input type="text" readonly class="form-control" id="samples" placeholder="-" value="{{samples}}">
                </div>
            </div>
            <div class="form-group row text-center">
                <div class="col-md-12">
                    <button type="submit" class="btn btn-primary">Train</button>
                </div>
            </div>
            <div class="form-group row">
                <label for="score" class="col-md-2 col-form-label">Score</label>
                <div class="col-md-10">
                    <input type="text" readonly class="form-control" id="score" placeholder="-" value="{{score}}">
                </div>
            </div>
        </form>
    </div>
</body>
</html>
```

### HouseModelPrediction.html
```
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1,shrink-to-fit=no">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <title>House Price Prediction</title>
</head>

<body>
    <div class="jumbotron jumbotron-fluid" style="background-color:grey;">
        <div class="container text-center">
            <h1>House Price Prediction</h1>
            <p class="lead">You can use this website to calculate your house price</p>
        </div>
    </div>
    <div class="container">
        <form action="/HouseModelPrediction/" method="POST">
            {% csrf_token %}
            <div class="form-group row">
                <label for="area" class="col-md-2 col-form-label">Total Area</label>
                <div class="col-md-10">
                    <input type="text" class="form-control" id="area" name="area" placeholder="1500" value="{{area}}">
                </div>
            </div>
            <div class="form-group row">
                <label for="bathrooms" class="col-md-2 col-form-label">Bathrooms</label>
                <div class="col-md-10">
                    <input type="text" class="form-control" id="bathrooms" name="bathrooms" placeholder="2"
                        value="{{bathrooms}}">
                </div>
            </div>
            <div class="form-group row">
                <label for="bhk" class="col-md-2 col-form-label">Bedroom Hall Kitchen</label>
                <div class="col-md-10">
                    <input type="text" class="form-control" id="bhk" name="bhk" placeholder="3" value="{{bhk}}">
                </div>
            </div>
            <div class="form-group row">
                <label for="location" class="col-md-2 col-form-label">Location</label>
                <div class="col-md-10">
                    <select id="location" name="location" class="col-md-10 form-control">
                        {% for loc in locations %}
                        {% if loc == location %}
                        <option selected value="{{loc}}">{{loc}}</option>
                        {% else %}
                        <option value="{{loc}}">{{loc}}</option>
                        {% endif %}
                        {% endfor %}
                    </select>
                </div>
            </div>
            <div class="form-group row text-center">
                <div class="col-md-12">
                    <button type="submit" class="btn btn-primary">Calculate</button>
                </div>
            </div>
            <div class="form-group row">
                <label for="price" class="col-md-2 col-form-label">Price(Lakh)</label>
                <div class="col-md-10">
                    <input type="text" readonly class="form-control" id="price" placeholder="-" value="{{price}}">
                </div>
            </div>
        </form>
    </div>
</body>

</html>
```
### views.py
```
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

```

## PYTHON CODE:
![python3](./static/img/python_code.png)

## OUTPUT:
![HouseModelTraining](./static/img/HMT.png)
![HouseModelPrediction](./static/img/HMP.png)

## RESULT:
Thus a website is designed to train the house price model and to predict the price and is hosted in the URL http://chethan.student.saveetha.in:8000/HouseModelTraining/ and http://chethan.student.saveetha.in:8000/HouseModelPrediction/.
from django.shortcuts import render
import joblib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'ml_model', 'iris_classifier.pkl')
model = joblib.load(model_path)

def Irisflower(request):
    result = None
    if request.method == "POST":
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(features)

        target_name = ['setosa', 'versicolor', 'virginica']
        result = target_name[prediction[0]]

    return render(request, 'iris.html', {'result': result})

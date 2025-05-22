import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

#define the columns and the outputs
iris = load_iris()
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
data = pd.DataFrame(iris.data,columns=column_names)
data['class'] = iris.target


x = data[column_names]
y = data['class']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(x_train,y_train)

predictions = model.predict(x_test)
print("Accuraccy", accuracy_score(y_test,predictions))

joblib.dump(model, "ml_model/iris_classifier.pkl")
from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.Irisflower, name='iris_predict'),
]
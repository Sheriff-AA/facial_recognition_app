from django.urls import path
from . import views


"""
BASE ENDPOINT /faceapp
"""
app_name = "faceapp"


urlpatterns = [
    path('register/', views.register_face, name='register_face'),
    path('recognize/', views.recognize_face, name='recognize_face'),
]



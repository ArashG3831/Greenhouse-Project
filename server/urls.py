from django.urls import include, path

urlpatterns = [
    path('', include('sensor.urls')),

]

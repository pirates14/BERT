from django.urls import path

from mysite.views import test

urlpatterns =[
    path('', test, name='test'),
]
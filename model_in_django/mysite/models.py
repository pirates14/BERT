from django.db import models

# Create your models here.

class Predictions(models.Model):
    context = models.CharField(max_length=300)

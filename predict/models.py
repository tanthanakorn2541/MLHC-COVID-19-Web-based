from django.db import models

# Create your models here.
class Files(models.Model):
    images = models.FileField(upload_to='images')

class Class_index:
    path: str
    title: str
    result: str
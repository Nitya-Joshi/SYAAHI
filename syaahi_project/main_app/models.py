from django.db import models

class BookSimilarity(models.Model):
    book1_id = models.IntegerField()
    book2_id = models.IntegerField()
    similarity_score = models.FloatField()

# Create your models here.

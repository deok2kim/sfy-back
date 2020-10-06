from django.db import models
# from django.conf import settings
from imagekit.models import ProcessedImageField
from imagekit.processors import ResizeToFill, ResizeToFit

class Animal(models.Model):
	name = models.CharField(max_length=50)
	info = models.TextField()
	# sound = models.FileField()

class Photo(models.Model):
	image = ProcessedImageField(
		processors=[ResizeToFill(500, 500)],
		format='JPEG',
		options={'quality': 90},
		blank=True,
		)

class Post(models.Model):
	# user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, blank=True)
	animal = models.ForeignKey(Animal, on_delete=models.CASCADE)
	created_at = models.DateTimeField(auto_now_add=True)
	image = ProcessedImageField(
		processors=[ResizeToFill(500, 500)],
		format='JPEG',
		options={'quality': 90},
		blank=True,
		)
	info = models.TextField()
	name = models.CharField(max_length=50)
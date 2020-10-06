from rest_framework import serializers
from .models import Animal, Post, Photo
from accounts.serializers import UserSerializer

class AnimalSerializer(serializers.ModelSerializer):
    class Meta:
        model = Animal
        fields = '__all__'
    
class PostSerializer(serializers.ModelSerializer):
    # user = UserSerializer(required=False)
    class Meta:
        model = Post
        fields = '__all__'

class PostListSerializer(serializers.ModelSerializer):
    # user = UserSerializer(required=False)
    class Meta:
        model = Post
        fields = '__all__'

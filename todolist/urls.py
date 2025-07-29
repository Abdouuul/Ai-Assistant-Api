from django.urls import path, include
from rest_framework import routers
from todolist.views import TaskViewSet

router = routers.DefaultRouter()
router.register('', TaskViewSet, basename='task')
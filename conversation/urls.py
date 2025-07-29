from django.urls import path
from conversation.views import ConversationView

urlpatterns = [
    path('', ConversationView.as_view(), name='conversation'),
]

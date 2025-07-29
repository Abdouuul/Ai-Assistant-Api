from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

# Create your views here.

class ConversationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        message = request.data.get("message")
        if not message:
            return Response({"error":"Message is required"}, status=400)
        
        

        return Response({
            "reply":"...",
            "created_task": None,
            "is_task": False,
        })
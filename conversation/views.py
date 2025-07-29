from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from conversation.graph import assistant_graph, AssistantState

# Create your views here.

class ConversationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        message = request.data.get("message")
        if not message:
            return Response({"error":"Message is required"}, status=400)
        
        # initial_state: AssistantState = {
        #     "message": message, 
        #     "user_id": request.user.id
        # }

        # Cleaner way to do it
        initial_state = AssistantState(
            message=message, 
            user_id=request.user.id,
        )
        

        result = assistant_graph.invoke(initial_state)
        print('Result : ', result)
        return Response({
            "reply": result['reply'],
            "is_task": result['is_task'],
        })
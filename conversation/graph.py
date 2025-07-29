from langchain_ollama import OllamaLLM
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from todolist.models import Task
from django.contrib.auth.models import User

llm = OllamaLLM(model="llama3.1")

class IsTodo(BaseModel):
    is_task: bool

detect_prompt = PromptTemplate.from_template(
    """Tu es un assistant qui détermine si l'input de l'utilisateur devrait être traité comme une tâche Todo ou non.
    Tu dois répondre avec un booléem `is_task`:
    - `true` si l'input est une tâche à créer
    - `false` sinon

    Input de l'utilisateur:
    {message}
    """
)

def detect_task_intent(state):
    message = state.get("message")
    structured_llm = llm.with_structured_output(IsTodo, prompt=detect_prompt)

    result = structured_llm.invoke({"message":message})

    return {
        "is_task": result.is_task,
        **state
    }

class ExtractedTask(BaseModel):
    task_content: str

extract_prompt = PromptTemplate.from_template(
    """Tu dois extraire la tâche à créer de l'input de l'utilisateur.
    Tu dois répondre avec un champ unique `task_content`.
    Exemples :
    - "Il faut que je réserve mon billet d'avion" --> "Réserver mon billet d'avion"
    - "Je dois faire les courses" --> "Faire les courses"
    - "N'oublie pas de m'acheter du pain" --> "Acheter du pain"
    - "Ajoute à la todolist de préparer une éval bien compliquée !" --> "Préparer une éval bien compliquée"

    Input de l'utilisateur :
    {message}                                         
""")

def extract_task_content(state):
    message = state.get("message")
    structured_llm = llm.with_structured_output(ExtractedTask, prompt=extract_prompt)
    result = structured_llm.invoke({"message": message})
    return {
        "task_content": result.task_content,
        **state
    }


def create_task(state):
    task_content = state.get("task_content")
    user_id = state.get("user_id")
    user = User.objects.get(id=user_id)
    task = Task(
        content = task_content,
        user=user
    )
    task.save()

class ReplyContent(BaseModel):
    reply: str

acknowledge_prompt = PromptTemplate.from_template("""
    Tu dois répondre `a l'utilisateur pour lui confirmer aue la tâchhe a bien été créée.
    Tu dois répondre d'une façon sympathique.

    Tâche ajoutée à la Todo :
        {task_content}
    Message original de l'utilisateur :
        {message}
""")

def acknowledge_task_creation(state):
    task_content = state.get("task_content")
    message = state.get("message")

    structured_llm = llm.with_structured_output(ReplyContent, prompt=acknowledge_prompt)
    result = structured_llm.invoke({
        "task_content": task_content,
        "message": message
    })

    return {
        "reply": result.reply,
        **state
    }
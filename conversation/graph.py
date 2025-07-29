from langchain_ollama import OllamaLLM
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from todolist.models import Task
from django.contrib.auth.models import User
from typing import Optional
from langgraph.graph import StateGraph, END

llm = OllamaLLM(model="llama3.1")

class AssistantState(BaseModel):
    user_id: Optional[int] = None
    message: Optional[str] = None
    is_task: bool = False
    task_content: Optional[str] = None
    reply: Optional[str] = None

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

def detect_task_intent(state: AssistantState) -> AssistantState:
    message = state.message
    structured_llm = llm.with_structured_output(IsTodo, prompt=detect_prompt)
    chain = detect_prompt | structured_llm
    result = chain.invoke({"message": message})
    return AssistantState(
        is_task = result.is_task,
        **state.model_dump()
    )

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

def extract_task_content(state: AssistantState) -> AssistantState:
    message = state.get("message")
    structured_llm = llm.with_structured_output(ExtractedTask, prompt=extract_prompt)
    chain = extract_prompt | structured_llm
    result = structured_llm.invoke({"message": message})
    
    return AssistantState(
        task_content = result.task_content,
        **state.model_dump()
    )


def create_task(state: AssistantState) -> AssistantState:
    task_content = state.get("task_content")
    user_id = state.get("user_id")
    user = User.objects.get(id=user_id)
    task = Task(
        title = task_content,
        user=user
    )
    task.save()

    return state

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

def acknowledge_task_creation(state: AssistantState) -> AssistantState:
    task_content = state.get("task_content")
    message = state.get("message")

    structured_llm = llm.with_structured_output(ReplyContent, prompt=acknowledge_prompt)
    chain = acknowledge_prompt | structured_llm
    result = chain.invoke({
        "task_content": task_content,
        "message": message
    })

    return AssistantState(
        reply = result.reply,
        **state.model_dump()
    )

response_prompt = PromptTemplate.from_template("""
Tu dois répondre à l'utilisateur en t'appuyant sur test connaissance générales. 
Question : {message}""")

def response_to_user(state):
    message = state.get("message")
    structured_llm = llm.with_structured_output(ReplyContent)
    chain = response_prompt | structured_llm 
    result = chain.invoke({"message": message})

    return AssistantState(
        reply = result.reply,
        **state.model_dump()
    )

# Graph creation

graph = StateGraph(AssistantState)
graph.add_node("detect_task", detect_task_intent)
graph.add_node("extract_task", extract_task_content)
graph.add_node("create_task", create_task)
graph.add_node("acknowledge_task", acknowledge_task_creation)
graph.add_node("response_to_user", response_to_user)

graph.set_entry_point("detect_task")
graph.add_conditional_edges("detect_task", lambda state: "extract_task" if state.is_task else "respond_to_user")
graph.add_edge("extract_task", "create_task")
graph.add_edge("create_task", "acknowledge_task")
graph.add_edge("acknowledge_task", END)
graph.add_edge("repond_to_user", END)

assistant_graph = graph.compile()
assistant_graph.get_graph().draw_mermaid_png(output_file_path="assistant_graph.png")
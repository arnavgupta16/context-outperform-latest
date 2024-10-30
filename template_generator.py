from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from langchain.graphs import Graph
from groq import Groq

# Initialize Groq
client = Groq(
    api_key="your_api_key_here"
)

@dataclass
class DocumentState:
    """Represents the state of the document generation workflow"""
    document_type: str
    template: str = ""
    document: str = ""
    feedback: List[str] = field(default_factory=list)
    is_satisfied: bool = False
    messages: List[Dict[str, str]] = field(default_factory=list)

def get_groq_response(prompt: str) -> str:
    """Helper function to get response from Groq"""
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.2-90b-vision-preview",
        temperature=0.7,
        max_tokens=2048
    )
    return completion.choices[0].message.content

def generate_template(state: DocumentState) -> DocumentState:
    """Generate a document template based on the user's requirements"""
    prompt = f"""You are a document template generator. Create a detailed template for the following document type:
{state.document_type}

Please provide a structured template with clear sections and placeholders."""
    state.template = get_groq_response(prompt)
    return state

def gather_information(state: DocumentState) -> DocumentState:
    """Gather necessary information from the user to fill the template"""
    prompt = f"""Based on the following template, identify what information is needed and ask the user specific questions to fill it:

Template:
{state.template}

Please list the specific information needed from the user."""
    state.messages.append({"role": "assistant", "content": get_groq_response(prompt)})
    return state

def generate_document(state: DocumentState) -> DocumentState:
    """Generate the final document based on the template and user information"""
    user_info = state.messages[-1]["content"]
    prompt = f"""Generate a complete document based on the template and provided information:

Template:
{state.template}

User Information:
{user_info}

Please create a polished, professional document incorporating all the provided information."""
    state.document = get_groq_response(prompt)
    return state

def get_feedback(state: DocumentState) -> DocumentState:
    """Get feedback from the user on the generated document"""
    prompt = f"""Review the following document and ask the user for specific feedback:

Document:
{state.document}

Please ask the user about specific aspects they'd like to improve or modify."""
    state.messages.append({"role": "assistant", "content": get_groq_response(prompt)})
    return state

def implement_feedback(state: DocumentState) -> DocumentState:
    """Implement the user's feedback into the document"""
    feedback = state.messages[-1]["content"]
    prompt = f"""Revise the document based on the user's feedback while maintaining the original requirements:

Original Document:
{state.document}

User Feedback:
{feedback}

Please provide an improved version incorporating the feedback."""
    state.document = get_groq_response(prompt)
    state.feedback.append(feedback)
    return state

def check_satisfaction(state: DocumentState) -> Tuple[str, DocumentState]:
    """Check if the user is satisfied with the document"""
    last_user_message = state.messages[-1]["content"].lower()
    state.is_satisfied = "satisfied" in last_user_message or "good" in last_user_message
    return ("end" if state.is_satisfied else "feedback", state)

# Create the workflow graph
workflow = Graph()

# Add nodes
workflow.add_node("generate_template", RunnableLambda(generate_template))
workflow.add_node("gather_info", RunnableLambda(gather_information))
workflow.add_node("generate_doc", RunnableLambda(generate_document))
workflow.add_node("get_feedback", RunnableLambda(get_feedback))
workflow.add_node("implement_feedback", RunnableLambda(implement_feedback))
workflow.add_node("check_satisfaction", RunnableLambda(check_satisfaction))

# Define edges
workflow.add_edge("generate_template", "gather_info")
workflow.add_edge("gather_info", "generate_doc")
workflow.add_edge("generate_doc", "get_feedback")
workflow.add_edge("get_feedback", "check_satisfaction")
workflow.add_edge("check_satisfaction", "implement_feedback")
workflow.add_edge("implement_feedback", "get_feedback")

# Compile the graph
chain = workflow.compile()

def run_document_workflow(document_type: str) -> DocumentState:
    """Run the document generation workflow with user interaction"""
    state = DocumentState(document_type=document_type)
    
    try:
        while not state.is_satisfied:
            state = chain.invoke(state)
            
            if state.messages and state.messages[-1]["role"] == "assistant":
                print(f"\nAssistant: {state.messages[-1]['content']}")
                user_input = input("\nUser: ")
                state.messages.append({"role": "user", "content": user_input})

        print("\nFinal Document:")
        print(state.document)
        return state

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    print("Welcome to the Document Generation System!")
    print("----------------------------------------")
    document_type = input("What type of document would you like to generate? ")
    result = run_document_workflow(document_type)
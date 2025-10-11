import os
import streamlit as st
import tempfile
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict
from helper_functions.rag import Preprocess  # Use your RAG class

st.sidebar.title("API Configuration")
api_key = st.sidebar.text_input("Enter your ChatGroq API Key:", type="password")
if not api_key:
    st.info("Enter your ChatGroq API key in the sidebar to proceed.")
    st.stop()

st.title("SaaS T&C Risk Analyzer (Agentic RAG Demo)")
st.markdown("""
Upload a SaaS Terms & Conditions PDF.  
Ask legal/compliance questions.
""")
uploaded_file = st.file_uploader("Upload a SaaS T&C PDF document:", type=["pdf"])

USE_BINARY = st.checkbox("Use binary quantization for embeddings (32x smaller, faster)", value=True)
chunk_size = st.slider('Chunk size (words)', min_value=100, max_value=2000, value=500, step=50)
chunk_overlap = st.slider('Chunk overlap (words)', min_value=0, max_value=500, value=50, step=10)

if uploaded_file is not None:
    # Save the uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(uploaded_file.read())
        pdf_path = tmpfile.name

    # Only process if not already done for this file
    last_pdf_hash = st.session_state.get("last_pdf_hash")
    import hashlib
    pdf_hash = hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()
    reprocess = (last_pdf_hash != pdf_hash or 
                 st.session_state.get("last_chunk_size") != chunk_size or 
                 st.session_state.get("last_chunk_overlap") != chunk_overlap or
                 st.session_state.get("last_use_binary") != USE_BINARY)
    
    if reprocess:
        st.session_state.prep = Preprocess(pdf_path)
        st.session_state.chunks = st.session_state.prep.chunk(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.session_state.prep.vectorize(binary_quantize=USE_BINARY)
        st.session_state.prep.add_to_faiss(index_path="faiss_index", use_binary=USE_BINARY)
        st.session_state["last_pdf_hash"] = pdf_hash
        st.session_state["last_chunk_size"] = chunk_size
        st.session_state["last_chunk_overlap"] = chunk_overlap
        st.session_state["last_use_binary"] = USE_BINARY
        st.success("Document processed!")
    st.markdown(f"**Document processed** with {len(st.session_state.chunks)} chunks.")
else:
    st.info("Upload a PDF to analyze!")
    st.stop()


@tool
def retrieve_chunks(query: str) -> str:
    """Returns the 3 most relevant chunks for the user's query."""
    results = st.session_state.prep.search(
        query=query,
        top_k=3,
        use_binary=USE_BINARY,
        index_path="faiss_index"
    )
    if not results:
        return "No relevant information found."
    return "\n\n".join([f"[Chunk {i+1}] {res['text']}" for i, res in enumerate(results)])

FEW_SHOTS = [
    HumanMessage(content="List clauses relating to data ownership in this T&C."),
    AIMessage(content="Sure! Searching the document for relevant clauses..."),
    SystemMessage(content="[Chunk 1] Data is owned by the customer. [Chunk 2] The provider may aggregate data for analytics."),
    AIMessage(content="Here are the relevant clauses:\n- [Chunk 1] Data is owned by the customer.\n- [Chunk 2] The provider may aggregate data for analytics.")
]

AGENT_SYSTEM_PROMPT = """You are Janie, an AI legal assistant. Use the `retrieve_chunks` tool to answer user's questions about the SaaS Terms & Conditions (T&C). Reason about the question, decide what information you need, then retrieve and synthesize an answer with concrete citations."""

class AgentState(TypedDict):
    messages: list[BaseMessage]

def call_model(state: AgentState):
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    messages += FEW_SHOTS
    messages += state['messages']
    for i, msg in enumerate(messages):
        print(i, type(msg))
        assert isinstance(msg, (HumanMessage, AIMessage, SystemMessage)), "Invalid message type!"

    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0,api_key=api_key)
    # Ensure all in messages are valid types
    for msg in messages:
        assert isinstance(msg, (HumanMessage, SystemMessage, AIMessage)), f"Found wrong type: {type(msg)}"

    response = llm.invoke(messages, tools=[retrieve_chunks])

    return {"messages": [response]}


def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "continue"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([retrieve_chunks]))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")
agent = workflow.compile()

user_query = st.text_area("Type your legal/contractual question here:", height=120)
if st.button("Ask Janie"):
    chat_history = [HumanMessage(content=user_query)]
    with st.spinner("Agent reasoning and answering..."):
        for event in agent.stream({"messages": chat_history}, stream_mode="values"):
            for value in event.values():
                if isinstance(value, dict) and "messages" in value:
                    latest = value["messages"][-1]
                    if isinstance(latest, AIMessage) and not hasattr(latest, "tool_calls"):
                        st.markdown(f"**Janie:** {latest.content}")

if st.checkbox("Show top chunks for your last query"):
    res = st.session_state.prep.search(
        query=user_query,
        top_k=3,
        use_binary=USE_BINARY,
        index_path="faiss_index"
    )
    st.write(res)

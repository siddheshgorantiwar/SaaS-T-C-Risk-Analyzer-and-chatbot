import os
import streamlit as st
import tempfile
import hashlib
from typing_extensions import TypedDict
from typing import Annotated, Sequence
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json
from helper_functions.rag import Preprocess

st.sidebar.title("API Configuration")
api_key = st.sidebar.text_input("Enter your ChatGroq API Key:", type="password")
if not api_key:
    st.info("Enter your ChatGroq API key in the sidebar to proceed.")
    st.stop()

st.title("SaaS T&C Risk Analyzer (ReAct Agent with Multi-Document RAG)")
st.markdown("""
Upload multiple SaaS Terms & Conditions PDFs.  
Ask legal/compliance questions across all documents.
""")

uploaded_files = st.file_uploader(
    "Upload SaaS T&C PDF documents:", 
    type=["pdf"], 
    accept_multiple_files=True
)

USE_BINARY = st.checkbox("Use binary quantization for embeddings (32x smaller, faster)", value=True)
chunk_size = st.slider('Chunk size (words)', min_value=100, max_value=2000, value=500, step=50)
chunk_overlap = st.slider('Chunk overlap (words)', min_value=0, max_value=500, value=50, step=10)

if uploaded_files:
    if "doc_processors" not in st.session_state:
        st.session_state.doc_processors = {}
    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = []
    
    current_files = {f.name: f for f in uploaded_files}
    current_file_hashes = {}
    
    for file_name, file_obj in current_files.items():
        file_obj.seek(0)
        file_hash = hashlib.md5(file_obj.read()).hexdigest()
        current_file_hashes[file_name] = file_hash
        file_obj.seek(0)
    
    reprocess = (
        st.session_state.get("last_file_hashes") != current_file_hashes or 
        st.session_state.get("last_chunk_size") != chunk_size or 
        st.session_state.get("last_chunk_overlap") != chunk_overlap or
        st.session_state.get("last_use_binary") != USE_BINARY
    )
    
    if reprocess:
        st.session_state.doc_processors = {}
        st.session_state.all_chunks = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (file_name, file_obj) in enumerate(current_files.items()):
            status_text.text(f"Processing {file_name}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                tmpfile.write(file_obj.read())
                pdf_path = tmpfile.name
            
            prep = Preprocess(pdf_path)
            chunks = prep.chunk(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            prep.vectorize(binary_quantize=USE_BINARY)
            
            doc_index_path = f"faiss_index_{file_name.replace('.pdf', '')}"
            prep.add_to_faiss(index_path=doc_index_path, use_binary=USE_BINARY)
            
            st.session_state.doc_processors[file_name] = {
                'prep': prep,
                'index_path': doc_index_path,
                'num_chunks': len(chunks)
            }
            st.session_state.all_chunks.extend(chunks)
            
            progress_bar.progress((idx + 1) / len(current_files))
            
            os.unlink(pdf_path)
        
        st.session_state["last_file_hashes"] = current_file_hashes
        st.session_state["last_chunk_size"] = chunk_size
        st.session_state["last_chunk_overlap"] = chunk_overlap
        st.session_state["last_use_binary"] = USE_BINARY
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"Processed {len(current_files)} documents!")
    
    st.markdown("### Document Statistics")
    for file_name, doc_info in st.session_state.doc_processors.items():
        st.markdown(f"**{file_name}**: {doc_info['num_chunks']} chunks")
    
    st.markdown(f"**Total chunks across all documents**: {len(st.session_state.all_chunks)}")
else:
    st.info("Upload PDF documents to analyze!")
    st.stop()

# ==================== ReAct Agent Implementation ====================

class AgentState(TypedDict):
    """The state of the ReAct agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_retrieve_tool(doc_processors, use_binary):
    """Factory function to create a multi-document retrieve tool."""
    @tool
    def retrieve_chunks(query: str, document_name: str = "all") -> str:
        """
        Retrieves the 5 most relevant chunks for the user's query.
        
        Args:
            query: The search query
            document_name: Specific document name to search, or "all" to search across all documents
        
        Returns:
            Formatted string with relevant chunks and their source documents
        """
        all_results = []
        
        if document_name == "all":
            docs_to_search = doc_processors.items()
        else:
            docs_to_search = [(name, info) for name, info in doc_processors.items() 
                            if document_name.lower() in name.lower()]
            if not docs_to_search:
                return f"No document found matching '{document_name}'. Available documents: {', '.join(doc_processors.keys())}"
        
        for doc_name, doc_info in docs_to_search:
            results = doc_info['prep'].search(
                query=query,
                top_k=3,
                use_binary=use_binary,
                index_path=doc_info['index_path']
            )
            
            for res in results:
                res['source_doc'] = doc_name
                all_results.append(res)
        
        all_results.sort(key=lambda x: x.get('distance', float('inf')))
        top_results = all_results[:5]
        
        if not top_results:
            return "No relevant information found."
        
        formatted = []
        for i, res in enumerate(top_results):
            formatted.append(
                f"[Chunk {i+1} from {res['source_doc']}]\n{res['text']}\n"
                f"(Relevance score: {1 - res.get('distance', 1):.3f})"
            )
        
        return "\n\n".join(formatted)
    
    return retrieve_chunks

def create_list_documents_tool(doc_processors):
    """Create a tool to list available documents."""
    @tool
    def list_documents() -> str:
        """Lists all available documents that can be queried."""
        doc_list = list(doc_processors.keys())
        return f"Available documents ({len(doc_list)}):\n" + "\n".join(f"- {doc}" for doc in doc_list)
    
    return list_documents

REACT_SYSTEM_PROMPT = """You are Janie, an expert AI legal assistant specializing in SaaS Terms & Conditions analysis.

You have access to multiple T&C documents and can search them to answer user questions.

Your reasoning process follows the ReAct pattern:
1. **Thought**: Reason about what information you need
2. **Action**: Use tools to retrieve relevant information
3. **Observation**: Analyze the retrieved information
4. **Answer**: Synthesize a comprehensive answer with citations

Available Tools:
- retrieve_chunks: Search for relevant information in documents (can search all or specific documents)
- list_documents: List all available documents

When answering:
- Always cite which document(s) your information comes from
- If information conflicts across documents, clearly state the differences
- Be precise and quote relevant sections when possible
- If you need more information, don't hesitate to use tools multiple times

Think step-by-step and be thorough in your analysis."""

def call_model_node(state: AgentState, llm_with_tools, system_prompt: str):
    """Node that calls the LLM with ReAct prompting."""
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState, tools_by_name: dict):
    """Node that executes tool calls."""
    outputs = []
    last_message = state["messages"][-1]
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        try:
            tool_result = tools_by_name[tool_name].invoke(tool_args)
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result,
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        except Exception as e:
            outputs.append(
                ToolMessage(
                    content=f"Error executing {tool_name}: {str(e)}",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
    
    return {"messages": outputs}

def should_continue(state: AgentState):
    """Conditional edge to determine if we should continue or end."""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    
    return "end"

def create_react_agent(doc_processors, use_binary, api_key_val):
    """Create a ReAct agent workflow."""
    
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0, api_key=api_key_val)
    
    retrieve_tool = create_retrieve_tool(doc_processors, use_binary)
    list_docs_tool = create_list_documents_tool(doc_processors)
    tools = [retrieve_tool, list_docs_tool]
    
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node(
        "agent", 
        lambda state: call_model_node(state, llm_with_tools, REACT_SYSTEM_PROMPT)
    )
    workflow.add_node(
        "tools", 
        lambda state: tool_node(state, tools_by_name)
    )
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# ==================== User Interface ====================

user_query = st.text_area("Type your legal/contractual question here:", height=120)

if st.button("Ask Janie (ReAct Agent)"):
    if not st.session_state.doc_processors:
        st.error("Please upload and process PDF documents first!")
    else:
        agent = create_react_agent(
            st.session_state.doc_processors, 
            USE_BINARY, 
            api_key
        )
        
        initial_messages = [HumanMessage(content=user_query)]
        
        with st.spinner("Janie is thinking (ReAct reasoning)..."):
            thought_container = st.expander("Agent Reasoning Process", expanded=True)
            answer_container = st.container()
            
            thoughts = []
            final_response = None
            
            for event in agent.stream({"messages": initial_messages}, stream_mode="values"):
                messages = event.get("messages", [])
                if not messages:
                    continue
                
                latest = messages[-1]
                
                if isinstance(latest, AIMessage) and hasattr(latest, "tool_calls") and latest.tool_calls:
                    for tool_call in latest.tool_calls:
                        thought = f"**Action**: {tool_call['name']}\n**Input**: {json.dumps(tool_call['args'], indent=2)}"
                        thoughts.append(thought)
                        with thought_container:
                            st.markdown(thought)
                
                elif isinstance(latest, ToolMessage):
                    observation = f"**Observation from {latest.name}**:\n``````"
                    thoughts.append(observation)
                    with thought_container:
                        st.markdown(observation)
                
                elif isinstance(latest, AIMessage):
                    if not (hasattr(latest, "tool_calls") and latest.tool_calls):
                        final_response = latest
            
            if final_response and isinstance(final_response, AIMessage):
                with answer_container:
                    st.markdown("### 💡 Janie's Answer")
                    st.markdown(final_response.content)
            else:
                st.error("No final answer received from the agent.")

if st.checkbox("Show top chunks for your last query (across all documents)"):
    if st.session_state.doc_processors and user_query:
        st.markdown("### Raw Chunk Retrieval")
        
        for doc_name, doc_info in st.session_state.doc_processors.items():
            st.markdown(f"#### From: {doc_name}")
            res = doc_info['prep'].search(
                query=user_query,
                top_k=2,
                use_binary=USE_BINARY,
                index_path=doc_info['index_path']
            )
            
            if res:
                for i, chunk in enumerate(res):
                    st.markdown(f"**Chunk {i+1}** (Distance: {chunk.get('distance', 'N/A'):.4f})")
                    st.text_area("", chunk['text'], height=150, key=f"chunk_{doc_name}_{i}", label_visibility="collapsed")
            else:
                st.info(f"No results found in {doc_name}")
    else:
        st.info("Please enter a query first.")

if st.checkbox("Show ReAct Agent Architecture"):
    st.markdown("""
    ### ReAct Agent Flow
    
    ```
    User Query → Agent (Reasoning) → Tools (Action) → Agent (Observation) → ... → Final Answer
    ```
    
    The agent iteratively:
    1. **Reasons** about what information it needs
    2. **Acts** by calling tools to retrieve information
    3. **Observes** the results
    4. Repeats until it has enough information to answer
    """)

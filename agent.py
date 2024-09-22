import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import os

class RiskItem(BaseModel):
    clause: str = Field(description="The name or type of the clause")
    risk: str = Field(description="Description of the potential risk or issue")
    recommendation: str = Field(description="Recommendation for addressing the risk")

class RiskAnalysis(BaseModel):
    risks: List[RiskItem] = Field(description="List of identified risks in the T&C")

output_parser = PydanticOutputParser(pydantic_object=RiskAnalysis)

template = """
You are an AI legal assistant named Janie, specialized in analyzing SaaS Terms and Conditions (T&C) for potential risks to buyers. Your task is to review the given T&C and identify any clauses that may pose risks or issues for the buyer.

Please analyze the following T&C text and provide a list of potential risks, focusing on the following key areas:
1. Data ownership and usage rights
2. Service level agreements (SLAs) and uptime guarantees
3. Liability limitations and indemnification
4. Termination clauses and data retrieval
5. Privacy and security measures
6. Intellectual property rights
7. Payment terms and pricing changes
8. Warranty and disclaimer of warranties
9. Compliance with regulations (e.g., GDPR, CCPA)
10. Dispute resolution and governing law

T&C Text:
{tc_text}

{format_instructions}

Provide your analysis in a clear, concise manner, highlighting the most critical issues first.
"""

prompt = ChatPromptTemplate.from_template(template)

st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

st.sidebar.markdown("""
## How to use this tool:
1. Enter your Groq API Key in the field above.
2. Paste the full text of the SaaS Terms and Conditions in the main text area.
3. Click the "Analyze T&C" button.
4. Review the identified risks and recommendations.
5. Use the chat feature to ask follow-up questions.
""")

st.title("SaaS T&C Risk Analyzer")

tc_text = st.text_area("Enter the SaaS Terms and Conditions text here:", height=300)

if st.button("Analyze T&C"):
    if api_key and tc_text:
        with st.spinner("Analyzing T&C..."):
            os.environ["GROQ_API_KEY"] = api_key

            try:
                model = ChatGroq(model_name="mixtral-8x7b-32768")

                messages = prompt.format_messages(
                    tc_text=tc_text,
                    format_instructions=output_parser.get_format_instructions()
                )

                response = model.invoke(messages)
                parsed_output = output_parser.parse(response.content)

                st.subheader("Identified Risks:")
                for risk in parsed_output.risks:
                    with st.expander(risk.clause):
                        st.write(f"**Risk:** {risk.risk}")
                        st.write(f"**Recommendation:** {risk.recommendation}")
                
                st.session_state['tc_text'] = tc_text

            except Exception as e:
                st.error(f"An error occurred: {e}")
    elif not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    else:
        st.error("Please enter the T&C text to analyze.")

st.subheader("Chat with Janie")
chat_input = st.text_input("Type your message here:")

if st.button("Send"):
    if api_key and chat_input:
        with st.spinner("Sending message..."):
            try:
                model = ChatGroq(model_name="mixtral-8x7b-32768")

                tc_text = st.session_state.get('tc_text', '')
                if tc_text:
                    retrieved_text = tc_text  

                    rag_prompt = f"""
                    You are an AI legal assistant named Janie. Based on the provided T&C text, answer the following question:
                    {chat_input}
                    T&C Text:
                    {retrieved_text}
                    """

                    messages = [{'role': 'user', 'content': rag_prompt}]
                    response = model.invoke(messages)
                    
                    st.write(f"**Assistant:** {response.content}")

                else:
                    st.error("No T&C text available for retrieval.")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    elif not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    else:
        st.error("Please enter a message to send.")
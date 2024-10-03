# SaaS T&C Risk Analyzer and Chatbot

## Overview

The SaaS T&C Risk Analyzer and Chatbot is an AI-powered tool designed to analyze SaaS Terms and Conditions (T&C) for potential risks and provide actionable insights. It also includes a chatbot feature that allows users to ask questions about the T&C, leveraging Retrieval-Augmented Generation (RAG) to enhance the relevance of responses.

## Features

- **Risk Analysis**: Analyzes T&C text and identifies potential risks and recommendations in key areas such as data ownership, service agreements, liability limitations, and more.
- **Chatbot**: A conversational agent that provides context-aware responses based on the input T&C text. Utilizes RAG to enhance the accuracy and relevance of its responses.

## Setup and Configuration

1. **API Key**: You will need an API key from Groq to use the ChatGroq model. Enter your API key in the sidebar configuration section.
2. **T&C Text Input**: Paste the full text of the SaaS Terms and Conditions in the main text area for analysis.

## Usage

1. **Analyze T&C**:
   - Enter the SaaS Terms and Conditions text in the provided text area.
   - Click the "Analyze T&C" button.
   - Review the identified risks and recommendations in the results section.

2. **Chat with Janie**:
   - Type your message or question related to the T&C in the chat input field.
   - Click the "Send" button.
   - The chatbot will provide context-aware responses based on the analyzed T&C text.

## Assumptions and Design Considerations

- **T&C Storage**: The entire T&C text is stored in memory during the session to enable context-aware responses from the chatbot. This approach assumes that the input T&C text is of manageable size and can be processed efficiently in memory.
- **Retrieval Mechanism**: For simplicity, the entire T&C text is used as the context for RAG. In a more advanced setup, a dedicated retrieval system could be implemented to extract relevant sections based on user queries.
- **API Integration**: The tool relies on Groq’s ChatGroq model for both T&C analysis and chatbot responses. It assumes that the API key provided is valid and that the model is capable of handling the tasks as described.
- **User Interaction**: The interface assumes basic user interaction, where users will input text and queries through the provided fields and buttons.

## Example

1. **Analyzing T&C**:
   - Input: “The company may terminate the agreement with 30 days' notice.”
   - Output: Identifies potential issues with the termination clause and provides recommendations.

2. **Chat with Janie**:
   - Query: “What are the data ownership rights in this T&C?”
   - Response: Provides information based on the analyzed T&C text regarding data ownership rights.

## Troubleshooting

- **Error Handling**: If an error occurs, ensure that the API key is correctly entered and that the T&C text is valid. Check the error messages for guidance on resolving issues.

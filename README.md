# Grievance ChatBot

Welcome to the Grievance ChatBot GitHub repository! This ChatBot is designed to provide support and answers related to CPGRAMS (Centralised Public Grievance Redress and Monitoring System).

## Introduction

The Grievance ChatBot leverages advanced technologies such as streamlit, torch, langchain, and more to offer an interactive and intelligent conversational experience. It utilizes a combination of language models, embeddings, and vector stores to understand and respond to user queries.

## Flowchart
![cpgram](https://github.com/AdityaJ9801/Ai_Chatbot-CPGRAM-/assets/124603391/2e4182c8-0699-461c-b471-aa85d963bd8c)

## Installation

To get started, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/AdityaJ9801/Ai_Chatbot-CPGRAM-.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Install Cuda dependences if running on local pc with gpu:
    ```bash
   https://pytorch.org/get-started/locally/
    ```
4. download quantized Mistral7b model:
   ```bash
    streamlit run app.py
    ```
5. Run the ChatBot:
    ```bash
    streamlit run app.py
    ```
## Usage

The ChatBot responds to queries related to CPGRAMS. You can interact with it by asking questions or exploring predefined buttons like "What is CPGRAM?" or "How to fill grievance form?".

## Dependencies

- streamlit
- torch
- langchain
- streamlit_chat
- Pillow
- better_profanity

## Features

### 1. Intelligent Conversations
The ChatBot engages in intelligent conversations, providing relevant and context-aware responses to user queries.

### 2. Offensive Language Detection
Utilizes the `better_profanity` library to detect and handle offensive language, ensuring a respectful and safe interaction environment.

### 3. GPU/CPU Compatibility
Checks for GPU availability using PyTorch, allowing seamless execution on GPU if available, or falling back to CPU.

### 4. Streamlit Interface
Employs the Streamlit framework to create a user-friendly interface, making it easy for users to interact with the ChatBot.

### 5. Conversation History
Maintains a conversation history that is displayed using Streamlit messages, providing users with a contextual view of the ongoing conversation.

## Code Structure

The project follows a modular structure for better organization. Here's an overview of the code structure:

```plaintext
📦 BreadcrumbsAi_Chatbot-CPGRAM-
 ┣ 📂 Vector_Data
 ┣ 📜 app.py
 ┣ 📜 chatbot.png
 ┣ 📜 README.md
 ┣ 📜 Mistral7b.gguf

# Grievance ChatBot

Welcome to the Grievance ChatBot GitHub repository! This ChatBot is designed to provide support and answers related to CPGRAMS (Centralised Public Grievance Redress and Monitoring System).

## Overview

The Grievance ChatBot leverages cutting-edge technologies such as Streamlit, PyTorch, and LangChain to deliver an interactive and intelligent conversational experience. It incorporates sophisticated language models, embeddings, and vector stores to comprehend and respond effectively to user queries.

## Flowchart
![CPGRAM Flowchart](https://github.com/AdityaJ9801/Ai_Chatbot-CPGRAM-/assets/124603391/2e4182c8-0699-461c-b471-aa85d963bd8c)

## Installation

To get started, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/AdityaJ9801/Ai_Chatbot_CPGRAM.git
    cd Ai_Chatbot_CPGRAM
    ```
2. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4. Install CUDA dependencies if using a local GPU:
    ```bash
    # Follow instructions at https://pytorch.org/get-started/locally/
    ```
5. Download the quantized Mistral7b model (provide instructions or link).

6. Run the ChatBot:
    ```bash
    streamlit run app.py
    ```

## Usage

The ChatBot responds to queries related to CPGRAMS. Interact by asking questions or exploring predefined buttons like "What is CPGRAM?" or "How to fill grievance form?".

## Features

1. **Intelligent Conversations**
   - Engages in intelligent conversations, providing context-aware responses to user queries.

2. **Offensive Language Detection**
   - Utilizes the `better_profanity` library to detect and handle offensive language, ensuring a respectful interaction environment.

3. **GPU/CPU Compatibility**
   - Checks for GPU availability using PyTorch, seamlessly switching to CPU if a GPU is unavailable.

4. **Streamlit Interface**
   - Employs the Streamlit framework to create a user-friendly interface, simplifying user interactions.

5. **Conversation History**
   - Maintains a conversation history displayed using Streamlit messages, providing users with contextual views of ongoing conversations.

## Code Structure

The project follows a modular structure for better organization. Here's an overview of the code structure:

```plaintext
📦 Ai_Chatbot-CPGRAM-
 ┣ 📂 Vector_Data
 ┣ 📜 app.py
 ┣ 📜 chatbot.png
 ┣ 📜 README.md
 ┣ 📜 Mistral7b.gguf

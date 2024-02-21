# Grievance ChatBot

Welcome to the Grievance ChatBot GitHub repository! This ChatBot is designed to provide support and answers related to CPGRAMS (Centralised Public Grievance Redress and Monitoring System).

## Introduction

The Grievance ChatBot leverages advanced technologies such as streamlit, torch, langchain, and more to offer an interactive and intelligent conversational experience. It utilizes a combination of language models, embeddings, and vector stores to understand and respond to user queries.

## Installation

To get started, follow these steps:

1. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
2. Clone the repository:
    ```bash
    https://github.com/AdityaJ9801/Ai_Chatbot-CPGRAM-.git
    ```
3. Install Cuda dependences if running on local pc with gpu:
    ```bash
   https://pytorch.org/get-started/locally/
    ```
4. download quantized Mistral7b model:
   '''bash    '''
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

## Code Structure

The project follows a modular structure for better organization. Here's an overview of the code structure:

```plaintext
📦 BreadcrumbsAi_Chatbot-CPGRAM-
 ┣ 📂 Vector_data
 ┣ 📜 app.py
 ┣ 📜 chatbot.png
 ┣ 📜 README.md
 ┣ 📜 Mistral7b.gguf
'''

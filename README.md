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
    .\venv\Scripts\activate  # On Mac: source venv/bin/activate
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4. Install CUDA dependencies if using a local GPU:
    ```bash
    # Follow instructions at https://pytorch.org/get-started/locally/
    ```
5. Download the quantized Mistral7b model:
    ```bash
    git lfs install
    git clone https://huggingface.co/Aditya757864/Mistral7B
    ```
6. Run the ChatBot:
    ```bash
    streamlit run app.py
    ```
## Code Structure

The project follows a modular structure for better organization. Here's an overview of the code structure:

```plaintext
📦 Ai_Chatbot-CPGRAM-
 ┣ 📂 Vector_Data
 ┣ 📜 app.py
 ┣ 📜 chatbot.png
 ┣ 📜 README.md
 ┣ 📜 Mistral7b.gguf
```
## Usage

The Grievance ChatBot serves as a user-friendly and efficient virtual assistant for citizens engaging with the CPGRAMS portal. Designed to streamline the grievance filing process, the chatbot leverages advanced language models and embeddings, notably the Mistral-7B-Instruct model, to provide Ministry-specific assistance. Users can interact seamlessly through the Streamlit interface, obtaining instant and accurate responses to common queries related to grievance submission. With a robust dataset from CPGRAMS, the chatbot's knowledge base ensures comprehensive support. The project's modular architecture, incorporating technologies like FAISS for vector storage, makes it adaptable for further enhancements. Ultimately, the Grievance ChatBot enhances user experience, simplifying interactions with CPGRAMS and expediting grievance resolution for citizens.


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

# Language Model (LLM) - Mistral 7B-Instruct 
<p align="center">
<img src="https://github.com/AdityaJ9801/Ai_Chatbot_CPGRAM/assets/124603391/968033b7-5042-405d-b687-016bd97a0047" alt="mistral-7b-v0" width="400">
</p>

## Overview
The Grievance ChatBot project harnesses the power of Mistral 7B-Instruct, a state-of-the-art language model (LLM) designed for advanced question-answering capabilities. This section provides an in-depth understanding of the Mistral 7B-Instruct model, its strategic quantization, and its crucial role within the Grievance ChatBot project.

## Mistral 7B-Instruct Model
Mistral 7B-Instruct is a language model renowned for its proficiency in handling instructional language. This model is specifically tailored for interpreting user queries and generating accurate responses within the contextual domain of CPGRAMS (Centralised Public Grievance Redress and Monitoring System).

### Quantization
The Grievance ChatBot project strategically employs the quantized version of the Mistral 7B-Instruct model (mistral-7b-instruct-v0.1.Q4_K_M.gguf). Quantization involves reducing the precision of the model's weights, resulting in enhanced memory efficiency and faster execution. This optimization is particularly crucial for real-time applications like the Grievance ChatBot, where responsiveness is paramount.

### Reasons for Special Use
The selection of the Mistral 7B-Instruct model is grounded in its specialized capabilities for handling instructional language. Given the nature of user queries related to grievance filing and CPGRAMS procedures, a model fine-tuned for instructive contexts aligns seamlessly with the project's objectives. Mistral 7B-Instruct excels in comprehending and generating responses in instructional and informational scenarios, making it an ideal choice for the chatbot's domain.

### Model Requirements
- **Name:** mistral-7b-instruct-v0.2.Q4_K_S.gguf
- **Quant Method:** Q4_K_S
- **Bits:** 4
- **Size:** 4.14
- **Max RAM Required:** 6.64 GB
- **Use Case:** Small with greater quality loss

## Model Comparison Chart
<p align="center">
<img src="https://github.com/AdityaJ9801/Ai_Chatbot_CPGRAM/assets/124603391/78ef72b8-ce7a-4a15-a3fe-1c34b63e310c" alt="mistral-7b-v0" width="400">
</p>


## Quantization Benefits
Quantizing the Mistral 7B-Instruct model brings several advantages:
- **Memory Efficiency:** Reduced precision leads to lower memory usage, enabling smoother deployment and execution.
- **Faster Inference:** Quantization enhances inference speed, contributing to a more responsive user experience in real-time interactions.

## References
1. [Langchain Documentation - Getting Started](https://python.langchain.com/docs/get_started/introduction)
2. [Langchain Documentation - Vector Stores with FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss)
3. [Mistral-7B-Instruct Model Paper](https://mistral.ai/assets/Mistral_7B_paper_v_0_1.pdf)
4. [Anaconda Blog - How to Build a Retrieval-Augmented Generation Chatbot](https://www.anaconda.com/blog/how-to-build-a-retrieval-augmented-generation-chatbot)




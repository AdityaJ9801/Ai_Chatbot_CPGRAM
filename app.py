import streamlit as st 
import torch 
from langchain_community.llms import LlamaCpp
from langchain import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.chains import RetrievalQA
from streamlit_chat import message
from PIL import Image
from better_profanity import profanity

def is_offensive(text):
    """
    Check if the given text contains offensive language using better-profanity.
    Returns True if offensive, False otherwise.
    """
    return profanity.contains_profanity(text)

# Use the is_offensive() function in your main() function

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using GPU:", torch.cuda.get_device_name(device))
    print("GPU index:", device.index)
else:
    print("Using CPU")
icon = Image.open("chatbot.png")
icon = icon.resize((64, 64)) # You can adjust the size as per your requirement
st.set_page_config(page_title="Grievance ChatBot", page_icon=icon)

custom_prompt_template = """Always Answer the following QUESTION based on the CONTEXT given and make sure that the answer is in bullet points along with a few conversating lines related to the question. If the CONTEXT doesn't contain the answer, or the question is outside the domain of expertise for CPGRAMS (Centralised Public Grievance Redress and Monitoring System), politely respond with "I'm sorry, but I don't have any information on that topic in my database. However, I'm here to help with any other questions or concerns you may have regarding grievance issues or anything else! Feel free to ask, and let's work together to find a solution. Your satisfaction is my priority!"

context : {context}

question : {question}

"""
def set_custom_prompt():
    """Prompt template for QA retrieval for each vector store"""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', "question"])
    return prompt

@st.cache_resource
def qa_llm():
    llm = LlamaCpp(
        streaming=True,
        model_path="Mistral7B/mistral-7b-instruct-v0.1.Q4_K_M .gguf",
        temperature=0.5,
        top_p=1,
        verbose=True,
        n_ctx=4096,
        device=device  # Set the device for the LLM
    )
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("Vector_Data",embeddings)
    prompt = set_custom_prompt()
    retriever = db.as_retriever(search_kwargs ={'k':2})
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    qa_llm.clear()  # Call the clear() method of the cached function
    return answer

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

def initialize_session_state():
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me any queries related to Grievance and CPGRAM Portal.."]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]

def main():
    st.title("🤖 CPGRAM Grievance Chatbot")
    initialize_session_state()
    reply_container = st.container()
    user_input = st.chat_input(placeholder="Please describe your queries here...", key='input')
    
    if st.button("What is CPGRAM?", key="cpram_button"):
        st.session_state['past'].append("What is CPGRAM?")
        with st.spinner('Generating response...'):
            answer = process_answer({'query': "What is CPGRAM?"})
        st.session_state['generated'].append(answer)
    elif st.button("How to fill grievance form?", key="grievance_button"):
        st.session_state['past'].append("How to fill grievance form?")
        with st.spinner('Generating response...'):
            answer = process_answer({'query': "How to fill grievance form?"})
        st.session_state['generated'].append(answer)
    elif user_input:
        if is_offensive(user_input):
            st.session_state['past'].append("User input flagged as offensive")
            st.session_state['generated'].append("I'm sorry, but I can't assist with offensive content.")
        else:
            st.session_state['past'].append(user_input)
            with st.spinner('Generating response...'):
                answer = process_answer({'query': user_input})
            st.session_state['generated'].append(answer)

    if st.session_state["generated"]:
        if st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))

  
if __name__ == '__main__':
    main()

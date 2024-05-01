import streamlit as st
import base64
import os
import requests

# App title
st.set_page_config(page_title="Llama 3 Chatbot")

with st.sidebar:
    st.title('Llama 3 Chatbot')
    advanced_rag = st.toggle('Advanced RAG')

    if advanced_rag:
        st.write('Using Advanced RAG')

    llm_url = st.text_input('Enter LLM URL:')
    if len(llm_url) == 0:
        st.warning('Please enter LLM URL!', icon='⚠️')
    api_url = st.text_input('Enter API URL:')
    if len(api_url) == 0:
        st.warning('Please enter API URL!', icon='⚠️')
    if len(llm_url) != 0 and len(api_url) != 0:
        st.success('Proceed to entering your prompt message!', icon='👉')
    #pdf_file = st.file_uploader("Upload PDF file ", type=("pdf"))

st.title("💬 Llama 3 Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    url = 'https://www.w3schools.com/python/demopage.php'
    payload = {'query': prompt, 'llm_url': llm_url}

    if not advanced_rag:
        response = requests.post(api_url + "/simple_rag/", json=payload).json()
        print(response)
        st.chat_message("assistant").write(response["answer"])
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    else:
        response = requests.post(api_url + "/advanced_rag/", json=payload).json()
        print(response)
        st.chat_message("assistant").write(response["answer"])
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
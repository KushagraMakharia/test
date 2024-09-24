import streamlit as st
import random
import time
from rag_llm import RagLLM

rag_llm = RagLLM("Huyen C. Designing Machine Learning Systems...2022.pdf")
qa_llm = rag_llm.create_qa_chain()

def response_generator(text):
    for chunk in qa_llm.invoke(text):
        yield chunk

    # response = random.choice(
    #     [
    #         "Hello there! How can I assist you today?",
    #         "Hi, human! Is there anything I can help you with?",
    #         "Do you need help?",
    #     ]
    # )
    # for word in response.split():
    #     yield word + " "
    #     time.sleep(0.05)


st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
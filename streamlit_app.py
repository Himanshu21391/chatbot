from app import ChatBot
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def generate_response(input):
    logging.debug(f"User input: {input}")
    result = bot.rag_chain.invoke(input)
    logging.debug(f"Raw result: {result}")
    if "Answer:" in result:
        return result.split("Answer:")[1].strip()
    else:
        return result.strip()

# Use logging in other parts as necessary

bot = ChatBot()
st.set_page_config(page_title="ChatBot")
with st.sidebar:
    st.title('Chatbot')



# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to ChatBot! How can I assist you in your electronics journey today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer from mystery stuff.."):
            response = generate_response(input)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

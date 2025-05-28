import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# Streamlit App Title
st.title("🤖 Welcome To Kartik ChatBot")

# Unique session ID
session_id = "chat123"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = ChatMessageHistory()

# Chat input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:")
    submit = st.form_submit_button("Send")

# Process user input
if submit and user_input:
    # Add user message
    st.session_state.history.add_user_message(user_input)

    # Create chat chain with memory
    chat_chain = prompt | llm
    chat_with_memory = RunnableWithMessageHistory(
        chat_chain,
        lambda session_id: st.session_state.history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # Get LLM response
    response = chat_with_memory.invoke(
        {"question": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    bot_reply = response.content if hasattr(response, "content") else str(response)

    # Add bot message
    st.session_state.history.add_ai_message(bot_reply)

    # Track messages for display
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "bot", "content": bot_reply})

# Set avatar URLs
user_logo = "https://cdn-icons-png.flaticon.com/512/9131/9131529.png"
bot_logo = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"

# Display messages with avatars
for msg in st.session_state.messages:
    if msg["role"] == "user":
        col1, col2 = st.columns([1, 10])
        with col1:
            st.image(user_logo, width=30)
        with col2:
            st.markdown(f"**You:** {msg['content']}")
    else:
        col1, col2 = st.columns([1, 10])
        with col1:
            st.image(bot_logo, width=30)
        with col2:
            st.markdown(f"**Bot:** {msg['content']}")

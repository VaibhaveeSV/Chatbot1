import streamlit as st
import requests

st.set_page_config(page_title="Resolute.ai FAQ Chatbot", layout="centered")
st.title("Resolute.ai ChatBot")

st.markdown("Ask any question")

question = st.text_input("Your Question:")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        try:
            response = requests.post("http://127.0.0.1:8000/chat", json={"question": question})
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.success("Answer:")
                st.write(answer)
            else:
                st.error("Something went wrong while getting a response from the server.")
        except Exception as e:
            st.error(f"Error: {e}")
import os
import openai

import streamlit as st
from langchain.llms import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Playing around with LangChain")
prompt = st.text_input('Enter prompt here')

llm = OpenAI(temperature=0.9, max_tokens=100)

if prompt:
    resp = llm(prompt)
    st.write(resp)
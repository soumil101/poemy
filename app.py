import os
import openai

import streamlit as st
from langchain.llms import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Playing around with LangChain")
prompt = st.text_input('Enter prompt here')
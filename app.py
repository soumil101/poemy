import os
import openai

import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

openai.api_key = os.getenv("OPENAI_API_KEY")

# app framework
st.title("Playing around with LangChain")
st.title("Poem to Haiku")
prompt = st.text_input('Pick a topic')

# templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='write a poem about {topic}'
)
script_template = PromptTemplate(
    input_variables = ['poem'],
    template='now turn {poem} into a haiku'
)

# llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='poem')
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='haiku')
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['poem', 'haiku'], verbose=True)

# show output on screen
if prompt:
    resp = sequential_chain({'topic': prompt})
    st.markdown('---')

    st.write('**Here is your poem:**')
    poem = resp['poem']
    st.write(poem)
    st.markdown('---')

    st.write('**Here is your haiku:**')
    haiku = resp['haiku']
    st.write(haiku)
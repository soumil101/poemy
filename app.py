import os
import openai
from dotenv import load_dotenv
import textwrap

import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title='Poemy', page_icon='🪶', layout='wide')

# app framework
st.title("Meet Poemy")
st.subheader("Poemy will generate a poem and a haiku for you. Just give it a topic.")
prompt = st.text_input('Pick a topic')

# templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='write a poem about {topic}'
)
script_template = PromptTemplate(
    input_variables = ['poem'],
    template='now turn {poem} into a haiku with 5-7-5 syllables'
)

# memory
poem_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
haiku_memory = ConversationBufferMemory(input_key='poem', memory_key='chat_history')

# llms
llm = OpenAI(temperature=0.9)
poem_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='poem', memory=poem_memory)
haiku_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='haiku', memory=haiku_memory)

def format_haiku(haiku):
    # Split the haiku into lines with 5, 7, and 5 syllables
    syllable_counts = [5, 7, 5]
    haiku_lines = []
    start = 0

    for count in syllable_counts:
        end = start + count
        haiku_lines.append(haiku[start:end])
        start = end

    # Join the lines with line breaks
    formatted_haiku = "\n".join(haiku_lines)

    return formatted_haiku

# show output on screen
if prompt:   
    st.markdown('---')

    poem = poem_chain.run(topic=prompt)
    haiku = haiku_chain.run(poem=poem)

    st.write('**Here is your poem:**')
    st.write(poem)

    st.markdown('---')

    st.write('**Here is your haiku:**')
    formatted_haiku = format_haiku(haiku)
    st.write(haiku)

st.markdown('---')

"built with langchain"
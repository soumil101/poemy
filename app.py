import os
import openai

import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

openai.api_key = os.getenv("OPENAI_API_KEY")

# app framework
st.title("Playing around with LangChain")
st.title("Poem and Haiku")
prompt = st.text_input('Pick a topic')

# templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='write a poem about {topic}'
)
script_template = PromptTemplate(
    input_variables = ['poem', 'wikipedia_research'],
    template='now turn {poem} into a haiku while leveraging this wikipedia research: {wikipedia_research}'
)

# memory
poem_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
haiku_memory = ConversationBufferMemory(input_key='poem', memory_key='chat_history')

# llms
llm = OpenAI(temperature=0.9)
poem_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='poem', memory=poem_memory)
haiku_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='haiku', memory=haiku_memory)

wiki = WikipediaAPIWrapper()

# show output on screen
if prompt:   
    st.markdown('---')

    poem = poem_chain.run(topic=prompt)
    wikipedia_research = wiki.run(prompt)
    haiku = haiku_chain.run(poem=poem, wikipedia_research=wikipedia_research)

    st.write('**Here is your poem:**')
    st.write(poem)

    st.markdown('---')

    st.write('**Here is your haiku:**')
    st.write(haiku)

    with st.expander("Poem History"):
        st.info(poem_memory.buffer)

    with st.expander("Haiku History"):
        st.info(haiku_memory.buffer)
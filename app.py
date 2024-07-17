import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate

import os 
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking 

os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']="Simple Q&A Chatbot with OPENAI"

##PromptTemplate

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistance please response to user Query"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens):
    
    openai.api_key=api_key
    #llm=ChatOpenAI(model=llm)
    llm = ChatOpenAI(model=llm, openai_api_key=api_key, temperature=temperature, max_tokens=max_tokens)

    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

## Title of the app
st.title('Chat bot for Q&A')

##Sidebar settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Open AI API key:",type="password")


## Drop down to select various Openai Model

llm=st.sidebar.selectbox("Select an Open AI model",["gpt-4o","gpt-4-turbo","gpt-4"])
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_token=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)


## Main Interface for user Input 

st.write('Go ahead and ask any Question')
user_input=st.text_input("You:")

if user_input:
    print(api_key)
    response=generate_response(user_input,api_key,llm,temperature,max_token)
    st.write(response)
    
    
else :
    st.write('Please provide the Query')




    
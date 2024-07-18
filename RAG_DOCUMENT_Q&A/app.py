import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai


from dotenv import load_dotenv
load_dotenv()

##load the Groq API
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the Question based on the Provided context only.
    Please provide the Most accurate response based on the Question
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader('/research_papers') ## Data Ingestion Step 
        st.session_state.docs=st.session_state.loader.load()  ## Complete Document loading 
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


user_prompt=st.text('Enter you Query from the research Paper')

if st.button("Docuemnt Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")
    
import time 

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    #retriver=st.session_state.vectors.as_retriver()  #Query from Vector DB
    retriever=st.session_state.vectors.as_retriever()
    retriver_chain=create_retrieval_chain(retriever,document_chain)
    
    
    start=time.process_time()
    response=retriver_chain.invoke({'input':user_prompt})
    print(f'Response time:{time.process_time()-start}')
    
    st.write(response['answer'])
    
    ## With a streamlit expander
    
    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response['contex']):
            st.write(doc.page_content)
            st.write('--------------')
    
    




import os
import subprocess
import json
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai

#Declaration of variables
global my_key
cwd=os.getcwd()#Current working Directory
tika_jar_path=os.path.join("tika_jar_file","tika-app-2.9.3.jar")#Relative path to tika jar file
file_path = os.path.join(cwd,"en_GC.pdf")#Path to PDF file
file_title = file_path.split("/")[-1]
book = f"{file_title}_faiss_index"
# Initialize FAISS index
dimension = 384  # embedding dimension for all-MiniLM-L6-v2
index = faiss.IndexFlatIP(dimension)  # Inner Product (used for cosine similarity)
metadata_store = []  # Will store tuples of (id, text, metadata)
# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

headers={
                  "authorization":st.secrets["AUTH_TOKEN"],"content-type":"application/json"
                  }

#Set UI
st.sidebar.title("RAG-Demo")


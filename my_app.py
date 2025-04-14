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


st.sidebar.title("RAG-Demo")


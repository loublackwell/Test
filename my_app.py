import os
import subprocess
import json
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
#from google import genai

st.sidebar.title("RAG-Demo")


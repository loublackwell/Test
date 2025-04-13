import os
import subprocess
import json
from google import genai
from tika import parser
from tika import initVM, parser
# Explicitly initialize Java VM first
initVM()
import chromadb
from sentence_transformers import SentenceTransformer

import streamlit as st
st.sidebar.title("RAG-Demo")


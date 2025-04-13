import os
import subprocess
import json
# Explicitly initialize Java VM first
initVM()
import chromadb
from sentence_transformers import SentenceTransformer

import streamlit as st
st.sidebar.title("RAG-Demo")


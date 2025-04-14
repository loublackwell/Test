import os
import subprocess
import json
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
#import google.generativeai as genai
from google import genai

#Declaration of variables
global my_key
global book
global cwd

cwd=os.getcwd()#Current working Directory
tika_jar_path=os.path.join("tika_jar_file","tika-app-2.9.3.jar")#Relative path to tika jar file
file_path = os.path.join(cwd,"en_GC.pdf")#Path to PDF file
file_title = file_path.split("/")[-1]
book = f"{file_title}_faiss_index"
st.write(book)
# Initialize FAISS index
dimension = 384  # embedding dimension for all-MiniLM-L6-v2
index = faiss.IndexFlatIP(dimension)  # Inner Product (used for cosine similarity)
metadata_store = []  # Will store tuples of (id, text, metadata)
# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

my_key={
                 "authorization":st.secrets["API_KEY"],"content-type":"application/json"
                 }




# ---- Core Functions ---- #

def extract_text_with_tika_jar(file_path, tika_jar_path):
    try:
        result = subprocess.run(
            ['java', '-jar', tika_jar_path, '-t', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except Exception as e:
        print(f"Error running Tika: {e}")
        return ""
    

def index_texts(text_list):
    global index, metadata_store
    embeddings = []

    for pos, text in enumerate(text_list):
        print(f"{pos+1}/{len(text_list)} {text[0:30]}...")
        if text.strip() != "":
            text_id = str(pos)
            embedding = model.encode(text)
            embeddings.append(embedding)
            metadata_store.append((text_id, text, {"text": text}))

    if embeddings:
        embeddings_array = np.array(embeddings).astype('float32')
        index.add(embeddings_array)
        print(f"Indexing completed. Total texts indexed: {len(metadata_store)}")


def save_index():
    faiss.write_index(index, f"{book}.index")
    with open(f"{book}_metadata.json", "w") as f:
        json.dump(metadata_store, f)


def load_index():
    global index, metadata_store
    index = faiss.read_index(f"{book}.index")
    with open(f"{book}_metadata.json", "r") as f:
        metadata_store = json.load(f)


def query_texts(query_text, top_k=5):
    query_embedding = model.encode(query_text).reshape(1, -1).astype('float32')
    distances, indices = index.search(query_embedding, top_k)

    ids = []
    documents = []
    metadatas = []
    dists = []

    for i in indices[0]:
        ids.append(metadata_store[i][0])
        documents.append(metadata_store[i][1])
        metadatas.append(metadata_store[i][2])
    for d in distances[0]:
        dists.append(d)

    results = {
        "ids": [ids],
        "metadatas": [metadatas],
        "documents": [documents],
        "distances": [dists]
    }

    verse_dict = {}
    for i in range(len(ids)):
        verse_dict[documents[i]] = ids[i]

    return results, documents, ids, verse_dict


def query_gemini(task):
    #Query LLM
    try:
        client = genai.Client(api_key=my_key)

        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=task
        )
        TEXT=response.text
    except Exception as e:
        st.write(f"Unable to query llm: {e}")
        response=""
        TEXT=""
    return TEXT




def build_prompt(expert,verses):
    task=f"""{expert} who has been given the following task:
             Based on the following question:
             Question:
             {question}
             
             Which one of the following text best answer and/or align with the question or statement? 
             If there are none
             return, "No answer found!"
             Place your answer in the following python dictionary format:
             
             CONSIDERATIONS:
             1. No Hallucinations allowed. Stick with completing the task given the provided context.
             2. If there are no relevant text, return no relevant text
             3. Only use the list of text to serach for answers
             4. The output should be a valid python dictionary
             
             
             OUTPUT FORMAT:
             ```
             {{"ANSWER":[<insert any matching answers here]}}
             ```

             LIST OF TEXT:
             {verses}
          """
    return task


def conlcusion(question,answers):
    task2=f"""Review the list of potentials answers to the following question and see if there is answer that 
             can be found from the list. If there are answers found, summarize in no more  than 5 sentences.
             QUESTION:
             {question}. 
             POTENTIAL ANSWERS:
             {answers}

             CONSIDERATIONS:
              1. No Hallucinations allowed. Stick with completing the task given the provided context.
              2. If there are no relevant text, return no relevant text
              3. Only use the list of text to serach for answers
              4. The output should be a valid python dictionary
              5. Do not provide/derive any answers that were not originally mentioned in the potential answers
              5. List the texts that you used to come to the answers.

              OUTPUT FORMAT:
                 ```
                 {{"ANSWER":[<insert summary here],"JUSTIFICATION":[<list of text that you used to get the answer]}}

            """
    return task2


def parse_query(out,verse_dict):
    dict_block={}
    report_dict={}
    answers=[]
    start=out.find("```")
    start_block=out[start+3:]
    if start>=0:
        end_block=start_block.find("```")
        if end_block>=0:
            middle_block=start_block[:end_block]
            json_start=middle_block.find("{")
            if json_start>=0:
                json_end=middle_block.rfind("}")
                if json_end>=0:
                    dict_block=eval(middle_block[json_start:json_end+1])
                    answers=dict_block.get('ANSWER')
                    if answers!=None:
            
                        for text,verse in verse_dict.items():
                            if text in answers:
                                report_dict[verse]=text
                
    return dict_block,answers,report_dict


# ---- Example Usage ---- #

# Step 1: Extract and index (run only once)
#nlines = 2
#out=extract_text_with_tika_jar(file_path, tika_jar_path)
#text_blocks = out.split('\n' * nlines)


#index_texts(text_blocks)
#save_index()



#Set UI
st.sidebar.title("RAG-Demo")


task="how old is the python language"
A=query_gemini(task)
st.write(A)


# Step 2: Load saved index
expert="""You are a Seventh Day adventist theologian with a PHD in Divinity from Andrews Univeristy in Michigan.
          You also specialize in Ellen G. Whites Writings.  
            """

"""
load_index()
answers_with_ids=[]
# Step 3: Ask a question
question = "who is the antichrist"
results, verses, IDS, verse_dict = query_texts(question, top_k=25)

task=build_prompt(expert,verses)
out=query_gemini(task)
st.write(out)
#print(out)
llm_dict1,answers1,report_dict1=parse_query(out,verse_dict)
for key,value in report_dict1.items():
    ID=f"{key}. {value}"
    #ID={"id":key,"text":value}
    answers_with_ids.append(ID)
task2=conlcusion(question,answers_with_ids)
out=query_gemini(task2)
llm_dict2,answers2,report_dict2=parse_query(out,verse_dict)
st.write(llm_dict2)
"""

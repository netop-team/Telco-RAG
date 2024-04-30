import os
import traceback
from Source.embeddings import get_embeddings
from Source.query import Query
from Source.generate import generate
import sys
import traceback
from Source.input import get_documents
from Source.chunking import chunk_doc
import git
import random
import json
api_key = "your-api-key"

folder_url = "https://huggingface.co/datasets/netop/Embeddings3GPP-R18"
clone_directory = "./3GPP-Release18"

if not os.path.exists(clone_directory):
    git.Repo.clone_from(folder_url, clone_directory)
    print("Folder cloned successfully!")
else:
    print("Folder already exists. Skipping cloning.")

def choose_random_question(data):
    while True:
        random_question = random.choice(list(data.values()))
        
        if '3GPP' in random_question['question']: 
            return random_question['question']
        else:
            continue

def TelcoRAG(query, api_key= api_key):
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
        question = Query(query, [])
        
        question.def_TA_question()

        question.predict_wg()

        document_ds = get_documents(question.wg)

        print(len(document_ds))
        # Chunk documents based on provided chunk size and overlap
        Document_ds = [chunk_doc(doc) for doc in document_ds]
        

        series_doc = {'Summaries':[]}
        for series_number in question.wg:
            series_doc[f'Series{series_number}'] = []
            for doc in Document_ds:
                if doc[0]['source'][:2].isnumeric():
                    if int(doc[0]['source'][:2]) == series_number:
                        series_doc[f'Series{series_number}'].append(doc)
                else:
                    if doc not in series_doc['Summaries']:
                        series_doc['Summaries'].append(doc)
        
        series_docs = get_embeddings(series_doc)

        embedded_docs = []
        for serie in series_docs.values():
            embedded_docs.extend([serie])
        question.get_question_context_faiss(batch=embedded_docs, k=20, use_context=False)
            
        question.candidate_answers()

        old_list =  question.wg
        question.predict_wg()
        new_series = {}
        for series_number in question.wg:
            if series_number not in old_list:
                new_series[f'Series{series_number}'] = []
                for doc in Document_ds:
                    if doc[0]['source'][:2].isnumeric():
                        if int(doc[0]['source'][:2]) == series_number:
                            new_series[f'Series{series_number}'].append(doc)
        new_series = get_embeddings(new_series)
        old_series={'Summaries': series_docs['Summaries']}
        for series_number in question.wg:
            if series_number in old_list:
                old_series[f'Series{series_number}'] = series_docs[f'Series{series_number}']
        embedded_docs = []
        for serie in new_series.values():
            embedded_docs.extend([serie])
        for serie in old_series.values():
            embedded_docs.extend([serie])
        question.get_question_context_faiss(batch=embedded_docs, k=20, use_context=False)

        response, context, query = generate(question)
        return response, context, query

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    try:
        with open("TeleQnA.json", "r") as file:
            questions= json.load(file)
        question = choose_random_question(questions)
        response= TelcoRAG(question)
        print(f"""Generated response to the question:
              {response[2]}
Is:
               {response[0]} """)
    except Exception as e: 
        print("Encountered an error and moving to the next case.")
        print(traceback.format_exc())
        


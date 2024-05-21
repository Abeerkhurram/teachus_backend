from fastapi import FastAPI, File, UploadFile
import os
from langchain_groq import ChatGroq
from functions import split_data,get_vectorstore,get_existing_vectorstore,retrieval_QA



app = FastAPI()

# Initialize model
llm = ChatGroq(temperature=0,
                groq_api_key="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",
                model_name="mixtral-8x7b-32768")
docs_folder_path = "backend_docs"

if not os.path.exists(docs_folder_path):
    os.makedirs(docs_folder_path)

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(docs_folder_path, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return {"result": "File uploaded successfully"}
    except Exception as e:
        return {"error": f"Failed to upload file: {e}"}

@app.get("/insert_data_into_vector_database/")
async def insert_data():
    if os.path.exists(docs_folder_path) and os.path.isdir(docs_folder_path):
        files_in_docs = os.listdir(docs_folder_path)
        if files_in_docs:
            file = files_in_docs[0]
            print(str(file))
            # Define split_data, get_vectorstore, and save_local methods
            docs = split_data(os.path.join(docs_folder_path, file))
            #print(len(docs))
            vectordb = get_vectorstore(docs)

            vectordb.save_local("VectorDb")
            return {"file": "data_injested"}
        else:
            return {"error": "The 'docs' folder is empty."}
    else:
        return {"error": "No docs folder exists"}

@app.post("/query/")
async def query(item: str):
    vector_docs = get_existing_vectorstore("VectorDb")
    chain = retrieval_QA(vector_docs)
    answer = chain({"question": item}, return_only_outputs=True)["answer"]
    return {"answer": answer}

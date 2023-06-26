from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from fastapi import FastAPI,UploadFile, File
from fastapi.responses import JSONResponse
app = FastAPI()
import uvicorn
import requests
import os
import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

# Create an instance of OpenAI language model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0.4)
from fastapi.middleware.cors import CORSMiddleware
# memory = ConversationSummaryBufferMemory(
#     llm=llm,
#     output_key='answer',
#     memory_key='chat_history',
#     return_messages=True)
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
memory={}
if not os.path.exists("faiss_db"):
    os.makedirs("faiss_db")

if not os.path.exists("files"):
    os.makedirs("files")
dbs = {}
prompt = {}
@app.post("/upload")
async def upload_file(user_id: str,file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open('files/file_{}.pdf'.format(user_id),'wb') as gf:
            gf.write(contents)
    except requests.exceptions.RequestException as err:
        return {"error": f"Error occurred during file upload: {str(err)}"}
    
    try:
        loader = PyPDFLoader("files/file_{}.pdf".format(user_id))
        pages = loader.load_and_split()
        # SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
        chunks = pages
        db_name = "db_{}".format(user_id)
        dbs[db_name] = FAISS.from_documents(chunks, embeddings)
        dbs[db_name].save_local("faiss_db/faiss_{}".format(db_name))
        # f"db_{user_id}" = FAISS.from_documents(chunks, embeddings)
    
        # f"db_{user_id}".save_local("faiss_index_{}".format(user_id))
        os.remove("files/file_{}.pdf".format(user_id))
        return {"filename": f"file uploaded successfully for user {user_id}"}
    except requests.exceptions.RequestException as err:
        return JSONResponse(status_code=400, content={"error": f"Error occurred during file upload: {str(err)}"})

# Definition of  API endpoint for asking question
@app.post("/askqa/{user_id}")
async def askqa(user_id:str,query: str = 'brief explanation of this case' ):
    try:
        if user_id not in memory:
            memory[user_id] =  ConversationSummaryBufferMemory(
            llm=llm,
            output_key='answer',
            memory_key='chat_history',
            return_messages=True)
        db_name = "faiss_db_" + user_id
        new_db = FAISS.load_local(f"faiss_db/{db_name}", embeddings)    
        # new_db = FAISS.load_local("faiss_index", embeddings)
    except FileNotFoundError as file_err:
        return JSONResponse(status_code=404, content={'error': f"File not found. for user {user_id} First upload the docs: {str(file_err)}"})

    # Create an instance of ConversationalRetrievalChain with the OpenAI LLM and the retriever
    qa = ConversationalRetrievalChain.from_llm(llm=llm,
    memory=memory[user_id],
    chain_type="map_rerank",
    retriever= new_db.as_retriever(search_kwargs={"k": 3, "include_metadata": True}),
    return_source_documents=True,
    get_chat_history=lambda h : h,
    verbose=False)
    try:
        prompt[user_id] = query
        response = qa({"question": prompt[user_id]})
        suggestion = qa({"question": "suggest me three questions similar to the previous questions"})
        suggestion = list(suggestion['answer'].split('?'))
        return {'answer': response['answer'],'suggestion':suggestion}
    except requests.exceptions.HTTPError as err:
        # Handle HTTP errors here
        return JSONResponse(status_code=404, content={'error': str(err)})
    

@app.get("/clearall-user/{user_id}")
async def clearall_user(user_id: str):
    try:
        db_name = "db_" + user_id
        shutil.rmtree(f"faiss_db/faiss_{db_name}")
        if user_id in memory:
            del memory[user_id]
            del dbs[db_name]
            del prompt[user_id]
        return JSONResponse(content={'message': f'Cache cleared for user {user_id}'})
    except FileNotFoundError as file_err:
        return JSONResponse(status_code=404, content={'error': f"Cache does not exist for user {user_id}: {str(file_err)}"})
@app.get("/clearall")
async def clear_all():
    try:
        shutil.rmtree("faiss_db")
        memory.clear()
        dbs.clear()
        return JSONResponse(content={'message': 'All caches cleared'})
    except FileNotFoundError as file_err:
        return JSONResponse(status_code=404, content={'error': f"All caches have already been cleared: {str(file_err)}"})
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
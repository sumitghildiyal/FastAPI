from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
import tempfile

app = FastAPI()

# Global variables
vector_store = None
retriever = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set your OpenAI API Key in environment variables.")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store, retriever

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Load and split PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    # Create embeddings and store in FAISS
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()

    os.remove(tmp_path)

    return {"message": "File processed and stored successfully."}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global retriever
    if not retriever:
        return JSONResponse(status_code=400, content={"error": "No document uploaded yet."})

    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.run(question)

    return {"question": question, "answer": result}

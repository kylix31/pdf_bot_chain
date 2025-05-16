import tempfile
import uuid
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from .chat_history import ChatHistory

app = FastAPI()
sessions: dict[str, dict[str, Any]] = {}

embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")
chat_model = ChatVertexAI(model_name="gemini-2.0-flash-lite-001", temperature=0)


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile):
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs_split = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(docs_split, embeddings)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),
        prompt=contextualize_q_prompt,
    )

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    session_id = str(uuid.uuid4())
    chat_history = ChatHistory()
    sessions[session_id] = {"chain": rag_chain, "chat_history": chat_history}

    return JSONResponse(
        content={
            "session_id": session_id,
        }
    )


@app.post("/ask_pdf/{session_id}")
async def ask_pdf(session_id: str, question: str = Form(...)):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    chain = session["chain"]
    chat_history = session["chat_history"]
    chat_history.add_user_message(question)
    response = chain.invoke(
        {"input": question, "chat_history": chat_history.get_history()}
    )
    chat_history.add_assistant_message(response.get("answer", ""))

    return JSONResponse(
        content={
            "session_id": session_id,
            "question": question,
            "answer": response.get("answer", ""),
            "chat_history": chat_history.get_history(),
        }
    )

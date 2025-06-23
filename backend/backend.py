import logging
import asyncio
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Query, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, constr
from jose import JWTError, jwt
from datetime import datetime, timedelta
import requests
import os
import tiktoken
import re
from bs4 import BeautifulSoup
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from db import (
    init_db,
    close_db,
    save_chat,
    get_chat_history as get_mongo_chat_history,
    get_user_sessions,
    get_user_by_email,
    create_user
)
from tools import(
    get_current_time, 
    get_weather, 
    calculator_tool, 
    search_google
)
from langchain_mongodb import MongoDBAtlasVectorSearch
from transformers import AutoTokenizer
from pymongo import MongoClient
import httpx
from contextlib import asynccontextmanager
from typing import Optional
import uuid
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI")
api_key = os.getenv("NVIDIA_API_KEY")
client = MongoClient(MONGODB_ATLAS_URI)
db = client["chatbot_vectors"]   

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

#------------------------------------------------------------------------

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(lifespan=lifespan)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        if user_email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": user_email}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def extract_text_from_url(url: str):
    try:
        logger.info(f"Extracting text from URL: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return "\n".join([line.get_text(strip=True) for line in soup.find_all(["h1", "h2", "h3", "p"])])
    except Exception as e:
        logger.error(f"URL extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"URL processing failed: {str(e)}")

async def extract_from_url(url: str):
    loop = asyncio.get_event_loop()
    result = loop.run_in_executor(None, extract_text_from_url, url)
    return await result

def extract_text_from_pdf(pdf_path: str):
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(e)}")


async def extract_from_pdf(pdf_path: str):
    loop = asyncio.get_event_loop()
    result = loop.run_in_executor(None, extract_text_from_pdf, pdf_path)
    return await result

def extract_text_from_image(image_path: str):
    """Extract text using OCR from both images and image-based PDF."""
    try:
        logger.info(f"Extracting text from image: {image_path}")
        ext = os.path.splitext(image_path)[1].lower()

        if ext == ".pdf":
            pages = convert_from_path(image_path)
            text = "\n".join([pytesseract.image_to_string(page) for page in pages])
        else:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
        
        return text if text.strip() else "No text found in the image/PDF."    
    
    except Exception as e:
        logger.error(f"Image extraction failed: {str(e)}")
        return f"Error extracting from image: {str(e)}"


async def extract_from_image(image_path: str):
    loop = asyncio.get_event_loop()
    result = loop.run_in_executor(None, extract_text_from_image, image_path)
    return await result


def split_and_store(text: str, source: str, session_id: str):
    try:
        logger.info(f"Splitting and storing data for session {session_id}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        collection = db["docs"]
        try:
            result = collection.delete_many({"session_id": session_id})
            logger.info(f"Deleted {result.deleted_count} existing documents for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to delete existing documents for session {session_id}: {e}")
        
        MongoDBAtlasVectorSearch.from_texts(
            texts=chunks,
            metadatas=[{"source": source, "session_id": session_id}] * len(chunks),
            embedding=embeddings,
            collection=collection,
            index_name="vector_index",
        )
        doc_count = collection.count_documents({"session_id": session_id})
        logger.info(f"Successfully stored {doc_count} document chunks for session {session_id}")
    except Exception as e:
        logger.error(f"Vector storage failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vector storage failed: {str(e)}")


class QueryReq(BaseModel):
    session_id: str
    question: str

class UserInfo(BaseModel):
    email: EmailStr
    password: constr(min_length=8)  # type: ignore


@app.post("/load")
async def load_data(
    session_id: str = Form(...),
    mode: int = Form(...),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    logger.info(f"Load requested: mode={mode}, session={session_id}, user={user_id}")

    try:
        text = ""
        source = ""

        if mode == 3:  # URL mode
            if not url:
                raise HTTPException(status_code=400, detail="URL is required for URL mode")
            text = await extract_from_url(url)
            source = url

        else:  # File mode
            if not file:
                raise HTTPException(status_code=400, detail="File is required for file mode")

            base, ext = os.path.splitext(file.filename or "")
            ext = ext.lower()
            if not ext:
                ext = ".pdf" if mode == 1 else ".png"
            if ext == ".jpeg":
                ext = ".jpg"

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            try:
                content = await file.read()
                temp_file.write(content)
                temp_file.close()

                if mode == 1:  # PDF
                    text = await extract_from_pdf(temp_file.name)
                elif mode == 2:  # Image
                    text = await extract_from_image(temp_file.name)
                else:
                    raise HTTPException(status_code=400, detail="Invalid mode. Use 1 for PDF, 2 for Image, 3 for URL")

                source = file.filename or "uploaded_file"

            finally:
                os.unlink(temp_file.name)

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No text content could be extracted from the source")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, split_and_store, text, source, session_id)
        await save_chat(session_id, "system", f"Document '{source}' has been uploaded and is now available for questions.", user_id)

        return {
            "message": "Document loaded successfully and is now available for questions",
            "source": source,
            "user_id": user_id,
            "text_length": len(text),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load failed for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/chat")
async def query_data(request: QueryReq, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        logger.info(f"Chat request from user={user_id}, session={request.session_id}")

        rows = await get_mongo_chat_history(request.session_id)
        context = ""
        retrieved_docs = []

        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            collection = db["docs"]

            # Check if documents exist for this session
            doc_count = collection.count_documents({"session_id": request.session_id})
            if doc_count == 0:
                logger.info(f"No documents found for session {request.session_id}")
                context = ""
            else:
                logger.info(f"Found {doc_count} documents for session {request.session_id}")

                vectorstore = MongoDBAtlasVectorSearch(
                    embedding=embeddings,
                    collection=collection,
                    index_name="vector_index",
                )

                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                retrieved_docs = retriever.invoke(request.question)

                if retrieved_docs and len(retrieved_docs) > 0:
                    context = "\n".join([doc.page_content for doc in retrieved_docs])
                    logger.info(f"Found {len(retrieved_docs)} relevant document chunks")
                else:
                    logger.info("No relevant document chunks found")

        except Exception as e:
            logger.warning(f"Could not retrieve document context: {str(e)}")
            context = ""

        # Build the input for the agent
        user_input = request.question
        user_input = (
    "REMINDER: Output ONLY the required format. Do not add extra text or markdown.\n\n"
    + user_input
)
        if context:
            user_input += f"\n\nRelevant document context:\n{context}"
    
        # Add chat history context (last 5 messages)
        if rows:
            history_context = "\n".join([f"{row['role']}: {row['content']}" for row in rows[-5:]])
            user_input += f"\n\nRecent conversation history:\n{history_context}"

        await save_chat(request.session_id, "user", request.question, user_id)

        # NVIDIA Chat model
        llm = ChatNVIDIA(model="nvidia/llama-3.3-nemotron-super-49b-v1")

        # Tools for agent
        tools = [get_current_time, get_weather, calculator_tool, search_google]

        # IMPROVED PROMPT: More concise and direct
        template = """Answer the user's question. Use tools only when needed.

Available tools: get_current_time, get_weather, calculator_tool, search_google

Use this format ONLY when you need a tool:
Thought: I need to use a tool
Action: tool_name
Action Input: input_for_tool
Observation: (result appears here)
Final Answer: your response

For simple questions, just answer directly:
Final Answer: your response

Question: {input}
{agent_scratchpad}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                "tool_names": ", ".join([tool.name for tool in tools])
            }
        )

        # Create agent and executor with better error handling
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        max_iterations=3,  # Reduced from 5
        max_execution_time=60,  # Reduced timeout
        handle_parsing_errors=True,
        return_intermediate_steps=True,  # This helps with debugging
        )

        # Execute the agent with timeout handling
        try:
# After agent_executor.invoke(...)
            result = await agent_executor.ainvoke({"input": user_input})
            response_text = result.get("output", str(result))

            # Count tokens in final answer:
            model = "nvidia/llama-3.3-nemotron-super-49b-v1"
            tokenizer = AutoTokenizer.from_pretrained(model)

            # Tokenize final answer
            final_answer = result["output"]
            tokens_response = tokenizer(final_answer)
            print("The no. of tokens in final answer :", len(tokens_response["input_ids"]))

            # Tokenize intermediate steps (combining all into a single string first)
            steps = result.get("intermediate_steps", [])
            steps_str = " ".join([f"{action}\n{observation}" for action, observation in steps])

            tokens_steps = tokenizer(steps_str)
            print("The no. of tokens in intermediate steps :", len(tokens_steps["input_ids"]))

            # Try to extract 'Final Answer:' from the output
            final_answer_match = re.search(r"Final Answer:\s*(.*)", response_text, re.DOTALL)
            if final_answer_match:
                response_text = final_answer_match.group(1).strip()
            else:
                # Fallback: Try to extract the last Observation
                obs_matches = re.findall(r"Observation:\s*(.*)", response_text)
                if obs_matches:
                    response_text = obs_matches[-1].strip()
                else:
                    # Fallback: Return the whole output (at least something)
                    response_text = response_text.strip()
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            # Fallback: Use LLM directly without tools
            fallback_prompt = f"Please answer this question directly: {request.question}"
            if context:
                fallback_prompt += f"\n\nContext: {context}"
            fallback_result = llm.invoke(fallback_prompt)
            response_text = fallback_result.content
        
        await save_chat(request.session_id, "assistant", response_text, user_id)

        logger.info(f"Chat response generated for session {request.session_id}")
        return {
            "answer": response_text, 
            "has_document_context": bool(context), 
            "context_chunks": len(retrieved_docs)
        }

    except httpx.TimeoutException:
        logger.error(f"NVIDIA API timeout for session {request.session_id}")
        raise HTTPException(status_code=504, detail="AI service timeout - please try again")
    except httpx.HTTPStatusError as e:
        logger.error(f"NVIDIA API error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=502, detail="AI service error - please try again")
    except Exception as e:
        logger.error(f"Query failed for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/history")
async def get_chat_history(session_id: str = Query(...), current_user: dict = Depends(get_current_user)):
    try:
        logger.info(f"Fetching history for session {session_id}")
        rows = await get_mongo_chat_history(session_id)

        history = [{"role": row["role"], "content": row["content"]} for row in rows]
        return {"history": history}
    except Exception as e:
        logger.error(f"History fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def get_sessions(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    try:
        logger.info(f"Fetching sessions for user {user_id}")
        sessions = await get_user_sessions(user_id)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Session fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/create")
async def create_session(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    try:
        session_id = str(uuid.uuid4())
        logger.info(f"Creating new session for user {user_id}: {session_id}")
        await save_chat(session_id, "system", "New session created", user_id)
        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Session creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/documents")
async def check_session_documents(session_id: str, current_user: dict = Depends(get_current_user)):
    try:
        collection = db["docs"]
        doc_count = collection.count_documents({"session_id": session_id})
        return {"has_documents": doc_count > 0, "document_count": doc_count}
    except Exception:
        return {"has_documents": False, "document_count": 0}


@app.post("/login")
async def login(request: UserInfo):
    logger.info(f"Login attempt: {request.email}")
    user = await get_user_by_email(request.email)
    if not user or not pwd_context.verify(request.password, user["password"]):
        logger.warning(f"Login failed for {request.email}")
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    access_token_expires = timedelta(hours=24)
    access_token = create_access_token(data={"sub": request.email}, expires_delta=access_token_expires)

    logger.info(f"Login successful for {request.email}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register_user(request: UserInfo):
    try:
        logger.info(f"Registering user: {request.email}")
        m = await get_user_by_email(request.email)
        if m:
            logger.warning(f"User with email {request.email} already exists")
            raise HTTPException(status_code=400, detail="User already exists")
        hashed_pw = pwd_context.hash(request.password)
        await create_user(request.email, hashed_pw)
        return {"message": "User registered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during registration: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
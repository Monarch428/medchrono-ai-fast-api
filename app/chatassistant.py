# app/chatassistant.py
import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from supabase import create_client, Client

# ---- STEP 0: Load environment variables ----
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")

if not supabase_url or not supabase_key:
    raise ValueError("❌ Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in your .env file.")

# ---- CONFIG ----
PDF_FOLDER = "./app/documents"
VECTOR_DB_PATH = "./app/vector_store"

# ---- Initialize Supabase Client ----
supabase: Client = create_client(supabase_url, supabase_key)

# ---- Initialize FastAPI router ----
router = APIRouter(prefix="/chat", tags=["Chat Assistant"])

# ---- Sync Documents from Supabase Storage ----
def sync_documents_from_supabase(case_id: str = None):
    """
    Download documents from Supabase Storage to local folder.
    If case_id is provided, only sync documents for that case.
    """
    print("🔄 Syncing documents from Supabase Storage...")

    # Create documents folder if it doesn't exist
    os.makedirs(PDF_FOLDER, exist_ok=True)

    try:
        # Get list of all files in Supabase Storage
        bucket_name = "documents"
        result = supabase.storage.from_(bucket_name).list()

        if not result:
            print("⚠️ No folders found in Supabase Storage")
            return 0

        downloaded_count = 0

        # Iterate through case folders
        for folder in result:
            folder_name = folder.get("name")

            # If case_id is specified, only sync that case
            if case_id and folder_name != case_id:
                continue

            # List files in the folder
            files = supabase.storage.from_(bucket_name).list(folder_name)

            for file_obj in files:
                file_name = file_obj.get("name")

                # Only process PDF files
                if not file_name.endswith(".pdf"):
                    continue

                storage_path = f"{folder_name}/{file_name}"
                local_path = os.path.join(PDF_FOLDER, file_name)

                # Download file from Supabase Storage
                try:
                    file_data = supabase.storage.from_(bucket_name).download(storage_path)

                    # Save to local folder
                    with open(local_path, "wb") as f:
                        f.write(file_data)

                    print(f"✅ Downloaded: {file_name}")
                    downloaded_count += 1

                except Exception as e:
                    print(f"❌ Failed to download {storage_path}: {str(e)}")

        print(f"✅ Synced {downloaded_count} documents from Supabase")
        return downloaded_count

    except Exception as e:
        print(f"❌ Error syncing documents: {str(e)}")
        return 0

# ---- Load or Build FAISS Vector Store ----
def load_or_create_vectorstore(force_rebuild: bool = False, case_id: str = None):
    """
    Load existing vector store or create a new one from documents.
    force_rebuild: If True, rebuild the vector store even if it exists
    case_id: If provided, only process documents for this case

    IMPORTANT: Only downloads documents from Supabase when force_rebuild=True
    This prevents unnecessary downloads on every backend startup!
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Check if vector store exists and we're not forcing a rebuild
    if os.path.exists(VECTOR_DB_PATH) and not force_rebuild:
        print("📦 Loading existing vector database from disk (no download needed)...")
        vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    # Only download and rebuild if force_rebuild=True or no vector store exists
    print("🧩 Rebuilding vector database...")

    # Sync documents from Supabase ONLY when rebuilding (case-specific if provided)
    if force_rebuild:
        print("🔄 Syncing documents from Supabase (user requested refresh)...")
        downloaded_count = sync_documents_from_supabase(case_id=case_id)
    else:
        print("ℹ️ No vector store found. Create one by clicking 'Refresh' in the chatbot UI.")
        # Create empty vector store for first-time setup
        from langchain_core.documents import Document
        docs = [Document(page_content="No documents available yet. Click the refresh button to sync documents from your database.", metadata={"source": "system"})]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)
        print("✅ Empty vector store created. Click refresh in chatbot to load documents.")
        return vectorstore

    # Process downloaded documents
    docs = []
    if not os.path.exists(PDF_FOLDER) or not os.listdir(PDF_FOLDER):
        print("⚠️ No documents downloaded. Creating empty vector store.")
        from langchain_core.documents import Document
        docs = [Document(page_content="No documents available yet.", metadata={"source": "system"})]
    else:
        for file in os.listdir(PDF_FOLDER):
            if file.endswith(".pdf"):
                file_path = os.path.join(PDF_FOLDER, file)
                try:
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())
                    print(f"✅ Loaded: {file}")
                except Exception as e:
                    print(f"❌ Failed to load {file}: {str(e)}")

    if not docs:
        print("⚠️ No documents loaded. Creating empty vector store.")
        from langchain_core.documents import Document
        docs = [Document(page_content="No documents available yet.", metadata={"source": "system"})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    print("✅ Vector database created and saved locally.")

    # 🗑️ CLEANUP: Delete PDFs after creating embeddings to save storage
    cleanup_downloaded_pdfs()

    return vectorstore


def cleanup_downloaded_pdfs():
    """
    Delete downloaded PDF files after creating embeddings.
    This saves server storage - we only keep the embeddings (~1-2% of PDF size).
    """
    try:
        if os.path.exists(PDF_FOLDER):
            pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
            deleted_count = 0

            for pdf_file in pdf_files:
                file_path = os.path.join(PDF_FOLDER, pdf_file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Could not delete {pdf_file}: {str(e)}")

            print(f"🗑️ Cleaned up {deleted_count} PDF files to save storage")
            print(f"✅ Embeddings are saved in vector store (much smaller!)")
    except Exception as e:
        print(f"⚠️ Cleanup error: {str(e)}")

# ---- Helper function to format documents ----
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---- Initialize components ----
# Load vector store (no download unless user clicks refresh)
vectorstore = load_or_create_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

print("✅ Chatbot ready! No downloads on startup - only when user clicks refresh.")

# Create prompt template
template = """You are an assistant for question-answering tasks. Use the following pieces of context to answer the question. 
If you don't know the answer, say that you don't have information about that in the provided documents. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Helper function to create chain input
def create_chain_input(question: str, history: str):
    """Create properly formatted input for the RAG chain"""
    return {
        "question": question,
        "chat_history": history,
        "context": question  # This will be used by the retriever
    }

# Create RAG chain using LCEL (LangChain Expression Language)
def get_rag_response(question: str, history: str) -> str:
    """Get response from RAG chain with proper formatting"""
    # Retrieve relevant documents using invoke method
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Format the prompt
    formatted_prompt = prompt.format(
        context=context,
        question=question,
        chat_history=history
    )

    # Get LLM response
    response = llm.invoke(formatted_prompt)
    return response.content

# Store chat history
chat_history = []

# ---- Request & Response Models ----
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    response: str

# ---- API Endpoints ----
@router.post("/", response_model=ChatResponse)
async def chat_with_pdf(request: ChatRequest):
    """Chat with the uploaded documents."""
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Format chat history as string
    history_text = ""
    for human_msg, ai_msg in chat_history[-3:]:  # Keep last 3 exchanges
        history_text += f"Human: {human_msg}\nAssistant: {ai_msg}\n"

    # Invoke the RAG chain with proper formatting
    try:
        answer = get_rag_response(query, history_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
    
    # Store in chat history
    chat_history.append((query, answer))
    
    return ChatResponse(response=answer)

class RefreshRequest(BaseModel):
    case_id: str = None  # Optional: only sync documents for this case


@router.post("/refresh")
async def refresh_documents(request: RefreshRequest = None):
    """
    Sync documents from Supabase and rebuild the vector store.
    Call this endpoint after uploading new documents.
    Optionally provide case_id to only sync documents for a specific case.
    """
    try:
        global vectorstore, retriever, rag_chain

        case_id = request.case_id if request else None

        # Sync documents from Supabase (case-specific if provided)
        count = sync_documents_from_supabase(case_id=case_id)

        # Rebuild vector store (will auto-cleanup PDFs after embedding)
        vectorstore = load_or_create_vectorstore(force_rebuild=True, case_id=case_id)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Note: We use get_rag_response() function now instead of a chain

        return {
            "success": True,
            "message": f"Synced {count} documents and rebuilt vector store (PDFs auto-cleaned to save storage)",
            "documents_synced": count,
            "storage_optimized": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh documents: {str(e)}")

@router.post("/clear-history")
async def clear_chat_history():
    """Clear the chat history."""
    global chat_history
    chat_history = []
    return {"success": True, "message": "Chat history cleared"}
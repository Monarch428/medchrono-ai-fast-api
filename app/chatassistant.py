# app/chatassistant.py
import os
import tempfile
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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
# Use /tmp for temporary document storage (works on most systems including Render)
PDF_FOLDER = os.path.join(tempfile.gettempdir(), "medchrono_documents")

# Base directory for vector stores (one per case for isolation)
if os.path.exists("/opt/render/project"):
    # On Render - use persistent storage
    VECTOR_STORE_BASE = "/opt/render/project/data/vector_stores"
    os.makedirs(VECTOR_STORE_BASE, exist_ok=True)
else:
    # Local development - use app directory
    VECTOR_STORE_BASE = os.path.join(os.path.dirname(__file__), "vector_stores")
    os.makedirs(VECTOR_STORE_BASE, exist_ok=True)

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

# ---- Get Vector Store Path for Case ----
def get_vector_store_path(case_id: str) -> str:
    """Get the vector store path for a specific case."""
    return os.path.join(VECTOR_STORE_BASE, f"case_{case_id}")

# ---- Load or Build FAISS Vector Store for Specific Case ----
def load_or_create_vectorstore_for_case(case_id: str, force_rebuild: bool = False):
    """
    Load existing vector store for a case or create a new one from documents.
    case_id: REQUIRED - The case ID to load documents for
    force_rebuild: If True, rebuild the vector store even if it exists

    This ensures each case has its own isolated vector store.
    """
    if not case_id:
        raise ValueError("case_id is required for case-specific vector store")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store_path = get_vector_store_path(case_id)

    # Check if vector store exists and we're not forcing a rebuild
    if os.path.exists(vector_store_path) and not force_rebuild:
        print(f"📦 Loading vector store for case {case_id} from disk...")
        vectorstore = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    # Build or rebuild vector store for this case
    print(f"🧩 Building vector store for case {case_id}...")

    # Fetch documents directly from Supabase database (not storage)
    try:
        # Query documents table for this case
        response = supabase.table("documents").select("*").eq("case_id", case_id).execute()

        print(f"📋 Found {len(response.data) if response.data else 0} documents for case {case_id}")

        if not response.data or len(response.data) == 0:
            print(f"⚠️ No documents found in database for case {case_id}")
            # Create empty vector store
            from langchain_core.documents import Document
            docs = [Document(
                page_content=f"No documents uploaded yet for this case. Please upload documents to start chatting.",
                metadata={"source": "system", "case_id": case_id}
            )]
        else:
            # Download and process each document
            docs = []
            print(f"🔄 Processing {len(response.data)} documents...")
            for doc_record in response.data:
                print(f"  - Document: {doc_record.get('filename', 'unknown')} (ID: {doc_record.get('id')})")
                try:
                    # Download file from storage
                    if doc_record.get("storage_path"):
                        print(f"    📥 Downloading from storage: {doc_record['storage_path']}")
                        file_data = supabase.storage.from_("documents").download(doc_record["storage_path"])

                        # Save temporarily and process
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        temp_file.write(file_data)
                        temp_file.close()

                        print(f"    📄 Loading PDF content...")
                        # Load PDF
                        loader = PyPDFLoader(temp_file.name)
                        loaded_docs = loader.load()

                        print(f"    ✅ Loaded {len(loaded_docs)} pages")

                        # Add metadata
                        for d in loaded_docs:
                            d.metadata["case_id"] = case_id
                            d.metadata["document_id"] = doc_record["id"]
                            d.metadata["filename"] = doc_record["filename"]

                        docs.extend(loaded_docs)

                        # Clean up temp file
                        os.unlink(temp_file.name)
                        print(f"✅ Processed: {doc_record['filename']} ({len(loaded_docs)} pages)")
                    else:
                        print(f"    ⚠️ No storage_path for document: {doc_record.get('filename')}")

                except Exception as e:
                    print(f"❌ Failed to process {doc_record.get('filename', 'unknown')}: {str(e)}")
                    import traceback
                    traceback.print_exc()

            if not docs:
                print(f"⚠️ No documents could be processed for case {case_id}")
                from langchain_core.documents import Document
                docs = [Document(
                    page_content=f"Documents exist but couldn't be processed. Please check document formats.",
                    metadata={"source": "system", "case_id": case_id}
                )]

        # Create embeddings
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save to case-specific location
        os.makedirs(vector_store_path, exist_ok=True)
        vectorstore.save_local(vector_store_path)
        print(f"✅ Vector store created for case {case_id} with {len(chunks)} chunks")

        return vectorstore

    except Exception as e:
        print(f"❌ Error creating vector store for case {case_id}: {str(e)}")
        # Return empty vector store as fallback
        from langchain_core.documents import Document
        docs = [Document(
            page_content=f"Error loading documents: {str(e)}",
            metadata={"source": "system", "case_id": case_id}
        )]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(vector_store_path, exist_ok=True)
        vectorstore.save_local(vector_store_path)
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

# ---- Initialize LLM (shared across all cases) ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

print("✅ Chatbot ready with case-specific vector stores!")

# Create prompt template
template = """You are an assistant for question-answering tasks about medical and legal documents.
Use the following pieces of context to answer the question.
If you don't know the answer, say that you don't have information about that in the provided documents.
Use three sentences maximum and keep the answer concise.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create RAG response for specific case
def get_rag_response_for_case(case_id: str, question: str, history: str) -> str:
    """Get response from RAG chain for a specific case"""
    # Load vector store for this case
    vectorstore = load_or_create_vectorstore_for_case(case_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Retrieve relevant documents
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

# Store chat history per case
chat_histories = {}

def get_chat_history(case_id: str) -> list:
    """Get chat history for a specific case"""
    if case_id not in chat_histories:
        chat_histories[case_id] = []
    return chat_histories[case_id]

# ---- Request & Response Models ----
class ChatRequest(BaseModel):
    question: str
    case_id: str  # REQUIRED: Case ID for isolation

class ChatResponse(BaseModel):
    response: str

# ---- API Endpoints ----
@router.post("/", response_model=ChatResponse)
async def chat_with_pdf(request: ChatRequest):
    """Chat with documents for a specific case."""
    query = request.question.strip()
    case_id = request.case_id.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not case_id:
        raise HTTPException(status_code=400, detail="case_id is required.")

    # Get case-specific chat history
    chat_history = get_chat_history(case_id)

    # Format chat history as string
    history_text = ""
    for human_msg, ai_msg in chat_history[-3:]:  # Keep last 3 exchanges
        history_text += f"Human: {human_msg}\nAssistant: {ai_msg}\n"

    # Get answer for this specific case
    try:
        answer = get_rag_response_for_case(case_id, query, history_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

    # Store in case-specific chat history
    chat_history.append((query, answer))

    return ChatResponse(response=answer)

class RefreshRequest(BaseModel):
    case_id: str  # REQUIRED: Case ID to refresh documents for

@router.post("/refresh")
async def refresh_documents(request: RefreshRequest):
    """
    Rebuild vector store for a specific case.
    Call this endpoint after uploading new documents to a case.
    This will fetch all documents for the case from Supabase and create embeddings.
    """
    try:
        case_id = request.case_id

        if not case_id:
            raise HTTPException(status_code=400, detail="case_id is required")

        # Rebuild vector store for this case (fetches from Supabase)
        vectorstore = load_or_create_vectorstore_for_case(case_id, force_rebuild=True)

        # Count documents in vector store
        # This is a rough estimate based on chunks
        doc_count = vectorstore.index.ntotal if hasattr(vectorstore.index, 'ntotal') else 0

        return {
            "success": True,
            "message": f"Vector store rebuilt for case {case_id}",
            "case_id": case_id,
            "chunks_created": doc_count,
            "storage_optimized": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh documents: {str(e)}")

class ClearHistoryRequest(BaseModel):
    case_id: str  # REQUIRED: Case ID to clear history for

@router.post("/clear-history")
async def clear_chat_history(request: ClearHistoryRequest):
    """Clear the chat history for a specific case."""
    case_id = request.case_id

    if not case_id:
        raise HTTPException(status_code=400, detail="case_id is required")

    if case_id in chat_histories:
        chat_histories[case_id] = []

    return {"success": True, "message": f"Chat history cleared for case {case_id}", "case_id": case_id}
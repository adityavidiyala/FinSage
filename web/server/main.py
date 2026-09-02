import logging
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict
from typing import List
from uuid import UUID
import schemas
import models
import auth
from database import engine, get_db
import jwt
import requests
import hashlib
import os
import sys
from uuid import uuid4
from fastapi import BackgroundTasks, UploadFile, File

# so we can import finance_rag.pipeline.build_index from web/server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from finance_rag.pipeline import build_index  # noqa: E402
from database import SessionLocal  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FinSage/data/uploads
UPLOAD_ROOT = os.getenv("UPLOAD_ROOT", os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "uploads")))

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="FinSage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

@app.post("/signup", response_model=schemas.UserResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pwd = auth.get_password_hash(user.password)
    new_user = models.User(name=user.name, email=user.email, hashed_password=hashed_pwd)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/login", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


from jwt.exceptions import InvalidTokenError

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except InvalidTokenError:
        raise credentials_exception
        
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

@app.get("/me", response_model=schemas.UserResponse)
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user


@app.post("/conversations", response_model=schemas.ConversationResponse)
def create_conversation(
    conversation: schemas.ConversationCreate, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    new_conv = models.Conversation(
        user_id=current_user.id,
        title=conversation.title or "New Chat"
    )
    db.add(new_conv)
    db.commit()
    db.refresh(new_conv)
    return new_conv

@app.get("/conversations", response_model=List[schemas.ConversationResponse])
def get_conversations(
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    # Strict isolation: Only fetch conversations belonging to the current user
    conversations = db.query(models.Conversation)\
        .filter(models.Conversation.user_id == current_user.id)\
        .order_by(models.Conversation.created_at.desc())\
        .all()
    return conversations

@app.get("/conversations/{conversation_id}", response_model=schemas.ConversationResponse)
def get_conversation(
    conversation_id: UUID, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    # Verify both the ID and the ownership
    conversation = db.query(models.Conversation)\
        .filter(
            models.Conversation.id == conversation_id,
            models.Conversation.user_id == current_user.id
        ).first()
        
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    return conversation


# middleman that connects the webserver backend with the model api
@app.post("/conversations/{conversation_id}/messages", response_model=schemas.MessageResponse)
def create_message(
    conversation_id: UUID,
    message: schemas.MessageCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # 1. Verify conversation ownership
    conversation = db.query(models.Conversation).filter(
        models.Conversation.id == conversation_id,
        models.Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 2. Check document status ONLY if documents were attached
    if conversation.documents:
        pending_docs = [doc.filename for doc in conversation.documents if doc.status == "parsing"]
        if pending_docs:
            raise HTTPException(
                status_code=400,
                detail=f"Documents still parsing: {', '.join(pending_docs)}. Please wait for them to finish.",
            )

        failed_docs = [doc.filename for doc in conversation.documents if doc.status == "failed"]
        if failed_docs:
            raise HTTPException(
                status_code=400,
                detail=f"Some attached documents failed to parse: {', '.join(failed_docs)}.",
            )

    ready_document_ids = [str(doc.id) for doc in conversation.documents if doc.status == "ready"]

    # 3. Save the user's incoming message
    user_msg = models.Message(
        conversation_id=conversation_id,
        role=message.role,
        content=message.content
    )
    db.add(user_msg)
    db.commit()

    # 4. Reconstruct History (Limit to last 3 turns / 6 messages to cap token costs)
    recent_messages = db.query(models.Message)\
        .filter(models.Message.conversation_id == conversation_id)\
        .order_by(models.Message.created_at.desc())\
        .limit(6)\
        .all()

    recent_messages.reverse()  # Chronological order

    history_turns = []
    temp_question = None
    for msg in recent_messages:
        if msg.id == user_msg.id:
            continue

        if msg.role == 'user':
            temp_question = msg.content
        elif msg.role == 'assistant' and temp_question:
            history_turns.append({"question": temp_question, "answer": msg.content})
            temp_question = None

    # 5. Call the stateless Model API (passes ready_document_ids, empty [] if no doc attached)
    try:
        response = requests.post(
            "http://127.0.0.1:8001/query",
            json={
                "question": message.content,
                "document_ids": ready_document_ids,
                "conversation_id": str(conversation_id),
                "history": history_turns if history_turns else None,
                "use_cache": True,
                "use_decomposition": True
            },
            timeout=120.0
        )
        response.raise_for_status()
        rag_data = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Model API error: {str(e)}")

    # 6. Save and return the Assistant's generated response
    assistant_msg = models.Message(
        conversation_id=conversation_id,
        role="assistant",
        content=rag_data["answer"],
        citations=rag_data.get("citations", []),
        usage=rag_data.get("usage", {})
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return assistant_msg


# loads the message history when a user clicks on an old conversation
@app.get("/conversations/{conversation_id}/messages", response_model=List[schemas.MessageResponse])
def get_messages(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Verify ownership before returning messages
    conversation = db.query(models.Conversation).filter(
        models.Conversation.id == conversation_id,
        models.Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    messages = db.query(models.Message)\
        .filter(models.Message.conversation_id == conversation_id)\
        .order_by(models.Message.created_at.asc())\
        .all()
        
    return messages


def _run_ingestion(document_id: str, pdf_path: str):
    print(f"--> [DEBUG] Starting background ingestion for {document_id} at {pdf_path}")
    db = SessionLocal()
    try:
        build_index(document_id=document_id, pdf_path=pdf_path)
        doc = db.query(models.Document).filter(models.Document.id == document_id).first()
        doc.status = "ready"
        db.commit()
        print(f"--> [DEBUG] Successfully indexed {document_id}")
    except Exception as e:
        print(f"--> [DEBUG] Error in build_index: {e}")
        logging.exception("build_index() failed for document_id=%s", document_id)
        doc = db.query(models.Document).filter(models.Document.id == document_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
    finally:
        db.close()


@app.post("/documents", response_model=schemas.DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    file_bytes = await file.read()
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    # Dedup check: has this exact user already uploaded this exact file?
    existing = db.query(models.Document).filter(
        models.Document.user_id == current_user.id,
        models.Document.content_hash == content_hash,
        models.Document.status != "failed",
    ).first()
    if existing:
        return existing  # reuse — no re-parsing, no new row

    document_id = uuid4()
    pdf_path = os.path.join(UPLOAD_ROOT, f"{document_id}.pdf")
    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)

    new_doc = models.Document(
        id=document_id,
        user_id=current_user.id,
        filename=file.filename,
        content_hash=content_hash,
        status="parsing",
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    background_tasks.add_task(_run_ingestion, str(document_id), pdf_path)

    return new_doc


@app.post("/conversations/{conversation_id}/documents", response_model=schemas.ConversationResponse)
def attach_document(
    conversation_id: UUID,
    body: schemas.DocumentAttachRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    conversation = db.query(models.Conversation).filter(
        models.Conversation.id == conversation_id,
        models.Conversation.user_id == current_user.id,
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    document = db.query(models.Document).filter(
        models.Document.id == body.document_id,
        models.Document.user_id == current_user.id,
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document not in conversation.documents:
        conversation.documents.append(document)
        db.commit()
        db.refresh(conversation)

    return conversation

@app.get("/documents/{document_id}", response_model=schemas.DocumentResponse)
def get_document(
    document_id: UUID,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    document = db.query(models.Document).filter(
        models.Document.id == document_id,
        models.Document.user_id == current_user.id,
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@app.get("/documents", response_model=List[schemas.DocumentResponse])
def list_documents(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return db.query(models.Document)\
        .filter(models.Document.user_id == current_user.id)\
        .order_by(models.Document.created_at.desc())\
        .all()

@app.patch("/conversations/{conversation_id}", response_model=schemas.ConversationResponse)
def update_conversation(
    conversation_id: UUID,
    conversation_update: schemas.ConversationUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    conv = (
        db.query(models.Conversation)
        .filter(
            models.Conversation.id == conversation_id,
            models.Conversation.user_id == current_user.id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if conversation_update.title is not None:
        conv.title = conversation_update.title
    if conversation_update.pinned is not None:
        conv.pinned = conversation_update.pinned

    db.commit()
    db.refresh(conv)
    return conv

@app.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    conv = (
        db.query(models.Conversation)
        .filter(
            models.Conversation.id == conversation_id,
            models.Conversation.user_id == current_user.id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conv)
    db.commit()
    return None
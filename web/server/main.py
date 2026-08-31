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

#middleman that connects the webserver backend with the model api

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

    # 2. Save the user's incoming message
    user_msg = models.Message(
        conversation_id=conversation_id,
        role=message.role,
        content=message.content
    )
    db.add(user_msg)
    db.commit()

    # 3. Reconstruct History (Limit to last 3 turns / 6 messages to cap token costs)
    recent_messages = db.query(models.Message)\
        .filter(models.Message.conversation_id == conversation_id)\
        .order_by(models.Message.created_at.desc())\
        .limit(6)\
        .all()
    
    recent_messages.reverse() # Chronological order
    
    history_turns = []
    temp_question = None
    for msg in recent_messages:
        # Skip the message we just saved so it isn't passed as history
        if msg.id == user_msg.id:
            continue
            
        if msg.role == 'user':
            temp_question = msg.content
        elif msg.role == 'assistant' and temp_question:
            history_turns.append({"question": temp_question, "answer": msg.content})
            temp_question = None

    # 4. Call the stateless Model API
    # Note: Ensure your api/main.py is running on a different port (e.g., 8001)
    try:
        response = requests.post(
            "http://127.0.0.1:8001/query",
            json={
                "question": message.content,
                "history": history_turns if history_turns else None,
                "use_cache": True,
                "use_decomposition": False
            },
            timeout=30.0
        )
        response.raise_for_status()
        rag_data = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Model API error: {str(e)}")

    # 5. Save and return the Assistant's generated response
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
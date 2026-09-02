from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any
from datetime import datetime
from uuid import UUID

# --- AUTHENTICATION SCHEMAS ---
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str

# --- DOCUMENT SCHEMAS ---
class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    content_hash: str
    status: str  # "parsing" | "ready" | "failed"
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# --- MESSAGE SCHEMAS ---
class MessageBase(BaseModel):
    role: str
    content: str

class MessageCreate(MessageBase):
    pass

class MessageResponse(MessageBase):
    id: UUID
    conversation_id: UUID
    citations: Optional[Any] = None
    usage: Optional[Any] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- CONVERSATION SCHEMAS ---
class ConversationBase(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = False

class ConversationCreate(ConversationBase):
    pass

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None
    
class ConversationResponse(ConversationBase):
    id: UUID
    created_at: datetime
    # Optionally include the list of messages when fetching a conversation
    messages: List[MessageResponse] = []
    documents: List[DocumentResponse] = []

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    content_hash: str
    status: str  # "parsing" | "ready" | "failed"
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class DocumentAttachRequest(BaseModel):
    document_id: UUID
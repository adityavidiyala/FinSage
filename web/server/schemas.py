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

class ConversationCreate(ConversationBase):
    pass

class ConversationResponse(ConversationBase):
    id: UUID
    created_at: datetime
    # Optionally include the list of messages when fetching a conversation
    messages: List[MessageResponse] = []

    model_config = ConfigDict(from_attributes=True)
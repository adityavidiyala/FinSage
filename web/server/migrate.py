from sqlalchemy import text
from database import engine

with engine.connect() as conn:
    conn.execute(
        text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS pinned BOOLEAN DEFAULT FALSE NOT NULL;")
    )
    conn.commit()
    print("Successfully added 'pinned' column to conversations table!")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# In-memory database
notes_db = []

class Note(BaseModel):
    title: str
    content: str

@app.post("/notes")
def add_note(note: Note):
    notes_db.append(note)
    return {"message": "Note added", "note_id": len(notes_db) - 1}

@app.get("/notes", response_model=List[Note])
def get_notes():
    return notes_db

@app.delete("/notes/{note_id}")
def delete_note(note_id: int):
    if 0 <= note_id < len(notes_db):
        deleted = notes_db.pop(note_id)
        return {"message": "Note deleted", "title": deleted.title}
    raise HTTPException(status_code=404, detail="Note not found")

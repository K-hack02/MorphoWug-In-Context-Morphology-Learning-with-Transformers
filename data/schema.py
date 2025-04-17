from pydantic import BaseModel

class WordPair(BaseModel):
    A: str
    B: str

class WordPairs(BaseModel):
    category: str
    pairs: list[WordPair]    

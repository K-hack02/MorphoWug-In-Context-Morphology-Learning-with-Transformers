from pydantic import BaseModel

class WordPair(BaseModel):
    word1: str
    word2: str

class WordPairs(BaseModel):
    category: str
    pairs: list[WordPair]    

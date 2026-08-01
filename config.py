

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

CHROMA_PERSIST_DIR = "chroma_db"
SAMPLE_DOCS_DIR = "sample_docs"
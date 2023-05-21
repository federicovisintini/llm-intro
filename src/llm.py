import os
from src.secret import OPENAI_API_KEY
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

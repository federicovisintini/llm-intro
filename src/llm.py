import os
from src.secret import OPENAI_API_KEY
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = OpenAI(temperature=0.7, model_name="text-davinci-003")

if __name__ == '__main__':
    print(llm("What would be a good company name for a company that makes colorful socks?"))

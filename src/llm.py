import os
from src.secret import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = OpenAI(temperature=0.7, model_name="text-davinci-003")

prompt = PromptTemplate(
    input_variables=["user_request"],
    template='{user_request} + try to come up with the output thinking out loud. '
             'First decompose the problem into substeps, and try to find the solution for the single substeps. '
             'Then, put them together, and print the user response enclosing it into “”. Use "" only for user output.'
)

chain = LLMChain(llm=llm, prompt=prompt)

if __name__ == '__main__':
    input = "What would you do if you were as powerful as a singularity?"
    print(chain.run(input))

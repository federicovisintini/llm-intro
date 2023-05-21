import os

from langchain.agents import initialize_agent, AgentType, load_tools

from src.secret import OPENAI_API_KEY, WOLFRAM_ALPHA_APPID
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["WOLFRAM_ALPHA_APPID"] = WOLFRAM_ALPHA_APPID

llm = OpenAI(temperature=0.7, model_name="text-davinci-003")

prompt = PromptTemplate(
    input_variables=["user_request"],
    template='{user_request} + try to come up with the output thinking out loud. '
             'First decompose the problem into substeps, and try to find the solution for the single substeps. '
             'Then, put them together, and print the user response enclosing it inside /// ///. '
             'Use /// /// only for user output.'
)

tools = load_tools(["wolfram-alpha"])

# chain = LLMChain(llm=llm, prompt=prompt)
# conversation = ConversationChain(llm=llm, verbose=True)
agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True)

if __name__ == '__main__':
    input = "What would you do if you were as powerful as a singularity?"
    # print(chain.run(input))

    # output = conversation.predict(input="Hi there!")
    # output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
    # print(output)

    agent.run(input)

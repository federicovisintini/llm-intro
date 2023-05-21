import os

from langchain.agents import initialize_agent, AgentType, load_tools

from src.secret import OPENAI_API_KEY, SERPAPI_API_KEY
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

llm = OpenAI(temperature=0, model_name="text-davinci-003")

prompt = PromptTemplate(
    input_variables=["user_request"],
    template='{user_request} + try to come up with the output thinking out loud. '
             'First decompose the problem into substeps, and try to find the solution for the single substeps. '
             'Then, put them together, and print the user response enclosing it inside /// ///. '
             'Use /// /// only for user output.'
)

tools = load_tools(["serpapi", "wikipedia", "llm-math"], llm=llm)

# chain = LLMChain(llm=llm, prompt=prompt)
# conversation = ConversationChain(llm=llm, verbose=True)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

if __name__ == '__main__':
    input = "What was the low temperature in SF yesterday in Celsius?" \
            "Was percentile of temperature in the year was it?"
    # print(chain.run(input))

    # output = conversation.predict(input="Hi there!")
    # output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
    # print(output)

    print(agent.run(input))

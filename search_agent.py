from dotenv import load_dotenv

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv(".env")

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini")
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm,
                           tools=tools,
                           prompt=react_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chain = agent_executor

def main():
    print("Search Agent using Langchain!!!")

    result = chain.invoke(input={
        "input": "search for 3 job postings for an AI Engineer using langchain in the bay area on LinkedIn and list their details"
    })
    print(result)


if __name__ == "__main__":
    main()
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI

# from callbacks import AgentCallbackHandler


load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')

    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str):
    for tool in tools:
        if tool_name == tool.name:
            return tool

    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    tools = [get_text_length]

    llm = ChatOpenAI(temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # Start conversation
    messages = [HumanMessage(content="What is the length of the word: DOG")]

    while True:
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"Observation={observation}")

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )

            continue

        print(ai_message.content)
        break

from composio_openai import ComposioToolSet
from openai import OpenAI

from toolUsage import mine_toolrun
from dataAndTimezone import getDateAndTimezone


DATE, TIMEZONE = getDateAndTimezone()


def composio_toolrun(response, composio_toolset):
    """
    Runs tools from response and returns the results.
    """
    valid_calls = [
        call for call in response.choices[0].message.tool_calls
        if not call.function.name.endswith('_')
    ]
    
    response.choices[0].message.tool_calls = valid_calls
    return [composio_toolset.handle_tool_calls(response)]

def execute_tools(todo: str, tools: dict, openai_api_key: str, composio_toolset: ComposioToolSet, api_keys: dict) -> tuple[int, str]:

    """
    Executes tools based on the given task and returns the results.
    """

    openai_client = OpenAI(api_key=openai_api_key)

    try:
        tool_results = []
        
        if tools['composio']:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                tools=tools['composio'],
                messages=[
                    {"role": "system", "content": f"Execute tools to perform tasks. Today's date is {DATE} and timezone is {TIMEZONE}"},
                    {"role": "user", "content": todo},
                ],
            )
            tool_results = composio_toolrun(response, composio_toolset)

            for tool in tool_results[0]:
                if tool['file']:
                    with open(tool['file'], 'rb') as file:
                        file_content = file.read()
                        tool['result'] = file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content
                        tool['file'] = None

        if tools['mine']:
            for tool in tools['mine']:
                api_key = api_keys.get(tool)
                tool_results.append(mine_toolrun(tool, todo, openai_api_key, api_key))

        return 200, tool_results
    
    except Exception as e:
        return 400, f'error: {e}'

def run(todo: str, tools: dict, composio_toolset: ComposioToolSet, api_keys: dict) -> tuple[int, str]:

    open_api_key = os.getenv('OPENAI_API_KEY')
    openai_client = OpenAI(api_key=open_api_key)

    code, tool_results = execute_tools(todo, tools, open_api_key, composio_toolset, api_keys)

    if code == 400:
        return code, tool_results

    print(tool_results)

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are an AI assistant. Today's date is {DATE} and the timezone is {TIMEZONE}. Answer to query and if you need you can use the tools."},
            {"role": "user", "content": f"Tool result: {tool_results}\n\nUser request: {todo}"},
        ],
        tools=tools['composio'],
    )

    if response.choices[0].message.tool_calls:
        tool_result = composio_toolrun(response, composio_toolset)
        print(tool_result)
        return 200, "Success"
    
    answer = response.choices[0].message.content
    print(answer)

    return 200, answer


if __name__ == "__main__":
    import os
    import dotenv
    from composio_openai import ComposioToolSet, Action

    dotenv.load_dotenv()
    open_ai_key = os.getenv("OPENAI_API_KEY")

    toolset = ComposioToolSet()
    tools = toolset.get_tools(actions=[Action.WEATHERMAP_WEATHER])

    print(run("what is the weather in sf", tools, open_ai_key, toolset))






# Version of a program nobody needs for now





# from langchain.agents import AgentExecutor, create_openai_functions_agent
# from typing import Annotated, Literal, TypedDict

# from langchain_core.messages import HumanMessage
# from langchain_openai import ChatOpenAI
# from langchain_core.tools import tool
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import END, START, StateGraph, MessagesState
# from langgraph.prebuilt import ToolNode

# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """You are an AI agent responsible for taking actions using various tools provided to you. 
#         You must utilize the correct APIs from the given tool-set based on the task at hand.""",
#         ),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )

# def should_continue(state: MessagesState) -> Literal["tools", END]:
#     messages = state['messages']
#     last_message = messages[-1]
#     if last_message.tool_calls:
#         return "tools"
#     return END


# def call_model(state: MessagesState):
#     messages = state['messages']
#     response = model.invoke(messages)
#     return {"messages": [response]}


# def run(todo:str, tools:list, openai_api_key:str) -> str:

#     model.bind_tools(tools)

#     tool_node = ToolNode(tools)


#     workflow = StateGraph(MessagesState)

#     workflow.add_node("agent", call_model)
#     workflow.add_node("tools", tool_node)

#     workflow.add_edge(START, "agent")

#     workflow.add_conditional_edges(
#         "agent",
#         should_continue,
#     )

#     workflow.add_edge("tools", 'agent')

#     checkpointer = MemorySaver()

#     app = workflow.compile(checkpointer=checkpointer)

#     final_state = app.invoke(
#         {"messages": [HumanMessage(content="what is the weather in sf")]},
#         config={"configurable": {"thread_id": 42}}
#     )

#     return final_state["messages"][-1].content

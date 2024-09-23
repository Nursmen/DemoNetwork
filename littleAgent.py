from datetime import datetime

from composio_openai import ComposioToolSet
from openai import OpenAI

from toolUsage import useTool

from streamlit_javascript import st_javascript

DATE = datetime.today().strftime("%Y-%m-%d")
TIMEZONE = st_javascript("""await (async () => {
    const now = new Date();

    const timezoneOffsetInMinutes = now.getTimezoneOffset();
    const offsetHours = Math.floor(Math.abs(timezoneOffsetInMinutes) / 60);
    const offsetMinutes = Math.abs(timezoneOffsetInMinutes) % 60;

    const sign = timezoneOffsetInMinutes > 0 ? "-" : "+";

    const formattedOffset = `UTC${sign}${String(offsetHours).padStart(2, '0')}:${String(offsetMinutes).padStart(2, '0')}`;

    return formattedOffset;
})().then(returnValue => returnValue)""")


def run(todo: str, tools: dict, openai_api_key: str, composio_toolset: ComposioToolSet, api_keys: dict) -> tuple[int, str]:
    
    openai_client = OpenAI(api_key=openai_api_key)

    try:
        tool_result = []
        
        if len(tools['composio']) > 0:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                tools=tools['composio'],
                messages=[
                    {"role": "system", "content": f"Excecute tools to do some work todays date is {DATE} and timezone {TIMEZONE}"},
                    {"role": "user", "content": todo},
                ],
            )

            tool_result = composio_toolset.handle_tool_calls(response)


        for tool in tool_result:
            if tool['file'] is not None:
                with open(tool['file'], 'rb') as file:
                    file_content = file.read()
                    if isinstance(file_content, bytes):
                        file_content = file_content.decode('utf-8')
                    tool['result'] = file_content
                    tool['file'] = None

        for tool in tools['mine']:
            tool_result.ap pend(useTool(tool, todo, openai_api_key, api_keys[tool]))

        print(tool_result)

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are an AI assistant. Use the following tool result to answer the user's request. Today's date is {DATE} and the timezone is {TIMEZONE}."},
                {"role": "user", "content": f"Tool result: {tool_result}\n\nUser request: {todo}"},
            ],
        )

        # Extract the model's answer
        answer = response.choices[0].message.content
        print(answer)

        # Return the answer
        return 200, answer
    except Exception as e:
        return 400, f'error: {e}'


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

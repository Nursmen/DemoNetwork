from datetime import datetime

from composio_openai import ComposioToolSet
from openai import OpenAI

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



def run(todo:str, tools:list, openai_api_key:str, composio_toolset:ComposioToolSet) -> str:
    
    openai_client = OpenAI(api_key=openai_api_key)

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        tools=tools,
        messages=[
            {"role": "system", "content": f"Excecute tools to do some work todays date is {DATE} and timezone {TIMEZONE}"},
            {"role": "user", "content": todo},
        ],
    )


    try:
        result = composio_toolset.handle_tool_calls(response)
    except Exception as e:
        return 400

    return 200
    

if __name__ == "__main__":
    import os
    import dotenv

    dotenv.load_dotenv()
    open_ai_key = os.getenv("OPENAI_API_KEY")

    print(run("what is the weather in sf", [], open_ai_key, ComposioToolSet()))


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

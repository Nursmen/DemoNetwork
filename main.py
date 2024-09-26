
# Here we import all necessary stuff

from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from composio_openai import ComposioToolSet, Action
from dataAndTimezone import getDateAndTimezone
from searcherTool import tool_searcher
from integrations import add_integration, check_integration
from login import login, logout, authentificate
from littleAgent import run
import pandas as pd
import os

import streamlit as st



def just_run(openai_api_key, prompt, memory):

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True, seed=42, temperature=0.5)
    tools = []
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        response = executor.invoke(prompt, cfg)
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]

    return response["output"]

def check_my_tool(tool, st):
    my_tools = pd.read_csv('./tools_mine.csv')
    the_tool = my_tools[my_tools['Name'] == tool]
    if the_tool['Need API KEY'].values[0] != 'No' and tool not in st.session_state.api_keys:
        st.write(f"Please provide API key for {the_tool['Name'].values[0]}:")
        st.write(f"Get key: {the_tool['Get key'].values[0]}")
        st.write(f"Cost: {the_tool['Cost'].values[0]}")
        return tool
    return None

def check_composio_tools(composio_toolset):
    apps = {'name': [], 'link': []}

    for tool in st.session_state.tools_needed:
        app = tool.split('_')[0].lower()
        if not check_integration(app) and not tool.endswith('_'):
            try:
                apps['name'].append(app)
                apps['link'].append(add_integration(app))
            except Exception as e:
                st.error(f"Authentication not required for {app}: {str(e)}")
    
    return apps


# Start streamlit

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

st.set_page_config(page_title="Nurses demo", page_icon="👍")
st.title("Nurses demo")




# Here we show date and timezone

DATE, TIMEZONE = getDateAndTimezone()

st.write(f"Date: {DATE} and timezone: {TIMEZONE}")





# Sidebar settings

logout()

if "key" not in st.session_state:
    st.session_state.url, st.session_state.key = login()
st.sidebar.markdown(
    f"""
    Please go to
    [this link]({st.session_state.url})
    and pass us the secret code  
    """
)
code = st.sidebar.text_input("secret code", type='password')


openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

os.environ["OPENAI_API_KEY"] = openai_api_key






# Massages in chat

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, 
    return_messages=True, 
    memory_key="chat_history", 
    output_key="output"
)

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)





# First run ro give models history some context

if 'first' not in st.session_state and openai_api_key:
    with st.chat_message("user"):
        st.write("You are a usefull planner that can devide a problem into smaller parts. Understood?")


# Some initializations

if 'choosing_tools' not in st.session_state:
    st.session_state.choosing_tools = True

if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {}

if 'current_tool' not in st.session_state:
    st.session_state.current_tool = None

tools_needed = []

# Handle input

if prompt := st.chat_input(placeholder="Ask bot to do something..."):
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    if not authentificate(code, st.session_state.key):
        st.info("Please add your Composio code to continue.")
        st.stop()


    # When we try to get right response

    if st.session_state.choosing_tools == True:

        _ = just_run(openai_api_key, prompt, memory)

        tools_needed = prompt.split('\n')
        tools_needed = set(tool_searcher.invoke(tool) for tool in tools_needed)

        st.session_state.tools_needed = tools_needed
        st.session_state.prompt = prompt
    
        st.session_state.choosing_tools = False

        st.write("Did I find the tools right? \n\n ", tools_needed)

    else:

        # Yes I find this tools right
        # My api is ...
        # I visited all the URLs for composio

        if "yes" in prompt.lower() or any(char.isdigit() for char in prompt) or 'ready' in prompt.lower():

            if 'yes' not in prompt.lower() and 'ready' not in prompt.lower():
                st.session_state.api_keys[st.session_state.current_tool] = prompt
                st.session_state.current_tool = None
            
            composio_toolset = ComposioToolSet()
            apps = check_composio_tools(composio_toolset) 
            
            if not apps['name']:
                tools = {'composio': [], 'mine': []}

                for tool in st.session_state.tools_needed:
                    if tool.endswith('_'):
                        tools['mine'].append(tool)
                        st.session_state.current_tool = check_my_tool(tool, st)
                    else:
                        action = getattr(Action, tool)
                        tools['composio'].extend(composio_toolset.get_tools(actions=[action]))

                if st.session_state.current_tool is None:
                    prompt = f"You should use these tools: {', '.join(st.session_state.tools_needed)} to {st.session_state.prompt}"
                    st.chat_message("user").write(prompt)

                    with st.chat_message("assistant"):
                        response, answer = run(
                            todo=prompt,
                            tools=tools,
                            composio_toolset=composio_toolset,
                            api_keys=st.session_state.api_keys
                        )
                        
                        if response == 200:
                            st.write(answer)
                        else:
                            st.error("Something went wrong. Please try again.")

                    st.session_state.choosing_tools = True
            else:
                st.write("Please go to the following links:")
                for app, link in zip(apps['name'], apps['link']):
                    st.markdown(f"[{app}]({link})")
                st.write("When you are ready, please type 'ready'")

        else:
            st.session_state.choosing_tools = True
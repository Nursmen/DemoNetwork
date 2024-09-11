from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

import dotenv

import os
import streamlit as st
import re

from searcherTool import tool_searcher

dotenv.load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

import pandas as pd

App_names = ['APIFY', 'ASANA', 'ATTIO', 'BROWSERBASE_TOOL', 'BROWSER_TOOL', 'CLICKUP', 'CODEINTERPRETER', 'CODE_ANALYSIS_TOOL', 'CODE_FORMAT_TOOL', 'CODE_GREP_TOOL', 'CODE_INDEX_TOOL', 'CODE_MAP_TOOL', 'COMPOSIO', 'DISCORD', 'DISCORDBOT', 'DROPBOX', 'ELEVENLABS', 'EMBED_TOOL', 'EXA', 'FIGMA', 'FILETOOL', 'FIRECRAWL', 'GIT', 'GITHUB', 'GMAIL', 'GOOGLECALENDAR', 'GOOGLEDOCS', 'GOOGLEDRIVE', 'GOOGLEMEET', 'GOOGLESHEETS', 'GOOGLETASKS', 'GREPTILE', 'HACKERNEWS', 'HISTORY_FETCHER', 'HUBSPOT', 'IMAGE_ANALYSER', 'INDUCED_AI', 'JIRA', 'KLAVIYO', 'LINEAR', 'LISTENNOTES', 'MAILCHIMP', 'MATHEMATICAL', 'MULTIONAI', 'NASA', 'NOTION', 'PERPLEXITYAI', 'PIPEDRIVE', 'RAGTOOL', 'SCHEDULER', 'SERPAPI', 'SHELLTOOL', 'SLACK', 'SLACKBOT', 'SNOWFLAKE', 'SPIDERTOOL', 'SQLTOOL', 'TASKADE', 'TAVILY', 'TRELLO', 'TWITTER', 'TYPEFORM', 'WEATHERMAP', 'WEBTOOL', 'WORKSPACE_TOOL', 'YOUSEARCH', 'YOUTUBE', 'ZENDESK', 'ZEPTOOL']

prompt_template = """
You are an assistant designed to simplify tasks and choose the correct tools. Follow these steps:

1. Use your tools search feature to find the right tool for the task.
2. Rewrite the command to make the task easier for another model.
3. Output the tool name and the rewritten command.
Example: Command: "Schedule a meeting with my team for tomorrow at 10 AM." Apps: Google, Twitter, YouTube
Search tool result: GOOGLECALENDAR_CREATE_EVENT found for scheduling events.
Output:

Rewritten command: "Create a calendar event titled 'Team Meeting' at 10 AM tomorrow."
Tool: GOOGLECALENDAR_CREATE_EVENT
Your turn: Command: {command}
Apps: {apps}
"""

prompt_raw = PromptTemplate(
    input_variables=["command", "apps"],
    template=prompt_template,
)


tools = pd.read_csv('./tools/tools.csv')['tool'].to_numpy()








if "OPENAI_API_KEY" not in os.environ:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
else:
    openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Nurse: Chat with search", page_icon="😎")
st.title("😎 Nurse: Chat with search")

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("We are waiting...")
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

if prompt := st.chat_input(placeholder="Start argue."):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    tools = [tool_searcher]  
   
    chat_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        verbose=True,
        memory=memory,
        seed=0
    )


    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    prompt = prompt_raw.format(apps = App_names, command = prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        response = executor.invoke(prompt, cfg)

        st.write(response['output'])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]


    
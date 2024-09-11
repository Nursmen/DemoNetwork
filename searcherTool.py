from langchain.agents import tool
from langchain_openai import ChatOpenAI

import weaviate
from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever,
)

import cohere

import os
import dotenv

import re

dotenv.load_dotenv()

URL = os.getenv('WEAVIATE_URL')
APIKEY = os.getenv('WEAVIATE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

client = weaviate.Client(
    url=URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=APIKEY),
    additional_headers={
        "X-Openai-Api-Key": OPENAI_API_KEY,
    },
)

retriever = WeaviateHybridSearchRetriever(
    client=client,
    index_name="TOOLSET_TEST",
    text_key="text",
    attributes=[],
    create_schema_if_missing=True,
    k=25,
)

co = cohere.Client(api_key=COHERE_API_KEY)

model = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

prompt = """Which one of the following tools is more likly to be used for a given task? Return it in the same format as the tool-set\n\n"""

@tool
def tool_searcher(query: str):
    """
    Scan a predefined tools database and retrieve the most appropriate tool required for a given task. When a command is provided, the searcher automatically looks up the tools available in the connected apps to ensure the necessary tool exists.
    """
    firstFilter = [ff.page_content for ff in retriever.invoke(query)]

    results = co.rerank(model="rerank-english-v3.0", query=query, documents=firstFilter, top_n=20, return_documents=True)

    s = ", ".join([o.document.text for o in results.results])

    results = model.invoke(prompt + s + "\n\n" + query)

    tools_needed = re.findall(r'\b[A-Z_]+\b', results.content)[-1]

    results = [ff.page_content for ff in retriever.invoke(tools_needed)]

    return results[0]

if __name__ == "__main__":
    for i in range(1):
        print(tool_searcher("Checks the availability of specified users in Google Calendar for a given time range"))
        
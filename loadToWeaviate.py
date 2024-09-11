import pandas as pd
from langchain_core.documents import Document

import weaviate
from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever,
)
import os
import dotenv

dotenv.load_dotenv()

all_tools = pd.read_csv('./tools/tools.csv')['tool'].to_numpy()
all_tools = [Document(page_content=tool) for tool in all_tools]

URL = os.getenv('WEAVIATE_URL')
APIKEY = os.getenv('WEAVIATE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

cohere_api_key = os.getenv('COHERE_API')


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
)


retriever.add_documents(all_tools)

print(retriever.invoke('create event at google calendar'))
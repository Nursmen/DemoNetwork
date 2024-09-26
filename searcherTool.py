from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import weaviate
from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever,
)
import cohere
import os
import re


URL = 'https://xq2djbbhscm62fg72eqsg.c0.us-east1.gcp.weaviate.cloud'
APIKEY = 'UqAdx7BDphab5z4o1fh9OqeIpVsSunqtxUSX'
COHERE_API_KEY = '5EpCLYfOleHpMpjOuHjlQtylxlUstRWJnE2o8AJH'

@tool
def tool_searcher(query: str):
    """
    Scan a predefined tools database and retrieve the most appropriate tool required for a given task. When a command is provided, the searcher automatically looks up the tools available in the connected apps to ensure the necessary tool exists.
    """

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    client = weaviate.Client(
        url=URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=APIKEY),
        additional_headers={"X-Openai-Api-Key": OPENAI_API_KEY},
    )

    retriever = WeaviateHybridSearchRetriever(
        client=client,
        index_name="TOOLSET5",
        text_key="text",
        attributes=[],
        create_schema_if_missing=True,
        k=25,
    )

    co = cohere.Client(api_key=COHERE_API_KEY)
    model = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

    prompt = "Which of the following tools is most likely to be used for the given task? Return it in the same format as the tool-set\n\n"
    
    # First filter: Retrieve relevant tools
    first_filter = retriever.invoke(query)

    # Rerank results
    reranked_results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=[ff.page_content for ff in first_filter],
        top_n=20,
        return_documents=True
    )

    # Prepare input for the model
    tool_descriptions = ", ".join(o.document.text for o in reranked_results.results)
    
    # Get model's recommendation
    model_response = model.invoke(prompt + tool_descriptions + "\n\n" + query)

    # Extract tool name
    try:
        tools_needed = re.findall(r'\b[A-Z_]+\b', model_response.content)[-1]
    except IndexError:
        return None
    
    # Final retrieval
    final_results = retriever.invoke(tools_needed)
    
    return final_results[0].page_content if final_results else None

if __name__ == "__main__":
    for i in range(1):
        print(tool_searcher("Checks the availability of specified users in Google Calendar for a given time range"))
        
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(
    override=True,
)

llm = ChatOpenAI(
    model=os.getenv("LLM_DEFAULT_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

result = llm.invoke("你好")
print(result.content)
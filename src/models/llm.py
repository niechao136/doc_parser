import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


LLM_MODEL = os.getenv("LLM_MODEL", "gemma4")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY", "api_key")


llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY, # type: ignore
    temperature=0,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        },
        "reasoning_effort": "none",
        "skip_reasoning": True,
        "skip_special_tokens": True
    }
)
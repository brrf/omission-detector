import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30,
    max_retries=2)


def call_llm(model: str, prompt: str, temperature=0) -> tuple[str, dict]:
    """
    Centralized LLM caller.  
    """    
    resp = _client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":prompt}],
        temperature=temperature,
    )

    return resp.choices[0].message.content.strip()
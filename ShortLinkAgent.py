import os
from dataclasses import dataclass
import re
import httpx
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIModel

load_dotenv()

llm = os.getenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
key = os.getenv("LLM_MODEL_KEY")

# model = OpenAIModel(
#     model_name="llama3.2",
#     base_url='http://localhost:11434/v1',
#     api_key=key
# )
# model = OpenAIModel(
#     model_name="google/gemini-2.0-flash-001",
#     base_url='https://openrouter.ai/api/v1',
#     api_key=key
# )
# model = OpenAIModel(
#     model_name="MFDoom/deepseek-r1-tool-calling:latest",
#     base_url='http://localhost:11434/v1',
#     api_key=key
# )
model = GroqModel(
    model_name="deepseek-r1-distill-llama-70b",
    api_key=key
)
@dataclass
class Deps:
    client: httpx.AsyncClient


system_prompt = """
Prompt:

Your primary task is to shorten any URLs provided to you using the available tools and to answer questions related to short URLs.

Guidelines:
Always shorten URLs immediately when provided, without modifying them.
Never edit, reformat, or alter the shortened URL in any way.
If no URL is provided, politely ask the user for one.
If the user asks for details about a short URL, provide relevant information accurately.
If the user asks about anything else unrelated to URLs, respond with:
"Sorry, I can't help you with that."
Response Formatting:
Always return the shortened URL exactly as given by the API.
Format responses like this:
shortened URL
Interaction Style:
Be clear, friendly, and human-like when responding.
If the user greets you, greet them back naturally.
Your focus is only on URL shortening and related questions—nothing else.
"""

ShortLink_Agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=Deps,
    retries=2,
)

@ShortLink_Agent.tool
async def short_url(ctx: RunContext[Deps], url: str) -> str:
    """Shortens the given URL."""
    # Check if the URL format is valid using regex
    # if not re.match(r'https?://[^\s]+', url):
    #     return "Sorry, I can't help you with that."

    data = {'link': url}
    response = await ctx.deps.client.post(
        'https://jih.ltd/api/create',
        data=data,  # Use 'data' for form parameters
    )

    if response.status_code != 200:
        return f"Failed to shorten the link. Error: {response.status_code}"

    # If the response contains a valid shortened URL, return it
    json_response = response.json()
    if json_response:
        print(json_response)
        return f"[Shortened URL: {json_response}]"

    # return "Sorry, I can't help you with that."

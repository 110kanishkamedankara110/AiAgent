import os
from dataclasses import dataclass
import re
import httpx
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

load_dotenv()

llm = os.getenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
key = os.getenv("LLM_MODEL_KEY")

# model = OpenAIModel(
#     model_name="llama3.2",
#     base_url='http://localhost:11434/v1',
#     api_key=key
# )
model = OpenAIModel(
    model_name="google/gemini-2.0-flash-001",
    base_url='https://openrouter.ai/api/v1',
    api_key=key
)

@dataclass
class Deps:
    client: httpx.AsyncClient


system_prompt = """
Your main task is to shorten any URLs provided to you using the tools at your disposal. Never answer questions or take actions other than shortening URLs.

You should never ask the user anything before you act. As soon as you receive the URL, extract it and shorten it right away.

If the user wants to shorten a URL but hasn’t provided one, politely ask for the URL.

If the user asks for details about the URL, feel free to provide them.

If the user asks about anything else that’s not related to URLs, just reply with: "Sorry, I can't help you with that."

When you respond, keep it clear and friendly. Format your answer like this:

shortened URL

Always use the available tools to shorten the link, and never offer any other info.

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

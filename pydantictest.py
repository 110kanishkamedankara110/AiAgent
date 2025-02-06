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
key=os.getenv("LLM_MODEL_KEY")

model = OpenAIModel(
    model_name="llama3.2",
    base_url='http://localhost:11434/v1',
    api_key=key
)


@dataclass
class Deps:
    client: httpx.AsyncClient
    github_token: str | None = None


system_prompt = """
    you are a coding expert with access to Github to help the user manage their repository and get information from it,
    
    your only job is to assist with this and you dont answer other questions besides describing what you are able to do,

    Don't ask the user before taking an action, just do it. Always make sure you look at the repository with the provided tools before answering the user question,

    When answering a question about the repo, always start your answer with the full repo URL in the brackets and then give your answer on a new line. Like 
    
    [Using https://github.com/[repo URL from the user]]
    
    Your answer here...
"""

github_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=Deps,
    retries=2,
)


@github_agent.tool
async def get_repo_info(ctx: RunContext[Deps], github_url: str) -> str:
    """Get the repository information including size and description using GitHub API.

    Args:
         ctx: The context.
         github_url: The GitHub URL.

    Returns:
        str: The repository information as a formatted string.
    """

    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL"
    owner, repo = match.groups()

    headers = {"Authorization": f'token{ctx.deps.github_token}'} if ctx.deps.github_token else {}

    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}',
        headers=headers
    )

    if response.status_code != 200:
        return f"failed to get repository inf: {response.text}"

    data = response.json()
    size_mb = data['size'] / 1024

    return (
        f"Repository: {data['full_name']}\n"
        f"Description: {data['description']}\n"
        f"Size: {size_mb:.1f}MB\n"
        f"Stars: {data['stargazers_count']}\n"
        f"Language: {data['language']}\n"
        f"Created: {data['created_at']}\n"
        f"Last Updated: {data['updated_at']}"
    )


@github_agent.tool
async def get_repo_structure(ctx: RunContext[Deps], github_url: str) -> str:
    """Get the repository structure of a GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub URL.

    Returns:
        str: Directory structure as a formatted string.
    """

    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL"
    owner, repo = match.groups()

    headers = {"Authorization": f'token{ctx.deps.github_token}'} if ctx.deps.github_token else {}

    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1',
        headers=headers
    )
    if response.status_code != 200:
        response = await ctx.deps.client.get(
            f'https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1',
            headers=headers
        )
        if response.status_code != 200:
            print("notFound......")
            return f"failed to get repository structure: {response.text}"

    data = response.json()
    tree = data['tree']

    # Build directory structure
    structure = []
    for item in tree:
        if not any(excluded in item['path'] for excluded in ['.git/', 'node_modules/', '__pycache__/']):
            structure.append(f"{'📁 ' if item['type'] == 'tree' else '📄 '}{item['path']}")

    return "\n".join(structure)

@github_agent.tool
async def get_file_content(ctx: RunContext[Deps], github_url: str,file_path:str) -> str:
    """Get the content of a specific file from the GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub URL.
        file_path: Path to the file within the repository.

    Returns:
        str: File content as a string.
    """

    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL"
    owner, repo = match.groups()

    headers = {"Authorization": f'token{ctx.deps.github_token}'} if ctx.deps.github_token else {}

    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/main/{file_path}',
        headers=headers
    )
    if response.status_code != 200:
        response = await ctx.deps.client.get(
            f'https://api.github.com/repos/{owner}/{repo}/master/{file_path}',
            headers=headers
        )
        if response.status_code != 200:
            return f"failed to get file content: {response.text}"

    return response.text
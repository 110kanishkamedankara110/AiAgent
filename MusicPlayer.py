import os
from dataclasses import dataclass
import re
import httpx
from dotenv import load_dotenv
from filelock import asyncio
from flask import jsonify
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIModel
import subprocess

import json

import asyncio

load_dotenv()


def load_playlist():
    file_path = "pc_content.json"  # Replace with your actual file path
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


playlist = load_playlist()

key = os.getenv("LLM_MODEL_KEY")

model = GroqModel(
    model_name="deepseek-r1-distill-llama-70b",
    api_key=key
)

# model = OpenAIModel(
#     model_name="llama3.2",
#     base_url='http://localhost:11434/v1',
#     api_key=key
# )
@dataclass
class MusicDeps:
    client: httpx.AsyncClient


music_player_system_prompt = """
You are an expert music-playing agent designed to fetch playlist get song location, play them, and provide song-related information. 
Your primary function is to process user song requests and play them.

Your Rules & Responsibilities:
    getting song location:
        Always use the provided tools to get the song location.
        Never assume, modify, or make up song names—always use the exact name that given by the user.

    Playing Songs:
        Once the correct song location is fetched, play it using fetched location.
        Do not ask the user for confirmation before playing,execute the request immediately.


You do not answer any questions unrelated to song retrieval, playback, or information.
If a request is outside these tasks, ignore it.

Tool Usage:
    Always use the provided tools to fetch data and execute requests before responding to the user.
    
Your job is to only handle music-related tasks,
"""

location_fetch_system_prompt = """
you are an expert searcher, when you are provided with a partial song name search a fairly similar song location using that song name from the playlist,
    play list structure is
    [{
        "name": "song_name",
        "location": "song_location",
        "category": "song_category"
    },..]
    always fetch the location if the name vaguely similar to the name provided
    never ask any questions just return the song location if you find the song if not just return null
"""

music_player_agent = Agent(
    model,
    system_prompt=music_player_system_prompt,
    deps_type=MusicDeps,
    retries=2,
)

music_location_agent = Agent(
    model,
    system_prompt=location_fetch_system_prompt,
    deps_type=MusicDeps,
    retries=2,
)


@music_player_agent.tool()
async def get_song_location(ctx: RunContext, name: str):
    """Get the playlist from the pc.

    Args:
         ctx: The context.

    Returns:
        str: user requested song name.
        name: name of the song
    """
    import json

    query = f"""
    fetch the song
    playlist:
    {json.dumps(playlist, indent=4)}  
    Song name:
    {name}
    """

    result = await music_location_agent.run(
        query,
        deps=ctx.deps
    )
    return result

@music_player_agent.tool
async def get_playlist(ctx: RunContext[MusicDeps]) -> str:
    """Get the playlist from the pc.

    Args:
         ctx: The context.
    Returns:
        str: The playlist as a formatted string.
    """
    return f"""{playlist}"""


@music_player_agent.tool
async def play_song(ctx: RunContext[MusicDeps], location: str):
    """Play The Song.

       Args:
            ctx: The context.
            location: location of the song
       Returns:
            str: song status
    """
    print("playing song...", location)
    os.startfile(location)
    return "song is playing"

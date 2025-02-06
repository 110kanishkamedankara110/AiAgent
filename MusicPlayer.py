import os
from dataclasses import dataclass
import re
import httpx
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIModel
import subprocess

import json

load_dotenv()


def load_playlist():
    file_path = "pc_content.json"  # Replace with your actual file path
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

playlist=load_playlist()




llm = os.getenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
key = os.getenv("LLM_MODEL_KEY")

model = GroqModel(
    model_name="llama3-70b-8192",
    api_key=key
)

@dataclass
class MusicDeps:
    client: httpx.AsyncClient

system_prompt = """
You are an expert music-playing agent designed to fetch songs, play them, and provide song-related information. Your primary function is to process user song requests, even when they provide only partial names, and correctly match them to the full song name from the playlist.

Your Rules & Responsibilities:
    Accurate Song Matching:
        Always use the provided playlist to find the correct full name of the song based on the user’s partial input.
        
        Never assume, modify, or make up song names—always use the exact name as it appears in the playlist.

    Playing Songs:

        Once the correct song is identified, play it using its exact name from the playlist.
        
        Do not ask the user for confirmation before playing—execute the request immediately.


You do not answer any questions unrelated to song retrieval, playback, or information.
If a request is outside these tasks, ignore it.
Tool Usage:

Always use the provided tools to fetch data and execute requests before responding to the user.
Never attempt to process a song request without first retrieving the exact song name from the playlist.
Your job is to only handle music-related tasks, ensuring that every song request is matched and executed with the correct name from the playlist.
"""

music_player_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=MusicDeps,
    retries=2,
)



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
async def play_song(ctx: RunContext[MusicDeps], partial_name: str):
    """Play The Song.

       Args:
            ctx: The context.
            partial_name: name of the song
       Returns:
            str: song status
    """

    for song in playlist:
        song_name = song["name"]
        request_name=partial_name

        if song_name.lower() in request_name.lower():
            location = song["location"]
            print("playing song...",location)
            return "song is playing"
    return f"Error playing song: cannot find the song"









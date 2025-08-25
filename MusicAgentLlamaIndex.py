import os
from dataclasses import dataclass
import re
import httpx
from dotenv import load_dotenv
from phi.model.groq import Groq
import json
from phi.agent import Agent

load_dotenv()


def load_playlist():
    file_path = "pc_content.json"  # Replace with your actual file path
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


playlist = load_playlist()

key = os.getenv("LLM_MODEL_KEY")

llm = Groq(id="openai/gpt-oss-120b", api_key=key)


@dataclass
class MusicDeps:
    client: httpx.AsyncClient


# ---- STATE FIX ----
current_song = None
is_playing = False


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
        when you receive string like "playing song... " from the a tool stop calling tools 

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


def get_song_location(name: str) -> str:
    """Get the playlist from the pc."""
    result = Agent(
        model=llm,
        tools=[get_playlist],
        description=location_fetch_system_prompt,
        markdown=True,
        show_tool_calls=True,
    ).run(f"""
        fetch the location from the playlist 
        Song name : {name}
        """)

    print("location...", str(result.content))
    return str(result)


def get_playlist(**kwargs) -> str:
    """Get the playlist from the pc."""
    try:
        return json.dumps(playlist, ensure_ascii=False)
    except Exception:
        return "[]"


def play_song(location: str) -> str:
    """Play The Song without repeating unnecessarily."""
    global is_playing, current_song

    if is_playing and current_song == location:
        return f"Song is already playing: {location}"

    try:
        print("playing song...", location)
        os.startfile(location)  # non-blocking
        is_playing = True
        current_song = location
        return f"Now playing: {location}"
    except FileNotFoundError:
        ""
    except OSError:
        ""



# Optionally: reset state when user requests "stop" or "next"
def stop_song():
    global is_playing, current_song
    is_playing = False
    current_song = None
    return "Song stopped"


PhiDataAgent = Agent(
    tools=[get_song_location, play_song, stop_song],
    model=llm,
    description=music_player_system_prompt,
    markdown=True,
    show_tool_calls=True,
)

try:
    while True:
        user_input = input("🎵 Enter a command (or type 'exit' to quit): ")

        if user_input.lower() in ["exit", "quit", "stop"]:
            print("👋 Exiting music player...")
            break

        response = PhiDataAgent.run(user_input)
        print(response.content)

except Exception as e:
    ""


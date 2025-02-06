from os import system

import requests
from cohere.types import content

class Agent:
    def __init__(self,system):
        self.system = system
        self.messages=[]
        if(self.system is not None):
            self.messages.append({"role":"system","content":self.system})
    def __call__(self, message):
        if message:
            self.messages.append({"role":"user","content":message})
        result=self.execute()
        self.messages.append({"role":"assistant","content":result})
        return result

    def agent_llama(selff,prompt):
        url = "http://localhost:11434/api/generate"
        payload = {"model": "llama3", "prompt": prompt, "stream": False}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            print(f"❌ Error contacting LLaMA API: {e}")
            return ""
    def execute(self):
        return self.agent_llama(message_query(self.messages))



def message_query(queries: list) -> str:
    result = "";
    for query in queries:
        result += f"""<|start_header_id|>{query['role']}<|end_header_id|>\n{query['content']}\n<|eot_id|>"""
    return result



system_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

get_playlist:
e.g. get_playlist
returns the playlist

retrieve_song_location:
e.g. retrieve_song_location: africa by toto.mp4
returns song location of the song using the name in the playlist 

play_song
e.g. play_song: E://songs/africa by toto.mp4
plays the song using the location



""".strip()



def get_playlist() -> list:
    return ["africa by toto.mp4","call me maybe.mp4","last stand-hd-1088p (hobbit soundtrack).mp4"]

def retrieve_song_location(name) -> str:
    return f"E//songs/{name}"

def play_song(song_location) -> str:
    return f"playing {song_location}"

import re


def loop(max_iterations=10, query: str = ""):
    agent = Agent(system=system_prompt)

    tools = ["get_playlist", "retrieve_song_location","play_song"]

    next_prompt = query

    i = 0

    while i < max_iterations:
        i += 1
        result = agent(next_prompt)
        print(result)

        if "PAUSE" in result and "Action" in result:
            action = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
            if action:  # Ensure action is not empty
                chosen_tool = action[0][0]  # Always exists if action is not empty
                arg = action[0][1] if len(action[0]) > 1 else None  # Check if arg exists
            else:
                chosen_tool = None
                arg = None
            if chosen_tool in tools:
                result_tool = eval(f"{chosen_tool}('{arg}')")
                next_prompt = f"Observation: {result_tool}"

            else:
                next_prompt = "Observation: Tool not found"

            print(next_prompt)
            continue

        if "Answer" in result:
            break


loop(query="play last stand")



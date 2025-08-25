import asyncio
from typing import List
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic_ai.messages import UserPromptPart, ModelRequest, ModelMessage, ModelResponse, TextPart
import httpx

from MusicPlayer import music_player_agent
from ShortLinkAgent import ShortLink_Agent, Deps

app = Flask(__name__)
CORS(app)
deps = Deps(
    client=httpx.AsyncClient(),
)

@app.route('/ask', methods=['POST'])
def receive_data():
    data = request.get_json()  # Get JSON data from request body
    if not data:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    query = data.get("query")

    messages: List[ModelMessage] = []
    historyList = data.get("history")

    if historyList:
        for history in historyList:
            if history.get("role") == 'user':
                messages.append(
                    ModelRequest(parts=[UserPromptPart(content=history.get("message"))])
                )
            elif history.get("role") == 'bot':
                messages.append(
                    ModelResponse(parts=[TextPart(content=history.get("message"))])
                )
    result = asyncio.run(ShortLink_Agent.run(
        query,
        deps=deps,
        message_history=messages,
    ))

    return jsonify(result.data)

if __name__ == '__main__':
    app.run(debug=True)

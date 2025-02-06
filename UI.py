from typing import List

import httpx
import streamlit as st
import asyncio

from dotenv import load_dotenv
from pydantic_ai.messages import UserPromptPart, ModelRequest, ModelMessage, ModelResponse, TextPart

from ShortLinkAgent import ShortLink_Agent, Deps


load_dotenv()


async def main():
    st.title("Short Link Agent")

    # Initialize session state messages if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if isinstance(message, ModelRequest):
            role = "user"
            # Extract content from UserPromptPart
            content = next((part.content for part in message.parts if isinstance(part, UserPromptPart)), None)
        elif isinstance(message, ModelResponse):
            role = "assistant"
            # Extract content from TextPart
            content = next((part.content for part in message.parts if isinstance(part, TextPart)), None)
        else:
            print("Warning: Unrecognized message type. Skipping.")
            continue

        # Ensure content is present before displaying
        if content:
            with st.chat_message("human" if role == "user" else "ai"):
                st.markdown(content)
        else:
            print(f"Warning: Message content is missing or empty for {role}. Skipping.")

        # Handle new user input
    if prompt := st.chat_input("What would you like to shorten today?"):
        # Display user message in chat
        st.chat_message("user").markdown(prompt)

        # Add user input to chat history
        st.session_state.messages.append(ModelRequest(parts=[UserPromptPart(content=prompt)]))

        # Generate and display response
        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response_content = await UI().chat(message=prompt)
            message_placeholder.markdown(response_content.data)

        # Add response to chat history
        st.session_state.messages.append(ModelResponse(parts=[TextPart(content=response_content.data)]))


class UI:
    def __init__(self):
        self.messages: List[ModelMessage] = []
        self.deps = Deps(
            client=httpx.AsyncClient(),
        )

    async def chat(self,message:str):
            user_input = message

            result=await ShortLink_Agent.run(
                user_input,
                deps=self.deps,
                message_history=self.messages,
            )

            self.messages.append(
                ModelRequest(parts=[UserPromptPart(content=user_input)])
            )

            filtered_messages = [msg for msg in result.new_messages()
                                     if not (hasattr(msg, 'parts') and
                                            any(part.part_kind == 'user-prompt' or part.part_kind == 'text' for part in
                                                 msg.parts))]
            self.messages.extend(filtered_messages)


            self.messages.append(
                ModelResponse(parts=[TextPart(content=result.data)])
            )
            return result


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())

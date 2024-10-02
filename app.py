"""Streamlit Playground"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import boto3
import streamlit as st

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient

IMAGE_FILENAME = "trial chamber.png"
# MODEL_ID = "mistral.mistral-large-2402-v1:0"
MODEL_ID = "amazon.titan-text-express-v1"
INFERRENCE_CONFIG = {
    "temperature": 0.3,
    "topP": 0.8,
    "maxTokens": 256,
}


def main():
    st.set_page_config("Streamlit Playground", layout="wide")

    left, right = st.columns((1, 1))

    with right:
        self_download()

    with left:
        ask_the_ai()


def self_download():
    with open(__file__, encoding="utf-8") as fin:
        my_contents = fin.read()

    my_name = os.path.basename(__file__)

    st.subheader(my_name)
    st.code(my_contents)
    st.download_button("Download Source", my_contents, file_name=my_name)


@st.cache_resource
def get_boto_session() -> boto3.Session:
    return boto3.Session()


@st.cache_resource
def get_bedrock_client() -> BedrockRuntimeClient:
    return get_boto_session().client("bedrock-runtime")


def ask_the_ai():
    conversation_container = st.container(border=True)
    inspect_container = st.container(border=True)

    with inspect_container:
        st.subheader("Debugging")
        session_state_container = st.empty()

    def inspect_state():
        with session_state_container, st.container():
            st.write("### Session State")
            st.write(st.session_state)

    inspect_state()

    with conversation_container:
        st.subheader("Conversation")
        query = st.text_area(label="Ask the AI!", key="chat_with_ai_query")

    inspect_state()

    with conversation_container:
        response = None
        if query:
            with st.spinner("Consulting the AI..."):
                client = get_bedrock_client()
                response = client.converse(
                    modelId=MODEL_ID,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"text": query},
                            ],
                        }
                    ],
                )

    if response:
        inspect_state()

        with inspect_container:
            st.write("### Response")
            st.write(response)

        with conversation_container:
            st.write("response received!")
            for text in extract_assistant_text(response):
                st.text(text)


def make_user_message(*text_items: str) -> dict:
    return {"role": "user", "content": [{"text": item for item in text_items}]}


_marker = object()


def dig(obj, *keys: str | int, default=_marker):
    for key in keys:
        try:
            obj = obj[key]
        except (TypeError, ValueError, IndexError, KeyError, AttributeError) as exc:
            break
    else:
        # item found!
        return obj

    # item not found...
    if default is not _marker:
        return default

    raise LookupError(*keys)


def extract_assistant_text(response: dict) -> list[str]:
    return [
        item["text"]
        for item in dig(response, "output", "message", "content", default=())
        if "text" in item
    ]


main()

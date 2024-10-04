"""Streamlit Bedrock Playground"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import TYPE_CHECKING

import boto3
import streamlit as st

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime import type_defs as brtd

TITLE = "Streamlit Bedrock Playground"

MODEL_IDS = [
    "mistral.mistral-small-2402-v1:0",
    "mistral.mistral-large-2402-v1:0",
    "amazon.titan-text-lite-v1",
    "amazon.titan-text-express-v1",
    "amazon.titan-text-premier-v1:0",
]


def main():
    st.set_page_config(TITLE, layout="wide")

    render_sidebar()

    left, right = st.columns((1, 1))

    with left, st.container(border=True):
        params = params_dialog()
        submit = st.button("Submit")

    with right:
        response = st.session_state.get("response")

        response_container = st.container()
        inspect_container = st.container(border=True)

        if response:
            with response_container:
                render_response(response)

        with inspect_container:
            st.subheader("Parameters")
            st.write(params)

            if response:
                st.subheader("Response")
                st.write(response)

        if submit:
            with response_container:
                response = get_response(params)
                st.session_state.response = response
                st.rerun()


def get_response(params) -> brtd.ConverseResponseTypeDef:
    with st.spinner("Consulting the AI..."):
        return get_bedrock_client().converse(**params)


def render_response(response) -> None:
    st.subheader("Response Received!")
    text_response = "\n\n".join(extract_assistant_text(response))
    st.code(text_response, wrap_lines=True, language=None)
    st.download_button("Download Response", text_response, file_name="response.txt")


def render_sidebar():
    with st.sidebar:
        st.header(TITLE)
        st.write("Play around and build the perfect prompt.")
        self_download()


def params_dialog() -> dict:
    model_id = st.selectbox("Model ID", options=MODEL_IDS, key="model_id")
    prompt = st.text_area("Prompt")
    system = st.text_area("System (optional)")
    inference = inference_dialog()

    params = {
        "modelId": model_id,
        "messages": [make_user_message(prompt)],
    }

    if inference:
        params["inferenceConfig"] = inference

    if system:
        params["system"] = [{"text": system}]

    return params


def inference_dialog(max_max_tokens: int = 8192) -> brtd.InferenceConfigurationTypeDef | None:
    inference: brtd.InferenceConfigurationTypeDef = {}
    with st.expander("Inference Parameters"):
        left, right = st.columns((1, 2))
        set_temperature = left.toggle("Set Temperature", key="set_temperature")
        temperature = right.slider("Temperature", min_value=0.0, max_value=1.0, disabled=(not set_temperature))
        if set_temperature:
            inference["temperature"] = temperature

        left, right = st.columns((1, 2))
        set_top_P = left.toggle("Set topP", key="set_top_p")
        top_P = right.slider("topP", min_value=0.0, max_value=1.0, disabled=(not set_top_P))
        if set_top_P:
            inference["topP"] = top_P

        left, right = st.columns((1, 2))
        set_max_tokens = left.toggle("Set maxTokens", key="set_max_tokens")
        max_tokens = right.slider("maxTokens", min_value=0, max_value=max_max_tokens, disabled=(not set_max_tokens))
        if set_max_tokens:
            inference["maxTokens"] = max_tokens

    return inference or None


def self_download():
    with open(__file__, encoding="utf-8") as fin:
        my_contents = fin.read()

    my_name = os.path.basename(__file__)
    st.download_button("Download This App", my_contents, file_name=my_name)


@st.cache_resource
def get_boto_session() -> boto3.Session:
    return boto3.Session()


@st.cache_resource
def get_bedrock_client() -> BedrockRuntimeClient:
    return get_boto_session().client("bedrock-runtime")


def make_user_message(*text_items: str) -> dict:
    return {"role": "user", "content": [{"text": item for item in text_items}]}


def extract_assistant_text(response: brtd.ConverseResponseTypeDef) -> Generator[str]:
    assert (output := response.get("output"))
    assert (message := output.get("message"))
    assert (content := message.get("content"))
    for item in content:
        assert (text := item.get("text"))
        yield text


main()

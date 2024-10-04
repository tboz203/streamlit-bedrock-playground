"""Streamlit Bedrock Playground"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator, Iterator
from io import StringIO, TextIOBase
from typing import TYPE_CHECKING

import boto3
import streamlit as st

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime import type_defs as brtd

logger = logging.getLogger("app")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d %(levelname)8s %(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
        streaming = st.toggle("Stream response")
        submit = st.button("Submit")

    with right:
        if submit and streaming:
            generate_and_render_streaming_response(params)
        else:
            generate_and_render_response(params, submit)

    # make the (cached) bedrock client after everything else is rendered,
    # so it is available immediately on subsequent renders
    get_bedrock_client()


def generate_and_render_response(params, submit: bool) -> None:
    response = st.session_state.get("response")

    response_container = st.container()
    inspect_container = st.container(border=True)

    if response:
        with response_container:
            st.subheader("Response Received!")
            text_response = "\n\n".join(extract_assistant_text(response))
            # st.code(text_response, wrap_lines=True, language=None)
            st.markdown(text_response)
            st.download_button("Download Response", text_response, file_name="response.txt")

    with inspect_container:
        st.subheader("Parameters")
        st.write(params)

        if response:
            st.subheader("Response")
            st.write(response)

    if submit:
        with response_container.spinner("Consulting the AI..."):
            response = get_bedrock_client().converse(**params)
        st.session_state.response = response
        st.rerun()


def generate_and_render_streaming_response(params) -> None:
    response_container = st.container()
    inspect_container = st.container(border=True)
    collector_stream = StringIO()

    with inspect_container:
        st.subheader("Parameters")
        st.write(params)

    with response_container:
        st.subheader("Awaiting response...")
        response = get_bedrock_client().converse_stream(**params)

        st.session_state.response = response
        inspect_container.subheader("Response")
        inspect_container.write(response)

        stream = stream_wrapper(response["stream"], collector_stream)
        st.write_stream(stream)

        collector_stream.seek(0)
        response["stream"] = collector_stream

        text_response = collector_stream.getvalue()
        st.download_button("Download Response", text_response, file_name="response.txt")

    st.rerun()


def stream_wrapper(
    stream: Iterator[brtd.ConverseStreamResponseTypeDef | str],
    write_copy: TextIOBase | None = None,
) -> Generator[str]:
    for event in stream:
        match event:
            case {"contentBlockDelta": {"delta": {"text": text}}} | str(text):
                if write_copy is not None:
                    write_copy.write(text)
                yield text

    if hasattr(stream, "seek") and callable(stream.seek):
        stream.seek(0)


# def stream_event_print_handler(event: brtd.ConverseStreamResponseTypeDef) -> None:
#     match event:
#         case {"messageStart": {"role": role}}:
#             print(f"\nRole: {role}")
#         case {"messageStop": {"stopReason": reason}}:
#             print(f"\nStop reason: {reason}")

#         case {"metadata": metadata}:
#             print("\nMetadata:")
#             if usage := metadata.get("usage"):
#                 print(f"Input tokens: {usage['inputTokens']}")
#                 print(f"Output tokens: {usage['outputTokens']}")
#                 print(f"Total tokens: {usage['totalTokens']}")
#             if metrics := metadata.get("metrics"):
#                 print(f"Latency: {metrics['latencyMs']} milliseconds")

#         case {"contentBlockDelta": {"delta": {"text": text}}}:
#             print(text, end="")


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


@st.fragment
def self_download():
    my_name = os.path.basename(__file__)
    with open(__file__) as fin:
        st.download_button("Download This App", fin, file_name=my_name)


@st.cache_resource
def get_bedrock_client() -> BedrockRuntimeClient:
    return boto3.client("bedrock-runtime")


def make_user_message(*text_items: str) -> dict:
    return {"role": "user", "content": [{"text": item for item in text_items}]}


def extract_assistant_text(response: brtd.ConverseResponseTypeDef) -> Generator[str]:
    if stream := response.get("stream"):
        yield from stream_wrapper(stream)
        return

    assert (output := response.get("output"))
    assert (message := output.get("message"))
    assert (content := message.get("content"))
    for item in content:
        assert (text := item.get("text"))
        yield text


main()

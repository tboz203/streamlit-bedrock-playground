# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to use the Converse API to stream a response from Anthropic Claude 3 Sonnet (on demand).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from botocore.eventstream import EventStream
    from mypy_boto3_bedrock_runtime import type_defs as brtd

logger = logging.getLogger("streaming_example")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d %(levelname)8s %(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
# MODEL_ID = "amazon.titan-text-express-v1"
MODEL_ID = "mistral.mistral-small-2402-v1:0"


def stream_conversation(
    stream: EventStream[brtd.ConverseStreamResponseTypeDef],
) -> Generator[brtd.ConverseStreamResponseTypeDef]:
    for event in stream:
        match event:
            case {"messageStart": {"role": role}}:
                print(f"\nRole: {role}")
            case {"messageStop": {"stopReason": reason}}:
                print(f"\nStop reason: {reason}")

            case {"metadata": metadata}:
                print("\nMetadata:")
                if usage := metadata.get("usage"):
                    print(f"Input tokens: {usage['inputTokens']}")
                    print(f"Output tokens: {usage['outputTokens']}")
                    print(f"Total tokens: {usage['totalTokens']}")
                if metrics := metadata.get("metrics"):
                    print(f"Latency: {metrics['latencyMs']} milliseconds")

            case {"contentBlockDelta": {"delta": {"text": text}}}:
                print(text, end="")

        yield event


def main():
    """
    Entrypoint for streaming message API response example.
    """

    system_prompt = (
        "You are an app that creates playlists for a radio station that plays "
        "rock and pop music. Only return song names and the artist."
    )

    # Message to send to the model.
    input_text = "Create a list of 3 pop songs."

    message = {
        "role": "user",
        "content": [
            {
                "text": input_text,
            }
        ],
    }
    messages = [message]
    system_prompts = [
        {
            "text": system_prompt,
        }
    ]

    try:
        logger.info("Streaming messages with model %s", MODEL_ID)

        bedrock_client = boto3.client(service_name="bedrock-runtime")

        response = bedrock_client.converse_stream(
            modelId=MODEL_ID,
            messages=messages,
            system=system_prompts,
        )

        events = list(stream_conversation(response["stream"]))

        with open("event-stream.json", "w") as fout:
            json.dump(events, fout, indent=2)

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
    else:
        print(f"Finished streaming messages with model {MODEL_ID}.")


if __name__ == "__main__":
    main()

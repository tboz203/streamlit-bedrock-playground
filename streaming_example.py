# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to use the Converse API to stream a response from Anthropic Claude 3 Sonnet (on demand).
"""

from __future__ import annotations

import logging
import boto3
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import type_defs as brtd
    from botocore.eventstream import EventStream

logger = logging.getLogger("streaming_example")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d %(levelname)8s %(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
# MODEL_ID = "amazon.titan-text-express-v1"
MODEL_ID = "mistral.mistral-small-2402-v1:0"


def stream_conversation(stream: EventStream[brtd.ConverseStreamResponseTypeDef]) -> None:
    for event in stream:
        if "messageStart" in event:
            print(f"\nRole: {event['messageStart']['role']}")

        if "contentBlockDelta" in event:
            print(event["contentBlockDelta"]["delta"]["text"], end="")

        if "messageStop" in event:
            print(f"\nStop reason: {event['messageStop']['stopReason']}")

        if "metadata" in event:
            metadata = event["metadata"]
            if "usage" in metadata:
                print("\nToken usage")
                print(f"Input tokens: {metadata['usage']['inputTokens']}")
                print(f":Output tokens: {metadata['usage']['outputTokens']}")
                print(f":Total tokens: {metadata['usage']['totalTokens']}")
            if "metrics" in event["metadata"]:
                print(f"Latency: {metadata['metrics']['latencyMs']} milliseconds")


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

        stream_conversation(response["stream"])

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))

    else:
        print(f"Finished streaming messages with model {MODEL_ID}.")


if __name__ == "__main__":
    main()

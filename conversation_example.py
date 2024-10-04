# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to use the Converse API with Anthropic Claude 3 Sonnet (on demand).
"""

from __future__ import annotations

import logging
import boto3
from collections.abc import Sequence

from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime import type_defs as brtd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"


def generate_conversation(
    bedrock_client: BedrockRuntimeClient,
    model_id: str,
    system_prompts: Sequence[brtd.SystemContentBlockTypeDef],
    messages: Sequence[brtd.MessageTypeDef],
) -> brtd.ConverseResponseTypeDef:
    """
    Sends messages to a model.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        system_prompts (JSON) : The system prompts for the model to use.
        messages (JSON) : The messages to send to the model.

    Returns:
        response (JSON): The conversation that the model generated.

    """

    logger.info("Generating message with model %s", model_id)

    # Inference parameters to use.
    temperature = 0.5
    top_k = 200

    # Base inference parameters to use.
    inference_config: brtd.InferenceConfigurationTypeDef = {"temperature": temperature}
    # Additional inference parameters to use.
    additional_model_fields = {"top_k": top_k}

    # Send the message.
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields,
    )

    # Log token usage.
    token_usage = response["usage"]
    logger.info("Input tokens: %s", token_usage["inputTokens"])
    logger.info("Output tokens: %s", token_usage["outputTokens"])
    logger.info("Total tokens: %s", token_usage["totalTokens"])
    logger.info("Stop reason: %s", response["stopReason"])

    return response


def main():
    """
    Entrypoint for Anthropic Claude 3 Sonnet example.
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Setup the system prompts and messages to send to the model.
    system_prompts: list[brtd.SystemContentBlockTypeDef] = [
        {
            "text": (
                "You are an app that creates playlists for a radio station that plays rock and pop music. "
                "Only return song names and the artist."
            )
        }
    ]
    message_1: brtd.MessageTypeDef = {
        "role": "user",
        "content": [
            {
                "text": "Create a list of 3 pop songs.",
            }
        ],
    }
    message_2: brtd.MessageTypeDef = {
        "role": "user",
        "content": [
            {
                "text": "Make sure the songs are by artists from the United Kingdom.",
            }
        ],
    }
    messages = []

    try:
        bedrock_client = boto3.client(service_name="bedrock-runtime")

        # Start the conversation with the 1st message.
        messages.append(message_1)
        response = generate_conversation(bedrock_client, MODEL_ID, system_prompts, messages)

        # Add the response message to the conversation.
        assert "message" in response["output"]
        messages.append(response["output"]["message"])

        # Continue the conversation with the 2nd message.
        messages.append(message_2)
        response = generate_conversation(bedrock_client, MODEL_ID, system_prompts, messages)

        assert "message" in response["output"]
        messages.append(response["output"]["message"])

        # Show the complete conversation.
        for message in messages:
            print(f"Role: {message['role']}")
            for content in message["content"]:
                print(f"Text: {content['text']}")
            print()

    except ClientError as err:
        error = err.response.get("Error")
        assert error
        message = error.get("Message")
        assert message
        logger.error("A client error occurred: %s", message)
        print(f"A client error occured: {message}")

    else:
        print(f"Finished generating text with model {MODEL_ID}.")


if __name__ == "__main__":
    main()

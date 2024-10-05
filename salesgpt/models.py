import re
import json
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
import aioboto3

from salesgpt.tools import completion_bedrock


class BedrockCustomModel(ChatOpenAI):
    """A custom chat model that generates responses using Bedrock's completion service.

    Attributes:
        model (str): The identifier for the Bedrock model.
        system_prompt (str): A prompt that provides context for the model.
    """

    model: str
    system_prompt: str

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generates a response based on the provided messages.

        Args:
            messages: A list of messages including user input.
            stop: Optional list of stop strings.
            run_manager: Callback manager for LLM execution.

        Returns:
            ChatResult: The generated response encapsulated in a ChatResult object.
        """
        last_message = messages[-1]  # Get the last user message

        print(messages)  # Debugging output
        response = completion_bedrock(
            model_id=self.model,
            system_prompt=self.system_prompt,
            messages=[{"content": last_message.content, "role": "user"}],
            max_tokens=1000,
        )
        print("output", response)  # Debugging output
        
        content = response["content"][0]["text"]
        message = AIMessage(content=content)  # Create AI message from response
        generation = ChatGeneration(message=message)  # Wrap in ChatGeneration
        return ChatResult(generations=[generation])  # Return result

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generates a response based on the provided messages.

        Args:
            messages: A list of messages including user input.
            stop: Optional list of stop strings.
            run_manager: Callback manager for asynchronous LLM execution.
            stream: Optional boolean to indicate streaming.

        Returns:
            ChatResult: The generated response encapsulated in a ChatResult object.
        """
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            raise NotImplementedError("Streaming not implemented")
        
        last_message = messages[-1]  # Get the last user message

        print(messages)  # Debugging output
        response = await acompletion_bedrock(
            model_id=self.model,
            system_prompt=self.system_prompt,
            messages=[{"content": last_message.content, "role": "user"}],
            max_tokens=1000,
        )
        print("output", response)  # Debugging output
        
        content = response["content"][0]["text"]
        message = AIMessage(content=content)  # Create AI message from response
        generation = ChatGeneration(message=message)  # Wrap in ChatGeneration
        return ChatResult(generations=[generation])  # Return result

async def acompletion_bedrock(model_id: str, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 1000) -> Dict[str, Any]:
    """Asynchronously generates a message with Bedrock's completion service.

    Args:
        model_id: The identifier for the Bedrock model.
        system_prompt: A prompt that provides context for the model.
        messages: A list of message dictionaries for the model input.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        Dict[str, Any]: The response body containing the generated message.
    """
    session = aioboto3.Session()  # Create an async session with Boto3
    async with session.client(service_name="bedrock-runtime", region_name=os.environ.get("AWS_REGION_NAME")) as bedrock_runtime:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": messages,
            }
        )

        response = await bedrock_runtime.invoke_model(body=body, modelId=model_id)

        # Handle the streaming body
        response_body_bytes = await response['body'].read()
        response_body = json.loads(response_body_bytes.decode("utf-8"))

        return response_body

from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.base import Chain
from pydantic import Field

from salesgpt.chains import SalesConversationChain, StageAnalyzerChain
from salesgpt.custom_invoke import CustomAgentExecutor
from salesgpt.logger import time_logger
from salesgpt.parsers import SalesConvoOutputParser
from salesgpt.prompts import SALES_AGENT_TOOLS_PROMPT
from salesgpt.stages import CONVERSATION_STAGES
from salesgpt.templates import CustomPromptTemplateForTools
from salesgpt.tools import get_tools, setup_knowledge_base
import os
import requests

# Hugging Face BART model integration
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}


def query_huggingface(payload):
    """
    Send a request to Hugging Face's model API and return the response.

    Args:
        payload (dict): Payload with input text.

    Returns:
        dict: The response from the model.
    """
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def _create_retry_decorator(llm: Any) -> Callable[[Any], Any]:
    """
    Creates a retry decorator for handling API errors for Hugging Face API.

    Args:
        llm (Any): A generic object (here it's for compatibility reasons).

    Returns:
        Callable[[Any], Any]: A retry decorator.
    """
    import time
    import random

    def retry_decorator(func):
        async def wrapper(*args, **kwargs):
            retries = 3
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    print(f"Error during attempt {attempt + 1}: {str(e)}")
                    if attempt + 1 == retries:
                        raise
                    time.sleep(random.uniform(1, 3))
            return None
        return wrapper

    return retry_decorator


class SalesGPT(Chain):
    """Controller model for the Sales Agent."""

    conversation_history: List[str] = []
    conversation_stage_id: str = "1"
    current_conversation_stage: str = CONVERSATION_STAGES.get("1")
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_agent_executor: Union[CustomAgentExecutor, None] = Field(...)
    knowledge_base: Union[RetrievalQA, None] = Field(...)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)
    conversation_stage_dict: Dict = CONVERSATION_STAGES

    model_name: str = "facebook/bart-large"  # Using Hugging Face's BART model

    use_tools: bool = False
    salesperson_name: str = "Ted Lasso"
    salesperson_role: str = "Business Development Representative"
    company_name: str = "Sleep Haven"
    company_business: str = "Sleep Haven is a premium mattress company that provides customers with the most comfortable and supportive sleeping experience possible. We offer a range of high-quality mattresses, pillows, and bedding accessories that are designed to meet the unique needs of our customers."
    company_values: str = "Our mission at Sleep Haven is to help people achieve a better night's sleep by providing them with the best possible sleep solutions. We believe that quality sleep is essential to overall health and well-being, and we are committed to helping our customers achieve optimal sleep by offering exceptional products and customer service."
    conversation_purpose: str = "find out whether they are looking to achieve better sleep via buying a premier mattress."
    conversation_type: str = "call"

    def retrieve_conversation_stage(self, key):
        """
        Retrieves the conversation stage based on the provided key.

        Args:
            key (str): The key to look up in the conversation_stage_dict dictionary.

        Returns:
            str: The conversation stage corresponding to the key, or "1" if the key is not found.
        """
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        """
        Property that returns a list of input keys.

        Returns:
            List[str]: An empty list.
        """
        return []

    @property
    def output_keys(self) -> List[str]:
        """
        Property that returns a list of output keys.

        Returns:
            List[str]: An empty list.
        """
        return []

    @time_logger
    def seed_agent(self):
        """
        This method seeds the conversation by setting the initial conversation stage and clearing the conversation history.

        Returns:
            None
        """
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    @time_logger
    def determine_conversation_stage(self):
        """
        Determines the current conversation stage based on the conversation history.

        Returns:
            None
        """
        print(f"Conversation Stage ID before analysis: {self.conversation_stage_id}")
        print("Conversation history:")
        print(self.conversation_history)

        payload = {
            "inputs": {
                "conversation_history": "\n".join(self.conversation_history).rstrip(
                    "\n"
                ),
                "conversation_stage_id": self.conversation_stage_id,
                "conversation_stages": "\n".join(
                    [
                        str(key) + ": " + str(value)
                        for key, value in CONVERSATION_STAGES.items()
                    ]
                ),
            }
        }

        stage_analyzer_output = query_huggingface(payload)
        print("Stage analyzer output")
        print(stage_analyzer_output)

        self.conversation_stage_id = stage_analyzer_output.get("text", "1")
        self.current_conversation_stage = self.retrieve_conversation_stage(
            self.conversation_stage_id
        )

        print(f"Conversation Stage: {self.current_conversation_stage}")

@time_logger
async def adetermine_conversation_stage(self):
    print(f"Conversation Stage ID before analysis: {self.conversation_stage_id}")
    print("Conversation history:")
    print(self.conversation_history)
    stage_analyzer_output = await self.stage_analyzer_chain.ainvoke(
        input={
            "conversation_history": "\n".join(self.conversation_history).rstrip("\n"),
            "conversation_stage_id": self.conversation_stage_id,
            "conversation_stages": "\n".join(
                [str(key) + ": " + str(value) for key, value in CONVERSATION_STAGES.items()]
            ),
        },
        return_only_outputs=False,
    )
    print("Stage analyzer output")
    print(stage_analyzer_output)
    self.conversation_stage_id = stage_analyzer_output.get("text")
    self.current_conversation_stage = self.retrieve_conversation_stage(self.conversation_stage_id)
    print(f"Conversation Stage: {self.current_conversation_stage}")

def human_step(self, human_input):
    human_input = "User: " + human_input + " <END_OF_TURN>"
    self.conversation_history.append(human_input)

@time_logger
def step(self, stream: bool = False):
    if not stream:
        return self._call(inputs={})
    else:
        return self._streaming_generator()

@time_logger
async def astep(self, stream: bool = False):
    if not stream:
        return await self.acall(inputs={})
    else:
        return await self._astreaming_generator()

@time_logger
async def acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    inputs = {
        "input": "",
        "conversation_stage": self.current_conversation_stage,
        "conversation_history": "\n".join(self.conversation_history),
        "salesperson_name": self.salesperson_name,
        "salesperson_role": self.salesperson_role,
        "company_name": self.company_name,
        "company_business": self.company_business,
        "company_values": self.company_values,
        "conversation_purpose": self.conversation_purpose,
        "conversation_type": self.conversation_type,
    }

    if self.use_tools:
        ai_message = await self.sales_agent_executor.ainvoke(inputs)
        output = ai_message["output"]
    else:
        ai_message = await self.sales_conversation_utterance_chain.ainvoke(inputs, return_intermediate_steps=True)
        output = ai_message["text"]

    agent_name = self.salesperson_name
    output = agent_name + ": " + output
    if "<END_OF_TURN>" not in output:
        output += " <END_OF_TURN>"
    self.conversation_history.append(output)

    if self.verbose:
        tool_status = "USE TOOLS INVOKE:" if self.use_tools else "WITHOUT TOOLS:"
        print(f"{tool_status}\n#\n#\n#\n#\n------------------")
        print(f"AI Message: {ai_message}")
        print(f"Output: {output.replace('<END_OF_TURN>', '')}")

    return ai_message

@time_logger
def _prep_messages(self):
    prompt = self.sales_conversation_utterance_chain.prep_prompts([
        dict(
            conversation_stage=self.current_conversation_stage,
            conversation_history="\n".join(self.conversation_history),
            salesperson_name=self.salesperson_name,
            salesperson_role=self.salesperson_role,
            company_name=self.company_name,
            company_business=self.company_business,
            company_values=self.company_values,
            conversation_purpose=self.conversation_purpose,
            conversation_type=self.conversation_type,
        )
    ])

    inception_messages = prompt[0][0].to_messages()
    message_dict = {"role": "system", "content": inception_messages[0].content}
    return [message_dict]

@time_logger
def _streaming_generator(self):
    """
    Generates a streaming generator for partial LLM output manipulation.

    This method is used when the sales agent needs to take an action before the full LLM output is available.
    For example, when performing text to speech on the partial LLM output. The method returns a streaming generator
    which can manipulate partial output from an LLM in-flight of the generation.

    Returns
    -------
    generator
        A streaming generator for manipulating partial LLM output.
    """
    messages = self._prep_messages()

    return self.sales_conversation_utterance_chain.llm.completion_with_retry(
        messages=messages,
        stop="<END_OF_TURN>",
        stream=True,
        model=self.model_name,
    )


@time_logger
async def acompletion_with_retry(self, llm: Any, **kwargs: Any) -> Any:
    """
    Use tenacity to retry the async completion call.

    This method uses the tenacity library to retry the asynchronous completion call in case of failure.
    It creates a retry decorator using the '_create_retry_decorator' method and applies it to the
    '_completion_with_retry' function which makes the actual asynchronous completion call.

    Parameters
    ----------
    llm : Any
        The language model to be used for the completion.
    **kwargs : Any
        Additional keyword arguments to be passed to the completion function.

    Returns
    -------
    Any
        The result of the completion function call.
    """
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await acompletion(**kwargs)

    return await _completion_with_retry(**kwargs)


@time_logger
async def _astreaming_generator(self):
    """
    Asynchronous generator to reduce I/O blocking when dealing with multiple
    clients simultaneously.

    This function returns a streaming generator which can manipulate partial output from an LLM
    in-flight of the generation. This is useful in scenarios where the sales agent wants to take an action
    before the full LLM output is available. For instance, if we want to do text to speech on the partial LLM output.

    Returns
    -------
    AsyncGenerator
        A streaming generator which can manipulate partial output from an LLM in-flight of the generation.
    """
    messages = self._prep_messages()

    return await self.acompletion_with_retry(
        llm=self.sales_conversation_utterance_chain.llm,
        messages=messages,
        stop="<END_OF_TURN>",
        stream=True,
        model=self.model_name,
    )

@time_logger
def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes one step of the sales agent.

    This function overrides the input temporarily with the current state of the conversation,
    generates the agent's utterance using either the sales agent executor or the sales conversation utterance chain,
    adds the agent's response to the conversation history, and returns the AI message.

    Parameters
    ----------
    inputs : Dict[str, Any]
        The initial inputs for the sales agent.

    Returns
    -------
    Dict[str, Any]
        The AI message generated by the sales agent.
    """
    # override inputs temporarily
    inputs = {
        "input": "",
        "conversation_stage": self.current_conversation_stage,
        "conversation_history": "\n".join(self.conversation_history),
        "salesperson_name": self.salesperson_name,
        "salesperson_role": self.salesperson_role,
        "company_name": self.company_name,
        "company_business": self.company_business,
        "company_values": self.company_values,
        "conversation_purpose": self.conversation_purpose,
        "conversation_type": self.conversation_type,
    }

    # Generate agent's utterance
    if self.use_tools:
        ai_message = self.sales_agent_executor.invoke(inputs)
        output = ai_message["output"]
    else:
        ai_message = self.sales_conversation_utterance_chain.invoke(
            inputs, return_intermediate_steps=True
        )
        output = ai_message["text"]

    # Add agent's response to conversation history
    agent_name = self.salesperson_name
    output = agent_name + ": " + output
    if "<END_OF_TURN>" not in output:
        output += " <END_OF_TURN>"
    self.conversation_history.append(output)

    if self.verbose:
        tool_status = "USE TOOLS INVOKE:" if self.use_tools else "WITHOUT TOOLS:"
        print(f"{tool_status}\n#\n#\n#\n#\n------------------")
        print(f"AI Message: {ai_message}")
        print()
        print(f"Output: {output.replace('<END_OF_TURN>', '')}")

    return ai_message


@classmethod
@time_logger
def from_llm(cls, llm: ChatLiteLLM, verbose: bool = False, **kwargs) -> "SalesGPT":
    """
    Class method to initialize the SalesGPT Controller from a given ChatLiteLLM instance.

    This method sets up the stage analyzer chain and sales conversation utterance chain. It also checks if custom prompts
    are to be used and if tools are to be set up for the agent. If tools are to be used, it sets up the knowledge base,
    gets the tools, sets up the prompt, and initializes the agent with the tools. If tools are not to be used, it sets
    the sales agent executor and knowledge base to None.

    Parameters
    ----------
    llm : ChatLiteLLM
        The ChatLiteLLM instance to initialize the SalesGPT Controller from.
    verbose : bool, optional
        If True, verbose output is enabled. Default is False.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    SalesGPT
        The initialized SalesGPT Controller.
    """
    stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)
    sales_conversation_utterance_chain = SalesConversationChain.from_llm(
        llm, verbose=verbose
    )

    # Handle custom prompts
    use_custom_prompt = kwargs.pop("use_custom_prompt", False)
    custom_prompt = kwargs.pop("custom_prompt", None)

    sales_conversation_utterance_chain = SalesConversationChain.from_llm(
        llm,
        verbose=verbose,
        use_custom_prompt=use_custom_prompt,
        custom_prompt=custom_prompt,
    )

    # Handle tools
    use_tools_value = kwargs.pop("use_tools", False)
    if isinstance(use_tools_value, str):
        if use_tools_value.lower() not in ["true", "false"]:
            raise ValueError("use_tools must be 'True', 'False', True, or False")
        use_tools = use_tools_value.lower() == "true"
    elif isinstance(use_tools_value, bool):
        use_tools = use_tools_value
    else:
        raise ValueError(
            "use_tools must be a boolean or a string ('True' or 'False')"
        )
    sales_agent_executor = None
    knowledge_base = None

    if use_tools:
        product_catalog = kwargs.pop("product_catalog", None)
        tools = get_tools(product_catalog)

        prompt = CustomPromptTemplateForTools(
            template=SALES_AGENT_TOOLS_PROMPT,
            tools_getter=lambda x: tools,
            input_variables=[
                "input",
                "intermediate_steps",
                "salesperson_name",
                "salesperson_role",
                "company_name",
                "company_business",
                "company_values",
                "conversation_purpose",
                "conversation_type",
                "conversation_history",
            ],
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
        tool_names = [tool.name for tool in tools]
        output_parser = SalesConvoOutputParser(
            ai_prefix=kwargs.get("salesperson_name", ""), verbose=verbose
        )
        sales_agent_with_tools = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )

        sales_agent_executor = CustomAgentExecutor.from_agent_and_tools(
            agent=sales_agent_with_tools,
            tools=tools,
            verbose=verbose,
            return_intermediate_steps=True,
        )

    return cls(
        stage_analyzer_chain=stage_analyzer_chain,
        sales_conversation_utterance_chain=sales_conversation_utterance_chain,
        sales_agent_executor=sales_agent_executor,
        knowledge_base=knowledge_base,
        model_name=llm.model,
        verbose=verbose,
        use_tools=use_tools,
        **kwargs,
    )

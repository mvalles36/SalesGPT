import os
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from dotenv import load_dotenv
from salesgpt.salesgptapi import SalesGPTAPI

# Load the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

@pytest.fixture
def mock_salesgpt_step():
    """
    Fixture to mock the SalesGPT.step method for synchronous tests.
    """
    with patch("salesgpt.salesgptapi.SalesGPT.step") as mock_step:
        mock_step.return_value = "Mock response"
        yield mock_step

@pytest.fixture
def mock_salesgpt_astep():
    """
    Fixture to mock the SalesGPT.astep method for asynchronous tests.
    """
    with patch("salesgpt.salesgptapi.SalesGPT.astep", new_callable=AsyncMock) as mock_step:
        mock_step.return_value = AsyncMock(return_value={
            "response": "Mock response",
            "intermediate_steps": []  # Ensures the intermediate steps key is present
        })
        yield mock_step

class TestSalesGPTAPI:
    def test_initialize_agent_with_tools(self):
        """
        Test to ensure SalesGPTAPI initializes with tools enabled.
        """
        api = SalesGPTAPI(config_path="", use_tools=True)
        assert api.sales_agent.use_tools == True, \
            "SalesGPTAPI should initialize SalesGPT with tools enabled."

    def test_initialize_agent_without_tools(self):
        """
        Test to ensure SalesGPTAPI initializes with tools disabled.
        """
        api = SalesGPTAPI(config_path="", use_tools=False)
        assert api.sales_agent.use_tools == False, \
            "SalesGPTAPI should initialize SalesGPT with tools disabled."

    @pytest.mark.asyncio
    async def test_do_method_with_human_input(self, mock_salesgpt_astep):
        """
        Test the 'do' method with human input, ensuring the input is added to the conversation history
        and the payload response matches the mock response.
        """
        api = SalesGPTAPI(config_path="", use_tools=False)
        payload = await api.do(human_input="Hello")
        assert "User: Hello <END_OF_TURN>" in api.sales_agent.conversation_history, \
            "Human input should be added to the conversation history."
        assert payload["response"] == "Mock response", \
            f"The payload response should match the mock response. {payload}"

    @pytest.mark.asyncio
    async def test_do_method_with_human_input_anthropic(self, mock_salesgpt_astep):
        """
        Test the 'do' method with human input using the Anthropic model, ensuring correct conversation
        history handling and response.
        """
        api = SalesGPTAPI(config_path="", use_tools=False, model_name="anthropic.claude-3-sonnet-20240229-v1:0")
        payload = await api.do(human_input="Hello")
        assert "User: Hello <END_OF_TURN>" in api.sales_agent.conversation_history, \
            "Human input should be added to the conversation history."
        assert payload["response"] == "Mock response", \
            f"The payload response should match the mock response. {payload}"

    @pytest.mark.asyncio
    async def test_do_method_without_human_input(self, mock_salesgpt_astep):
        """
        Test the 'do' method without human input, ensuring the response matches the mock response.
        """
        api = SalesGPTAPI(config_path="", use_tools=False)
        payload = await api.do()
        assert payload["response"] == "Mock response", \
            "The payload response should match the mock response when no human input is provided."

    @pytest.mark.asyncio
    async def test_payload_structure(self, mock_salesgpt_astep):
        """
        Test to ensure the payload structure contains all the expected keys.
        """
        api = SalesGPTAPI(config_path="", use_tools=False)
        payload = await api.do(human_input="Test input")
        expected_keys = [
            "bot_name",
            "response",
            "conversational_stage",
            "tool",
            "tool_input",
            "action_output",
            "action_input",
        ]
        for key in expected_keys:
            assert key in payload, f"Payload missing expected key: {key}"

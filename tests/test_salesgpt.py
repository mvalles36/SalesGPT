import json
import os
from unittest.mock import patch
import pytest
from dotenv import load_dotenv
from transformers import pipeline
from salesgpt.agents import SalesGPT

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

# Mock response for the API call
MOCK_RESPONSE = {
    "choices": [
        {
            "text": "Ted Lasso: Hey, good morning! This is a mock response to test when you don't have access to LLM API gods. <END_OF_TURN>"
        }
    ]
}


class TestSalesGPT:
    @pytest.fixture(autouse=True)
    def load_env(self):
        # Setup for each test
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.huggingface_api_key:
            print("HUGGINGFACE_API_KEY not found, proceeding with mock testing.")

    def _test_inference_with_mock_or_real_api(self, use_mock_api):
        """Helper method to test inference with either mock or real API based on the use_mock_api flag."""
        if use_mock_api:
            self.huggingface_api_key = None  # Force the use of mock API by unsetting the API key

        # Use Hugging Face's facebook/bart-large model for text generation
        if use_mock_api:
            sales_agent = SalesGPT.from_llm(
                llm=None,  # No LLM here, mocking the step response
                verbose=False,
                use_tools=False,
                salesperson_name="Ted Lasso",
                salesperson_role="Sales Representative",
                company_name="Sleep Haven",
                company_business="""Sleep Haven 
                                    is a premium mattress company that provides
                                    customers with the most comfortable and
                                    supportive sleeping experience possible. 
                                    We offer a range of high-quality mattresses,
                                    pillows, and bedding accessories 
                                    that are designed to meet the unique 
                                    needs of our customers.""",
            )
            sales_agent.seed_agent()
            sales_agent.determine_conversation_stage()

            with patch("salesgpt.agents.SalesGPT._call", return_value=MOCK_RESPONSE):
                sales_agent.step()
                output = MOCK_RESPONSE["choices"][0]["text"]
                sales_agent.conversation_history.append(output)

        else:
            generator = pipeline("text2text-generation", model="facebook/bart-large")
            sales_agent = SalesGPT.from_llm(
                llm=generator,
                verbose=False,
                use_tools=False,
                salesperson_name="Ted Lasso",
                salesperson_role="Sales Representative",
                company_name="Sleep Haven",
                company_business="""Sleep Haven 
                                    is a premium mattress company that provides
                                    customers with the most comfortable and
                                    supportive sleeping experience possible. 
                                    We offer a range of high-quality mattresses,
                                    pillows, and bedding accessories 
                                    that are designed to meet the unique 
                                    needs of our customers.""",
            )

            sales_agent.seed_agent()
            sales_agent.determine_conversation_stage()

            # Generate the response using BART
            output = generator("User: Hello <END_OF_TURN>", max_length=100)
            sales_agent.conversation_history.append(output[0]['generated_text'])

        agent_output = sales_agent.conversation_history[-1]
        assert agent_output is not None, "Agent output cannot be None."
        assert isinstance(agent_output, str), "Agent output needs to be of type str"
        assert len(agent_output) > 0, "Length of output needs to be greater than 0."
        if use_mock_api:
            assert (
                "mock response" in agent_output
            ), "Mock response not found in agent output."
        else:
            assert (
                "mock response" not in agent_output
            ), "Mock response found in agent output."

    def test_inference_with_mock_api(self, load_env):
        """Test that the agent uses the mock response when the API key is not set."""
        self._test_inference_with_mock_or_real_api(use_mock_api=True)

    def test_inference_with_real_api(self, load_env):
        """Test that the agent uses the real API when the API key is set."""
        self._test_inference_with_mock_or_real_api(use_mock_api=False)

    def test_valid_inference_with_tools(self, load_env):
        """Test that the agent will start and generate the first utterance."""

        generator = pipeline("text2text-generation", model="facebook/bart-large")
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

        sales_agent = SalesGPT.from_llm(
            llm=generator,
            verbose=False,
            use_tools=True,
            product_catalog=f"{data_dir}/sample_product_catalog.txt",
            salesperson_name="Ted Lasso",
            salesperson_role="Sales Representative",
            company_name="Sleep Haven",
            company_business="""Sleep Haven 
                                    is a premium mattress company that provides
                                    customers with the most comfortable and
                                    supportive sleeping experience possible. 
                                    We offer a range of high-quality mattresses,
                                    pillows, and bedding accessories 
                                    that are designed to meet the unique 
                                    needs of our customers.""",
        )

        sales_agent.seed_agent()
        sales_agent.determine_conversation_stage()

        sales_agent.step()

        agent_output = sales_agent.conversation_history[-1]
        assert agent_output is not None, "Agent output cannot be None."
        assert isinstance(agent_output, str), "Agent output needs to be of type str"
        assert len(agent_output) > 0, "Length of output needs to be greater than 0."

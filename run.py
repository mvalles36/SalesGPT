import argparse
import json
import logging
import os
import sys
import warnings
import requests

from dotenv import load_dotenv
from salesgpt.agents import SalesGPT

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Validate required environment variables
if not os.getenv("HUGGINGFACE_API_KEY"):
    logging.error("HUGGINGFACE_API_KEY is not set in the environment.")
    sys.exit(1)

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

def query_hugging_face(payload):
    """Query the Hugging Face model and return the response."""
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Hugging Face API: {e}")
        return None

def main():
    # Initialize argparse
    parser = argparse.ArgumentParser(description="SalesGPT - Your Context-Aware AI Sales Assistant")

    # Add arguments
    parser.add_argument("--config", type=str, help="Path to agent config file", default="")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity", default=False)
    parser.add_argument("--max_num_turns", type=int, help="Maximum number of turns in the sales conversation", default=10)
    
    # Parse arguments
    args = parser.parse_args()

    # Configure sales agent
    if args.config == "":
        logging.info("No agent config specified, using standard config")
        sales_agent_kwargs = {
            "verbose": args.verbose,
            "use_tools": True,
            "product_catalog": "examples/sample_product_catalog.txt",
            "salesperson_name": "Ted Lasso",
        }
        sales_agent = SalesGPT.from_llm(llm=None, **sales_agent_kwargs)  # Pass None or your LLM object
    else:
        try:
            with open(args.config, "r", encoding="UTF-8") as f:
                config = json.load(f)
                logging.info(f"Agent config loaded from {args.config}")
                sales_agent = SalesGPT.from_llm(llm=None, verbose=args.verbose, **config)  # Pass None or your LLM object
        except FileNotFoundError:
            logging.error(f"Config file {args.config} not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from the config file {args.config}.")
            sys.exit(1)

    sales_agent.seed_agent()
    logging.info("Sales agent initialized. Beginning conversation...")
    print("=" * 10)

    try:
        for cnt in range(args.max_num_turns):
            # Get Hugging Face output
            output = query_hugging_face({
                "inputs": "Please provide a sales pitch."
            })
            if output:
                sales_response = output.get("generated_text", "No response from model.")
                print(f"Sales Agent: {sales_response}")

            human_input = input("Your response: ")
            sales_agent.human_step(human_input)
            print("=" * 10)

    except KeyboardInterrupt:
        logging.info("Conversation interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()

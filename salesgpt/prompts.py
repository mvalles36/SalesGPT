import asyncio
import json
import re
import requests

HUGGINGFACE_API_KEY = "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"
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

class SalesGPTAPI:
    def __init__(
        self,
        config_path: str,
        verbose: bool = True,
        max_num_turns: int = 20,
        product_catalog: str = "examples/sample_product_catalog.txt",
        use_tools=True,
    ):
        self.config_path = config_path
        self.verbose = verbose
        self.max_num_turns = max_num_turns
        self.product_catalog = product_catalog
        self.conversation_history = []
        self.use_tools = use_tools
        self.sales_agent = self.initialize_agent()
        self.current_turn = 0

    def initialize_agent(self):
        config = {"verbose": self.verbose}
        if self.config_path:
            with open(self.config_path, "r") as f:
                config.update(json.load(f))
            if self.verbose:
                print(f"Loaded agent config: {config}")
        else:
            print("Default agent config in use")

        if self.use_tools:
            print("USING TOOLS")
            config.update(
                {
                    "use_tools": True,
                    "product_catalog": self.product_catalog,
                    "salesperson_name": "Ted Lasso"
                    if not self.config_path
                    else config.get("salesperson_name", "Ted Lasso"),
                }
            )

        sales_agent = SalesGPT.from_llm(self.llm, **config)

        print(f"SalesGPT use_tools: {sales_agent.use_tools}")
        sales_agent.seed_agent()
        return sales_agent

    def query(self, payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    async def do(self, human_input=None):
        self.current_turn += 1
        current_turns = self.current_turn
        if current_turns >= self.max_num_turns:
            print("Maximum number of turns reached - ending the conversation.")
            return [
                "BOT",
                "In case you'll have any questions - just text me one more time!",
            ]

        if human_input is not None:
            self.sales_agent.human_step(human_input)

        # Construct the prompt using the conversation history
        prompt = self.create_prompt()

        ai_log = self.query({"inputs": prompt})
        await self.sales_agent.adetermine_conversation_stage()
        
        if self.verbose:
            print("=" * 10)
            print(f"AI LOG {ai_log}")

        if (
            self.sales_agent.conversation_history
            and "<END_OF_CALL>" in self.sales_agent.conversation_history[-1]
        ):
            print("Sales Agent determined it is time to end the conversation.")
            self.sales_agent.conversation_history[-1] = self.sales_agent.conversation_history[-1].replace("<END_OF_CALL>", "")

        reply = (
            self.sales_agent.conversation_history[-1]
            if self.sales_agent.conversation_history
            else ""
        )

        payload = {
            "bot_name": reply.split(": ")[0],
            "response": ": ".join(reply.split(": ")[1:]).rstrip("<END_OF_TURN>"),
            "conversational_stage": self.sales_agent.current_conversation_stage,
            "model_name": "facebook/bart-large",
        }
        return payload

    def create_prompt(self):
        return SALES_AGENT_TOOLS_PROMPT.format(
            salesperson_name=self.sales_agent.salesperson_name,
            salesperson_role=self.sales_agent.salesperson_role,
            company_name=self.sales_agent.company_name,
            company_business=self.sales_agent.company_business,
            company_values=self.sales_agent.company_values,
            conversation_purpose=self.sales_agent.conversation_purpose,
            conversation_type=self.sales_agent.conversation_type,
            conversation_history='\n'.join(self.sales_agent.conversation_history),
            agent_scratchpad=self.sales_agent.agent_scratchpad
        )

    async def do_stream(self, conversation_history: [str], human_input=None):
        current_turns = len(conversation_history) + 1
        if current_turns >= self.max_num_turns:
            print("Maximum number of turns reached - ending the conversation.")
            yield [
                "BOT",
                "In case you'll have any questions - just text me one more time!",
            ]
            raise StopAsyncIteration

        self.sales_agent.seed_agent()
        self.sales_agent.conversation_history = conversation_history

        if human_input is not None:
            self.sales_agent.human_step(human_input)

        stream_gen = self.sales_agent.astep(stream=True)
        for model_response in stream_gen:
            for choice in model_response.choices:
                message = choice["delta"]["content"]
                if message is not None:
                    if "<END_OF_CALL>" in message:
                        print("Sales Agent determined it is time to end the conversation.")
                        yield [
                            "BOT",
                            "In case you'll have any questions - just text me one more time!",
                        ]
                    yield message
                else:
                    continue


SALES_AGENT_TOOLS_PROMPT = """
Never forget your name is {salesperson_name}. You work as a {salesperson_role}.
You work at company named {company_name}. {company_name}'s business is the following: {company_business}.
Company values are the following. {company_values}
You are contacting a potential prospect in order to {conversation_purpose}
Your means of contacting the prospect is {conversation_type}

If you're asked about where you got the user's contact information, say that you got it from public records.
Keep your responses in short length to retain the user's attention. Never produce lists, just answers.
Start the conversation by just a greeting and how is the prospect doing without pitching in your first turn.
When the conversation is over, output <END_OF_CALL>
Always think about at which conversation stage you are at before answering:

1: Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Always clarify in your greeting the reason why you are calling.
2: Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.
3: Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.
4: Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.
5: Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.
6: Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.
7: Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.
8: End conversation: The prospect has to leave to call, the prospect is not interested, or next steps where already determined by the sales agent.

You must respond according to the previous conversation history and the stage of the conversation you are at.
Only generate one response at a time and act as {salesperson_name} only!

Begin!

Previous conversation history:
{conversation_history}

Thought:
{agent_scratchpad}
"""

# The other prompt definitions (SALES_AGENT_INCEPTION_PROMPT, STAGE_ANALYZER_INCEPTION_PROMPT) can be integrated similarly if needed.

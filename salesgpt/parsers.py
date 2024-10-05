import re
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish


class SalesConvoOutputParser(AgentOutputParser):
    ai_prefix: str = "AI"  # Change to reflect the salesperson's name if needed
    verbose: bool = False

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if self.verbose:
            print("TEXT")
            print(text)
            print("-------")
        
        # Regex pattern to match the action and action input
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text)

        if not match:
            # If no match is found, return the output as a finish action
            return AgentFinish(
                {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
            )

        action = match.group(1)  # Extract the action
        action_input = match.group(2)  # Extract the action input
        
        # Return the AgentAction object with the extracted values
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)

    @property
    def _type(self) -> str:
        return "sales-agent"

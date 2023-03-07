from llama_index import GPTSimpleVectorIndex
from langchain import OpenAI, LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import initialize_agent, Tool, ZeroShotAgent, AgentExecutor
from typing import Any, List, Optional, Tuple, Union
from berri_ai.QAAgent import QAAgent
from berri_ai.ComplexInformationQA import ComplexInformationQA

class ComplexQA(ComplexInformationQA):
    def __init__(self, model_key, index = None, prompt = None, functions = [], descriptions = [], max_tokens=1024):
        super().__init__(model_key, index, prompt, functions, descriptions)
        self.max_tokens = max_tokens

    def run(self, query_string: str, max_tokens: int = 256):
        llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=self.prompt)
        agent2 = QAAgent(llm_chain=self.llm_chain, tools=self.tools)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent2, tools=self.tools, verbose=True, return_intermediate_steps=True)

        # Set the max tokens to generate longer response
        self.llm_chain.llm.max_tokens = self.max_tokens

        answer = agent_executor({"input":query_string}, max_tokens)

        # Reset the max tokens back to default
        self.llm_chain.llm.max_tokens = 2048

        # Return the desired length of response
        return answer

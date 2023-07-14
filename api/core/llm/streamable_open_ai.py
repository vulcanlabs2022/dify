import os

from langchain.callbacks.manager import Callbacks
from langchain.schema import LLMResult, BaseMessage
from typing import Optional, List, Dict, Any, Mapping
from langchain import OpenAI
from pydantic import root_validator

from core.llm.error_handle_wraps import handle_llm_exceptions, handle_llm_exceptions_async
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")


class StreamableOpenAI(OpenAI):
    bs_api_base = ''

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            import openai

            values["client"] = openai.Completion
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        if values["streaming"] and values["n"] > 1:
            raise ValueError("Cannot stream results when n > 1.")
        if values["streaming"] and values["best_of"] > 1:
            raise ValueError("Cannot stream results when best_of > 1.")
        return values

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**super()._invocation_params, **{
            "api_type": 'openai',
            "api_base": self.bs_api_base,
            "api_version": None,
            "api_key": self.openai_api_key,
            "organization": self.openai_organization if self.openai_organization else None,
        }}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {**super()._identifying_params, **{
            "api_type": 'openai',
            "api_base": self.bs_api_base,
            "api_version": None,
            "api_key": self.openai_api_key,
            "organization": self.openai_organization if self.openai_organization else None,
        }}

    @handle_llm_exceptions
    def generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> LLMResult:
        return super().generate(prompts, stop, callbacks, **kwargs)

    @handle_llm_exceptions_async
    async def agenerate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> LLMResult:
        return await super().agenerate(prompts, stop, callbacks, **kwargs)

    def get_num_tokens(self, text: str) -> int:
        tokens = encoding.encode(text)
        return len(tokens)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        total = 0
        for message in messages:
            total += self.get_num_tokens(message.content)
        return total

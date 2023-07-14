import os

from langchain.callbacks.manager import Callbacks
from langchain.schema import BaseMessage, LLMResult
from langchain.chat_models import ChatOpenAI
from typing import Optional, List, Dict, Any

from pydantic import root_validator

from core.llm.error_handle_wraps import handle_llm_exceptions, handle_llm_exceptions_async
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")


class StreamableChatOpenAI(ChatOpenAI):
    bs_api_base = ''

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            import openai
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        try:
            values["client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            **super()._default_params,
            "api_type": 'openai',
            "api_base": self.bs_api_base,
            "api_version": None,
            "api_key": self.openai_api_key,
            "organization": self.openai_organization if self.openai_organization else None,
        }

    def get_messages_tokens(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in a list of messages.

        Args:
            messages: The messages to count the tokens of.

        Returns:
            The number of tokens in the messages.
        """
        tokens_per_message = 5
        tokens_per_request = 3

        message_tokens = tokens_per_request
        message_strs = ''
        for message in messages:
            message_strs += message.content
            message_tokens += tokens_per_message

        # calc once
        message_tokens += self.get_num_tokens(message_strs)

        return message_tokens

    @handle_llm_exceptions
    def generate(
            self,
            messages: List[List[BaseMessage]],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> LLMResult:
        return super().generate(messages, stop, callbacks, **kwargs)

    @handle_llm_exceptions_async
    async def agenerate(
            self,
            messages: List[List[BaseMessage]],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> LLMResult:
        return await super().agenerate(messages, stop, callbacks, **kwargs)

    def get_num_tokens(self, text: str) -> int:
        tokens = encoding.encode(text)
        return len(tokens)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        total = 0
        for message in messages:
            total += self.get_num_tokens(message.content)
        return total

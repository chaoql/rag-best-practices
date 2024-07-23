from typing import Optional, List, Mapping, Any, Sequence, Dict
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    ChatMessage,
    ChatResponse,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from llama_index.core import SimpleDirectoryReader, SummaryIndex
from zhipuai import ZhipuAI

DEFAULT_MODEL = 'glm-4'


def to_message_dicts(messages: Sequence[ChatMessage]) -> List:
    return [
        {"role": message.role.value, "content": message.content, }
        for message in messages if all([value is not None for value in message.values()])
    ]


def get_additional_kwargs(response) -> Dict:
    return {
        "token_counts": response.usage.total_tokens,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }


class ChatGLM(CustomLLM):
    num_output: int = DEFAULT_NUM_OUTPUTS
    context_window: int = Field(default=DEFAULT_CONTEXT_WINDOW,
                                description="The maximum number of context tokens for the model.", gt=0, )
    model: str = Field(default=DEFAULT_MODEL, description="The ChatGlM model to use. glm-4 or glm-3-turbo")
    api_key: str = Field(default=None, description="The ChatGLM API key.")
    reuse_client: bool = Field(default=True, description=(
        "Reuse the client between requests. When doing anything with large "
        "volumes of async API calls, setting this to false can improve stability."), )
    _client: Optional[Any] = PrivateAttr()

    def __init__(
            self,
            model: str = DEFAULT_MODEL,
            reuse_client: bool = True,
            api_key: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            reuse_client=reuse_client,
            **kwargs,
        )
        self._client = None

    def _get_client(self) -> ZhipuAI:
        if not self.reuse_client:
            return ZhipuAI(api_key=self.api_key)

        if self._client is None:
            self._client = ZhipuAI(api_key=self.api_key)
        return self._client

    @classmethod
    def class_name(cls) -> str:
        return "chatglm_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model,
        )

    def _chat(self, messages: List, stream=False) -> Any:
        response = self._get_client().chat.completions.create(
            model=self.model,  # 填写需要调用的模型名称
            messages=messages,
        )
        # print(f"_chat, response: {response}")
        return response

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        global rsp
        # print(f"complete: messages {messages} ")
        try:
            response = self._chat(messages, stream=False)

            rsp = CompletionResponse(text=str(response.choices[0].message.content),
                                     raw=response,
                                     additional_kwargs=get_additional_kwargs(response), )
            # print(f"complete: {rsp} ")
        except Exception as e:
            print(f"complete: exception {e}")

        return rsp

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response_txt = ""
        messages = [{"role": "user", "content": prompt}]
        response = self._chat(messages, stream=True)
        # print(f"stream_complete: {response} ")
        for chunk in response:
            # chunk.choices[0].delta # content='```' role='assistant' tool_calls=None
            token = chunk.choices[0].delta.content
            response_txt += token
            yield CompletionResponse(text=response_txt, delta=token)


# Settings.llm = ChatGLM(api_key="de0391a95e7549617d95f1e6ea82c8b9.Mj7e05IXWgj6Oclg")
#
# # define embed model
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
#
# # Load the your data
# documents = SimpleDirectoryReader("./test").load_data()
# index = SummaryIndex.from_documents(documents)
#
# # Query and print response
# query_engine = index.as_query_engine()
# response = query_engine.query("小明多大了")
# print(response)
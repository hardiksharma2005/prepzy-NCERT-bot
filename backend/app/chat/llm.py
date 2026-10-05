"""All LLM calls go through this class, so tests can swap it for a fake and count calls."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .. import config
from . import prompts


class AnswerOut(BaseModel):
    answer: str = Field(description="The answer to the student, in markdown")
    chapters: list[str] = Field(default_factory=list, description="Chapter names used")
    in_scope: bool = Field(description="False if the excerpts do not answer the question")
    topic: str = Field(default="", description="1-4 word main concept")


class LLMClient:
    def __init__(self):
        # Without a key the server still starts and serves cache hits; LLM calls fail clearly.
        common = dict(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY or "missing",
                      timeout=45, max_retries=2)
        # Free-tier models are sometimes "over capacity" (503); then the next one answers.
        self._answer_models = [ChatOpenAI(model=name, temperature=0.2, **common)
                               for name in dict.fromkeys([config.ANSWER_MODEL,
                                                          config.FALLBACK_MODEL])]
        self._small = ChatOpenAI(model=config.CONDENSE_MODEL, temperature=0, **common)
        self.calls = 0

    def _check(self) -> None:
        if not config.LLM_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set on the server")

    def condense(self, history: str, message: str) -> str:
        self._check()
        self.calls += 1
        out = self._small.invoke([
            SystemMessage(prompts.CONDENSE_SYSTEM),
            HumanMessage(prompts.CONDENSE_USER.format(history=history, message=message))])
        return out.content.strip().strip('"').splitlines()[0].strip() or message

    def answer(self, question: str, context: str) -> AnswerOut:
        self._check()
        self.calls += 1
        messages = [SystemMessage(prompts.ANSWER_SYSTEM),
                    HumanMessage(prompts.ANSWER_USER.format(context=context, question=question))]
        error = None
        for model in self._answer_models:
            try:
                out = model.with_structured_output(AnswerOut, method="function_calling") \
                    .invoke(messages)
                if isinstance(out, AnswerOut):
                    return out
            except Exception as exc:  # malformed tool call, or the model is unavailable
                error = exc
            try:  # plain text from the same model before moving on
                text = model.invoke(messages).content.strip()
                return AnswerOut(answer=text, chapters=[], in_scope=bool(text), topic="")
            except Exception as exc:
                error = exc
        raise error

    def transform(self, kind: str, question: str, answer: str, message: str,
                  context: str) -> str:
        self._check()
        self.calls += 1
        system = prompts.TRANSFORM_SYSTEM.format(
            instruction=prompts.TRANSFORM_INSTRUCTIONS[kind])
        messages = [SystemMessage(system), HumanMessage(prompts.TRANSFORM_USER.format(
            context=context, question=question, answer=answer, message=message))]
        error = None
        for model in self._answer_models:
            try:
                return model.invoke(messages).content.strip()
            except Exception as exc:
                error = exc
        raise error

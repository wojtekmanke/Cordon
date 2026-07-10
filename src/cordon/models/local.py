"""Inspect model providers for open models running on this machine.

`mlx/<repo>` runs a model with MLX on Apple Silicon and is the fast path for
agent runs. `torch/<repo>` runs it with PyTorch, which the probing track needs
for access to activations. Both render the conversation with the model's own
chat template and parse tool calls from the text it produces.

Tool output is passed to the chat template as it is. If it contains the
template's control tokens they are tokenised as control tokens: that is the
behaviour of a serving stack which does not escape them, and it is the
weakness the `chatml_spoof` attack and the `sanitise` defence are about.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from .chat import hf_messages, hf_tools, parse_reply

# One thread owns the GPU. MLX ties its streams to the thread that created
# them, and two models generating at once would only take turns anyway.
_GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cordon-gpu")
_LOADED: dict[tuple[str, str], Any] = {}

DEFAULT_MAX_TOKENS = 512


class MLXBackend:
    def __init__(self, name: str) -> None:
        from mlx_lm import load

        self.model, self.tokenizer = load(name)

    def render(self, messages: list[dict], tools: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tools=tools or None, add_generation_prompt=True, tokenize=False
        )

    def complete(self, prompt: str, max_tokens: int, temperature: float, seed: int | None):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        if seed is not None:
            mx.random.seed(seed)
        text, last = "", None
        for last in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        ):
            text += last.text
        if last is None:
            return "", 0, 0
        return text, last.prompt_tokens, last.generation_tokens


class TorchBackend:
    def __init__(self, name: str, dtype: str = "float16", device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype))
        self.model.to(device).eval()

    def render(self, messages: list[dict], tools: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tools=tools or None, add_generation_prompt=True, tokenize=False
        )

    def encode(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

    def complete(self, prompt: str, max_tokens: int, temperature: float, seed: int | None):
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.encode(prompt)
        n_in = inputs["input_ids"].shape[1]
        # Greedy unless a temperature is asked for. The repetition penalty some
        # models ship in their generation config is turned off so that both
        # backends decode the same way.
        if temperature > 0:
            sampling = {"do_sample": True, "temperature": temperature}
        else:
            sampling = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                **sampling,
            )
        new = out[0, n_in:]
        return self.tokenizer.decode(new, skip_special_tokens=True), n_in, int(new.shape[0])


def load_backend(kind: str, name: str, **args: Any):
    """Load a model once per process. Call from the GPU thread."""
    key = (kind, name)
    if key not in _LOADED:
        _LOADED[key] = MLXBackend(name) if kind == "mlx" else TorchBackend(name, **args)
    return _LOADED[key]


async def on_gpu(fn, *args):
    """Run `fn` on the thread that owns the GPU."""
    return await asyncio.get_running_loop().run_in_executor(_GPU, fn, *args)


class LocalAPI(ModelAPI):
    kind = ""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),  # noqa: B008
        **model_args: Any,
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)
        self.model_args = model_args

    def _generate(self, messages, tools, max_tokens, temperature, seed):
        backend = load_backend(self.kind, self.model_name, **self.model_args)
        prompt = backend.render(messages, tools)
        return backend.complete(prompt, max_tokens, temperature, seed)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        offered = [] if tool_choice == "none" else hf_tools(tools)
        max_tokens = config.max_tokens or DEFAULT_MAX_TOKENS
        text, n_in, n_out = await on_gpu(
            self._generate,
            hf_messages(input),
            offered,
            max_tokens,
            config.temperature or 0.0,
            config.seed,
        )
        prose, calls = parse_reply(text) if offered else (text.strip(), [])
        if calls:
            stop_reason = "tool_calls"
        else:
            stop_reason = "max_tokens" if n_out >= max_tokens else "stop"
        message = ChatMessageAssistant(
            content=prose, tool_calls=calls or None, model=self.model_name, source="generate"
        )
        return ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice(message=message, stop_reason=stop_reason)],
            usage=ModelUsage(input_tokens=n_in, output_tokens=n_out, total_tokens=n_in + n_out),
        )

    def max_connections(self) -> int:
        return 1


class MLXAPI(LocalAPI):
    kind = "mlx"


class TorchAPI(LocalAPI):
    kind = "torch"

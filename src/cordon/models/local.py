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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Steer:
    """An edit to the residual stream.

    `ablate` removes the component along `vector` after every layer. `add`
    adds `alpha` times `vector` after one layer.
    """

    vector: Any  # a numpy array of the model's width
    mode: str = "ablate"
    layer: int = 0
    alpha: float = 1.0

    @classmethod
    def load(cls, path: str, mode: str = "ablate", layer: int = 0, alpha: float = 1.0) -> Steer:
        import numpy as np

        return cls(np.load(path), mode, layer, alpha)


class Hub:
    """Where the taps on a model's layers report to.

    `record`, when a list, collects the residual stream at the last position
    after every layer. `edit`, when set, may change the residual stream after
    any layer. With neither set the taps do nothing.
    """

    def __init__(self) -> None:
        self.record: list | None = None
        self.edit = None

    def after(self, index: int, out):
        if self.edit is not None:
            out = self.edit(index, out)
        if self.record is not None:
            self.record.append(out[0, -1])
        return out


class MLXBackend:
    def __init__(self, name: str) -> None:
        from mlx_lm import load

        self.model, self.tokenizer = load(name)
        self.hub: Hub | None = None
        self._prefixes: dict[str, tuple[int, Any]] = {}

    def render(self, messages: list[dict], tools: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tools=tools or None, add_generation_prompt=True, tokenize=False
        )

    def _split(self, prompt: str) -> tuple[str, str]:
        """Split a prompt after its first turn, which holds the system prompt and tools.

        That turn is the same for every call an agent makes in a run, so its
        keys and values are computed once and reused.
        """
        cut = prompt.find("<|im_start|>", 1)
        return (prompt[:cut], prompt[cut:]) if cut > 0 else ("", prompt)

    def _prefix_cache(self, prefix: str):
        import copy

        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        if prefix not in self._prefixes:
            if len(self._prefixes) >= 8:
                self._prefixes.pop(next(iter(self._prefixes)))
            tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            cache = make_prompt_cache(self.model)
            self.model(mx.array(tokens)[None], cache=cache)
            mx.eval([c.state for c in cache])
            self._prefixes[prefix] = (len(tokens), cache)
        length, cache = self._prefixes[prefix]
        return length, copy.deepcopy(cache)

    def complete(self, prompt: str, max_tokens: int, temperature: float, seed: int | None):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        if seed is not None:
            mx.random.seed(seed)
        prefix, rest = self._split(prompt)
        # Edits to the residual stream change the keys and values, so a cached
        # prefix computed without them cannot be reused.
        steering = self.hub is not None and self.hub.edit is not None
        if prefix and not steering:
            cached, cache = self._prefix_cache(prefix)
            tokens = self.tokenizer.encode(rest, add_special_tokens=False)
            extra = {"prompt_cache": cache}
        else:
            cached, tokens, extra = 0, self.tokenizer.encode(prompt, add_special_tokens=False), {}
        text, last = "", None
        for last in stream_generate(
            self.model,
            self.tokenizer,
            tokens,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            **extra,
        ):
            text += last.text
        if last is None:
            return "", 0, 0
        return text, cached + len(tokens), last.generation_tokens

    # --- access to activations, used by the probing track ---

    def instrument(self) -> Hub:
        """Wrap every layer in a tap. Done on first use, so plain runs are untouched."""
        if self.hub is None:
            import mlx.nn as nn

            hub = Hub()

            class Tap(nn.Module):
                def __init__(self, inner, index: int) -> None:
                    super().__init__()
                    self.inner = inner
                    self.index = index

                def __call__(self, x, mask=None, cache=None):
                    return hub.after(self.index, self.inner(x, mask, cache))

            layers = self.model.model.layers
            for index, layer in enumerate(layers):
                layers[index] = Tap(layer, index)
            self.hub = hub
        return self.hub

    @property
    def n_layers(self) -> int:
        return len(self.model.model.layers)

    def hidden(self, prompt: str):
        """The residual stream after each layer at the last token of `prompt`: [layers, width]."""
        import mlx.core as mx
        import numpy as np

        hub = self.instrument()
        tokens = mx.array(self.tokenizer.encode(prompt, add_special_tokens=False))[None]
        hub.record = []
        try:
            self.model(tokens)
            states = mx.stack(hub.record).astype(mx.float32)
            mx.eval(states)
        finally:
            hub.record = None
        return np.array(states)

    def set_steering(self, steer: Steer | None) -> None:
        """Change the residual stream on every later forward pass, or stop doing so."""
        import mlx.core as mx

        hub = self.instrument()
        if steer is None:
            hub.edit = None
            return
        vector = mx.array(steer.vector)
        unit = vector / mx.linalg.norm(vector)

        def edit(index: int, out):
            if steer.mode == "ablate":
                u = unit.astype(out.dtype)
                return out - (out @ u)[..., None] * u
            if index == steer.layer:
                return out + (steer.alpha * vector).astype(out.dtype)
            return out

        hub.edit = edit


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
        # steer=<path to a .npy vector> with steer_mode, steer_layer and steer_alpha
        # edits the residual stream on every call made through this model.
        self.steer = None
        if "steer" in model_args:
            self.steer = Steer.load(
                model_args.pop("steer"),
                model_args.pop("steer_mode", "ablate"),
                int(model_args.pop("steer_layer", 0)),
                float(model_args.pop("steer_alpha", 1.0)),
            )
        self.model_args = model_args

    def _generate(self, messages, tools, max_tokens, temperature, seed):
        backend = load_backend(self.kind, self.model_name, **self.model_args)
        prompt = backend.render(messages, tools)
        if self.steer is None and getattr(backend, "hub", None) is None:
            return backend.complete(prompt, max_tokens, temperature, seed)
        backend.set_steering(self.steer)
        try:
            return backend.complete(prompt, max_tokens, temperature, seed)
        finally:
            backend.set_steering(None)

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

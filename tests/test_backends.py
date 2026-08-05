"""Reading and editing activations, on tiny randomly initialised models."""

import numpy as np
import pytest

from cordon.models.local import Steer


def projections(states, vector):
    return states @ (vector / np.linalg.norm(vector))


def test_torch_backend_reads_and_edits_the_residual_stream():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from cordon.models.local import TorchBackend

    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )  # fmt: skip
    backend = TorchBackend.__new__(TorchBackend)
    backend.model = transformers.Qwen2ForCausalLM(config).eval()
    backend.device = "cpu"
    backend.encode = lambda prompt: {"input_ids": torch.tensor([[1, 5, 9, 2, 7]])}

    plain = backend.hidden("x")
    assert plain.shape == (3, 32)
    vector = np.random.default_rng(0).standard_normal(32).astype(np.float32)

    backend.set_steering(Steer(vector, mode="ablate"))
    assert np.allclose(projections(backend.hidden("x"), vector), 0, atol=1e-4)

    backend.set_steering(Steer(vector, mode="add", layer=1, alpha=2.0))
    added = backend.hidden("x")
    assert np.allclose(added[0], plain[0], atol=1e-5), "layers before the edit are unchanged"
    assert np.allclose(added[1], plain[1] + 2.0 * vector, atol=1e-4)

    backend.set_steering(None)
    assert np.allclose(backend.hidden("x"), plain, atol=1e-6)


def test_mlx_backend_reads_and_edits_the_residual_stream():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    from cordon.models.local import MLXBackend

    mx.random.seed(0)
    args = ModelArgs(
        model_type="qwen2", hidden_size=32, num_hidden_layers=3, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=64,
    )  # fmt: skip

    class Tokens:
        def encode(self, prompt, add_special_tokens=False):
            return [1, 5, 9, 2, 7]

    backend = MLXBackend.__new__(MLXBackend)
    backend.model, backend.tokenizer, backend.hub, backend._prefixes = (
        Model(args),
        Tokens(),
        None,
        {},
    )

    logits_before = np.array(backend.model(mx.array([[1, 5, 9, 2, 7]])))
    plain = backend.hidden("x")
    assert plain.shape == (3, 32) and backend.n_layers == 3
    # Taps that are not recording or editing leave the model's output alone.
    assert np.allclose(np.array(backend.model(mx.array([[1, 5, 9, 2, 7]]))), logits_before)

    vector = np.random.default_rng(0).standard_normal(32).astype(np.float32)
    backend.set_steering(Steer(vector, mode="ablate"))
    assert np.allclose(projections(backend.hidden("x"), vector), 0, atol=1e-3)

    backend.set_steering(Steer(vector, mode="add", layer=1, alpha=2.0))
    added = backend.hidden("x")
    assert np.allclose(added[0], plain[0], atol=1e-4)
    assert np.allclose(added[1], plain[1] + 2.0 * vector, atol=1e-3)

    backend.set_steering(None)
    assert np.allclose(backend.hidden("x"), plain, atol=1e-5)

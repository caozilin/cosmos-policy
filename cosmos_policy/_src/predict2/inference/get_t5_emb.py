# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
from typing import ClassVar, List, Optional, Tuple, Union

import attrs
import torch
import transformers
from transformers import T5EncoderModel, T5TokenizerFast

transformers.logging.set_verbosity_error()

T5_MODEL_DIR = "checkpoints/google-t5/t5-11b"
T5_HF_REPO = "google-t5/t5-11b"
# Encoder-only export from scripts/export_t5_encoder_only.py (under HF_HUB_CACHE root)
T5_ENCODER_SUBDIR = "t5-11b-encoder"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_DEFAULT_HF_HUB_CACHE = os.path.join(_REPO_ROOT, "hf_cache")


def resolve_hf_hub_cache(cache_dir: Optional[str] = None) -> Optional[str]:
    """Resolve Hugging Face hub cache (``HF_HUB_CACHE`` or repo ``hf_cache/``)."""
    if cache_dir:
        return os.path.abspath(os.path.expanduser(cache_dir))
    env_cache = os.environ.get("HF_HUB_CACHE")
    if env_cache:
        return os.path.abspath(os.path.expanduser(env_cache))
    if os.path.isdir(_DEFAULT_HF_HUB_CACHE):
        return _DEFAULT_HF_HUB_CACHE
    return None


def t5_encoder_only_dir(cache_dir: str) -> str:
    return os.path.join(cache_dir, T5_ENCODER_SUBDIR)


def resolve_t5_encoder_only(cache_dir: Optional[str]) -> Optional[str]:
    """Return local encoder-only export path if ``<cache>/t5-11b-encoder/pytorch_model.bin`` exists."""
    if not cache_dir:
        return None
    path = t5_encoder_only_dir(cache_dir)
    if os.path.isfile(os.path.join(path, "pytorch_model.bin")):
        return path
    return None


def resolve_t5_load_path(cache_dir: Optional[str], model_name: str = T5_HF_REPO) -> str:
    """Prefer encoder-only export; otherwise HuggingFace repo id (full checkpoint)."""
    encoder_path = resolve_t5_encoder_only(cache_dir)
    if encoder_path is not None:
        return encoder_path
    return model_name


def assert_t5_encoder_only(cache_dir: str) -> None:
    path = resolve_t5_encoder_only(cache_dir)
    if path is None:
        raise FileNotFoundError(
            f"Encoder-only weights not found at {t5_encoder_only_dir(cache_dir)}. "
            f"Run: python scripts/export_t5_encoder_only.py"
        )
    missing = [
        n
        for n in ("pytorch_model.bin", "config.json", "spiece.model", "tokenizer.json")
        if not os.path.isfile(os.path.join(path, n))
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete encoder export at {path}: missing {missing}")


def assert_t5_hub_cache(cache_dir: str, model_name: str = T5_HF_REPO) -> None:
    """Fail fast if tokenizer / weights are missing from the local HF cache."""
    repo_slug = model_name.replace("/", "--")
    snapshots = os.path.join(cache_dir, f"models--{repo_slug}", "snapshots")
    if not os.path.isdir(snapshots):
        raise FileNotFoundError(
            f"T5 cache not found under {snapshots}. Set HF_HUB_CACHE to your hf_cache root "
            f"(see FRANKA.md) or download {model_name} into that directory."
        )

    def _has_file(name: str) -> bool:
        for path in glob.glob(os.path.join(snapshots, "*", name)):
            if os.path.isfile(os.path.realpath(path)):
                return True
        return False

    missing = [n for n in ("spiece.model", "tokenizer.json", "pytorch_model.bin") if not _has_file(n)]
    if missing:
        raise FileNotFoundError(
            f"Incomplete T5 cache at {snapshots}: missing {missing}. "
            f"Copy the full hf_cache/models--{repo_slug} tree from a machine that already has T5-11B, "
            f"or run: huggingface-cli download {model_name} --cache-dir {cache_dir}"
        )


def default_local_files_only(local_files_only: Optional[bool] = None) -> bool:
    if local_files_only is not None:
        return local_files_only
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")


def resolve_torch_dtype(dtype: Optional[str] = "fp32") -> torch.dtype:
    """Map user-facing dtype name to ``torch.dtype`` for ``from_pretrained``."""
    if dtype is None or dtype in ("fp32", "float32", "32"):
        return torch.float32
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16", "16"):
        return torch.float16
    raise ValueError(f"Unsupported T5 load dtype: {dtype!r}. Use fp32 or bf16.")


class CosmosT5TextEncoder(torch.nn.Module):
    """Handles T5 text encoding operations."""

    def __init__(
        self,
        model_name: str = "google-t5/t5-11b",
        device: str = "cuda",
        cache_dir=None,
        local_files_only=False,
        max_gpu_mem_gib: Optional[float] = None,
        dtype: str = "fp32",
    ):
        """Initializes the T5 tokenizer and encoder.

        Args:
            model_name: The name of the T5 model to use.
            device: ``cuda``, ``cpu``, or ``auto`` (GPU+CPU offload via device_map).
            max_gpu_mem_gib: When device is ``auto``, cap GPU memory (GiB) and offload the rest to CPU.
            dtype: Weight/compute dtype: ``fp32`` (default) or ``bf16``.
        """
        super().__init__()
        self.torch_dtype = resolve_torch_dtype(dtype)
        cache_dir = resolve_hf_hub_cache(cache_dir)
        local_files_only = default_local_files_only(local_files_only)
        load_path = resolve_t5_load_path(cache_dir, model_name=model_name)
        use_encoder_only = os.path.isdir(load_path)

        if cache_dir is not None:
            if use_encoder_only:
                assert_t5_encoder_only(cache_dir)
            else:
                assert_t5_hub_cache(cache_dir, model_name=model_name)
            print(
                f"Using HF hub cache: {cache_dir} "
                f"(local_files_only={local_files_only}, dtype={self.torch_dtype})"
            )
            if use_encoder_only:
                print(f"Loading encoder-only weights: {load_path}")

        tok_kwargs = (
            dict(local_files_only=True)
            if use_encoder_only
            else dict(cache_dir=cache_dir, local_files_only=local_files_only)
        )
        self.tokenizer = T5TokenizerFast.from_pretrained(
            load_path if use_encoder_only else model_name, **tok_kwargs
        )
        load_kwargs = dict(
            torch_dtype=self.torch_dtype,
            use_safetensors=False,
            low_cpu_mem_usage=True,
        )
        if use_encoder_only:
            load_kwargs["local_files_only"] = True
        else:
            load_kwargs["cache_dir"] = cache_dir
            load_kwargs["local_files_only"] = local_files_only
        self._uses_device_map = False
        if device == "auto":
            if max_gpu_mem_gib is None:
                max_gpu_mem_gib = 12.0
            max_memory = {0: f"{max_gpu_mem_gib}GiB", "cpu": "128GiB"}
            self.text_encoder = T5EncoderModel.from_pretrained(
                load_path, device_map="auto", max_memory=max_memory, **load_kwargs
            )
            self._uses_device_map = True
            self.device = next(self.text_encoder.parameters()).device
        else:
            self.text_encoder = T5EncoderModel.from_pretrained(load_path, **load_kwargs).to(device)
            self.device = device
        self.text_encoder.eval()

    @torch.inference_mode()
    def encode_prompts(
        self, prompts: Union[str, List[str]], max_length: int = 512, return_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encodes text prompts into hidden state representations using a T5 encoder.

        This function tokenizes the input prompts, processes them through a T5 text encoder,
        and returns the last hidden states. The encoded outputs beyond the actual sequence
        length are zero-padded. All prompts in a batch are padded to max_length.

        Args:
            prompts: Input text to encode. Can be a single string or a list of strings.
            max_length: Maximum sequence length for tokenization and padding. Longer
                sequences will be truncated. Defaults to 512.
            return_mask: If True, returns the attention mask along with encoded text.
                Defaults to False.

        Returns:
            If return_mask is False:
                torch.Tensor: Encoded text embeddings of shape (batch_size, max_length, hidden_size).
            If return_mask is True:
                tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - Encoded text embeddings of shape (batch_size, max_length, hidden_size)
                    - Attention mask of shape (batch_size, max_length) as boolean tensor

        Raises:
            ValueError: If the input prompts list is empty.

        Example:
            >>> encoder = CosmosT5TextEncoder()
            >>> prompts = ["Hello world", "Another example"]
            >>> embeddings = encoder.encode_prompts(prompts, max_length=128)
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if not prompts:
            raise ValueError("The input prompt list is empty.")

        batch_encoding = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_length=True,
            return_offsets_mapping=False,
        )

        input_device = self.device if not self._uses_device_map else next(self.text_encoder.parameters()).device
        input_ids = batch_encoding.input_ids.to(input_device)
        attn_mask = batch_encoding.attention_mask.to(input_device)

        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)

        encoded_text = outputs.last_hidden_state
        lengths = attn_mask.sum(dim=1).cpu()

        for batch_id in range(encoded_text.shape[0]):
            encoded_text[batch_id][lengths[batch_id] :] = 0

        if return_mask:
            return encoded_text, attn_mask.bool()
        return encoded_text


@attrs.define(slots=False)
class CosmosT5TextEncoderConfig:
    """
    Config for the T5 text encoder model
    """

    CKPT_PATH: ClassVar[str] = T5_MODEL_DIR
    NUM_TOKENS: ClassVar[int] = 512
    EMBED_DIM: ClassVar[int] = 1024

    ckpt_path: str = CKPT_PATH
    num_tokens: int = NUM_TOKENS
    embed_dim: int = EMBED_DIM


cosmos_encoder: Optional[CosmosT5TextEncoder] = None


def get_text_embedding(
    prompts: Union[str, List[str]],
    encoder: Optional[CosmosT5TextEncoder] = None,
    device: str = "cuda",
    max_length: int = 512,
    return_mask: bool = False,
    cache_dir: Optional[str] = None,
    local_files_only: Optional[bool] = None,
    text_encoder_class: str = "T5",
    max_gpu_mem_gib: Optional[float] = None,
    dtype: str = "fp32",
) -> torch.Tensor:
    """Encodes text prompts into T5 embeddings.

    Args:
        prompts: A single text prompt or a list of text prompts.
        encoder: An optional CosmosT5TextEncoder instance. If None, a global
            instance will be created or reused.
        device: The device to use for computations.
        max_length: The maximum length for the padded embedding.
        text_encoder_class: The class of the text encoder to use.

    Returns:
        A tensor of T5 embeddings.
    """
    assert text_encoder_class == "T5", f"text_encoder_class {text_encoder_class} is not supported"

    global cosmos_encoder
    torch_dtype = resolve_torch_dtype(dtype)

    if encoder is None:
        if cosmos_encoder is None or cosmos_encoder.torch_dtype != torch_dtype:
            cosmos_encoder = CosmosT5TextEncoder(
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                max_gpu_mem_gib=max_gpu_mem_gib,
                dtype=dtype,
            )
        encoder = cosmos_encoder

    if not encoder._uses_device_map:
        encoder.text_encoder.to(device)
        encoder.device = torch.device(device) if isinstance(device, str) else device

    if isinstance(prompts, str):
        prompts = [prompts]

    return encoder.encode_prompts(
        prompts,
        max_length=max_length,
        return_mask=return_mask,
    )

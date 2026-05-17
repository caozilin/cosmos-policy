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

from typing import ClassVar, List, Optional, Tuple, Union

import attrs
import torch
import transformers
from transformers import T5EncoderModel, T5TokenizerFast

transformers.logging.set_verbosity_error()

T5_MODEL_DIR = "checkpoints/google-t5/t5-11b"


class CosmosT5TextEncoder(torch.nn.Module):
    """Handles T5 text encoding operations."""

    def __init__(
        self,
        model_name: str = "google-t5/t5-11b",
        device: str = "cuda",
        cache_dir=None,
        local_files_only=False,
        max_gpu_mem_gib: Optional[float] = None,
    ):
        """Initializes the T5 tokenizer and encoder.

        Args:
            model_name: The name of the T5 model to use.
            device: ``cuda``, ``cpu``, or ``auto`` (GPU+CPU offload via device_map).
            max_gpu_mem_gib: When device is ``auto``, cap GPU memory (GiB) and offload the rest to CPU.
        """
        super().__init__()
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_files_only
        )
        load_kwargs = dict(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            torch_dtype=torch.bfloat16,
            # Local cache has pytorch_model.bin only; avoid re-downloading ~45G safetensors.
            use_safetensors=False,
        )
        self._uses_device_map = False
        if device == "auto":
            if max_gpu_mem_gib is None:
                max_gpu_mem_gib = 12.0
            max_memory = {0: f"{max_gpu_mem_gib}GiB", "cpu": "128GiB"}
            self.text_encoder = T5EncoderModel.from_pretrained(
                model_name, device_map="auto", max_memory=max_memory, **load_kwargs
            )
            self._uses_device_map = True
            self.device = next(self.text_encoder.parameters()).device
        else:
            self.text_encoder = T5EncoderModel.from_pretrained(model_name, **load_kwargs).to(device)
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
    cache_dir: str = None,
    local_files_only: str = False,
    text_encoder_class: str = "T5",
    max_gpu_mem_gib: Optional[float] = None,
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

    if encoder is None:
        if cosmos_encoder is None:
            cosmos_encoder = CosmosT5TextEncoder(
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                max_gpu_mem_gib=max_gpu_mem_gib,
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

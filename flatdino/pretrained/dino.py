# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flax DINO v1 model."""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from flax import nnx
import jax.numpy as jnp

from transformers import FlaxViTModel, ViTModel, ViTConfig
from transformers.models.vit.modeling_flax_vit import FlaxViTModule


__all__ = [
    "ViTConfig",
    "FlaxViTModule",
    "DinoViT",
]


class DinoViT(nnx.Module):
    def __init__(
        self,
        pretrained_path: str = "facebook/dino-vits16",
        resolution: int = 224,
        dtype: jnp.dtype = jnp.float32,
    ):
        flax_encoder = FlaxViTModel.from_pretrained(pretrained_path, from_pt=True, dtype=dtype)
        self.config = ViTConfig(**flax_encoder.config.to_dict())
        if resolution != self.config.image_size:
            raise ValueError(
                f"resolution ({resolution}) must match model image size ({self.config.image_size})"
            )
        network = FlaxViTModule(config=self.config, dtype=dtype)

        variables = {"params": flax_encoder.params}
        self.network = network.bind(variables)
        self.resolution = resolution

        del flax_encoder

    def encode(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
        capture_layers: Optional[Sequence[int]] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, List[jnp.ndarray]]]:
        output_hidden_states = bool(capture_layers)
        outputs = self.network(
            x,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
        )

        if not capture_layers:
            return outputs.last_hidden_state

        activations: List[jnp.ndarray] = []
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Hidden states were not returned by the ViT encoder.")
        num_hidden_layers = self.config.num_hidden_layers
        for layer_idx in capture_layers:
            if layer_idx < 0 or layer_idx >= num_hidden_layers:
                raise ValueError(
                    f"Layer index {layer_idx} is out of bounds for model with "
                    f"{num_hidden_layers} transformer blocks."
                )
            # hidden_states[0] corresponds to patch + class embedding, subsequent
            # entries are the outputs after each transformer block.
            activations.append(hidden_states[layer_idx + 1])

        # Return the final hidden state BEFORE LayerNorm (hidden_states[-1])
        # instead of last_hidden_state which has LayerNorm applied.
        # This matches the reference RAE discriminator which doesn't apply final LayerNorm.
        final_hidden_state = hidden_states[-1]
        return final_hidden_state, activations

    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
        capture_layers: Optional[Sequence[int]] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, List[jnp.ndarray]]]:
        return self.encode(x, deterministic=deterministic, capture_layers=capture_layers)


if __name__ == "__main__":
    import torch
    from einops import rearrange

    huggingface_names = ["facebook/dino-vits16", "facebook/dino-vits8"]

    @torch.no_grad()
    def check_dino_model(hf_name: str, res: int = 224):
        image = np.random.default_rng(0).normal(size=(1, res, res, 3))
        torch_image = rearrange(torch.tensor(image, dtype=torch.float32), "b h w c -> b c h w")
        jax_image = jnp.array(image, dtype=jnp.float32)

        flax_dino = DinoViT(hf_name, resolution=res)
        torch_dino = ViTModel.from_pretrained(hf_name).eval()

        torch_output = torch_dino(pixel_values=torch_image).last_hidden_state
        jax_output = flax_dino(jax_image)

        torch_output, jax_output = torch_output.cpu().numpy(), np.array(jax_output)
        diff = np.abs(torch_output - jax_output)
        print("jax_output.shape: ", jax_output.shape)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)

        print(f"model: {hf_name}\tmean err: {mean_diff}\tmax err: {max_diff}")

    for hf_name in huggingface_names:
        check_dino_model(hf_name)

from typing import Callable, Sequence, Literal
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from tqdm import tqdm


ActivationFn = Callable[[jax.Array], jax.Array]


def _broadcast_to_list(value: Sequence | int | str, length: int) -> list:
    if isinstance(value, str):
        return [value] * length
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"Expected {length} elements, got {len(value)}.")
        return list(value)
    return [value] * length


def _leaky_relu(x: jax.Array) -> jax.Array:
    return jax.nn.leaky_relu(x, negative_slope=0.2)


def _swish(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def _identity(x: jax.Array) -> jax.Array:
    return x


def _get_activation(act_fun: ActivationFn | str) -> ActivationFn:
    if callable(act_fun):
        return act_fun

    if act_fun == "LeakyReLU":
        return _leaky_relu
    if act_fun == "Swish":
        return _swish
    if act_fun == "ELU":
        return jax.nn.elu
    if act_fun == "none":
        return _identity

    raise ValueError(f"Unsupported activation function: {act_fun}")


def _center_crop(x: jax.Array, target_h: int, target_w: int) -> jax.Array:
    h, w = x.shape[1], x.shape[2]
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return x[:, start_h : start_h + target_h, start_w : start_w + target_w, :]


def _concat_with_crop(*tensors: jax.Array) -> jax.Array:
    valid_tensors = [t for t in tensors if t is not None]
    if not valid_tensors:
        raise ValueError("At least one tensor must be provided for concatenation.")

    target_h = min(t.shape[1] for t in valid_tensors)
    target_w = min(t.shape[2] for t in valid_tensors)

    aligned = [_center_crop(t, target_h, target_w) for t in valid_tensors]
    return jnp.concatenate(aligned, axis=-1)


def _upsample(x: jax.Array, mode: str, scale: int = 2) -> jax.Array:
    if scale == 1:
        return x

    n, h, w, c = x.shape
    new_shape = (n, h * scale, w * scale, c)
    if mode == "nearest":
        method = "nearest"
    elif mode == "bilinear":
        method = "bilinear"
    else:
        raise ValueError(f"Unsupported upsample mode: {mode}")

    return jax.image.resize(x, new_shape, method)


class ConvLayer(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        *,
        stride: int = 1,
        bias: bool = True,
        pad: str = "zero",
        downsample_mode: str = "stride",
        rngs: nnx.Rngs,
    ):
        pad = pad.lower()
        self.pad = pad
        self.pad_size = (kernel_size - 1) // 2
        self.downsample_mode = downsample_mode
        self.stride = stride

        conv_stride = stride if downsample_mode == "stride" else 1
        padding = "VALID" if pad == "reflection" else "SAME"

        self.conv = nnx.Conv(
            in_features,
            out_features,
            (kernel_size, kernel_size),
            conv_stride,
            padding=padding,
            use_bias=bias,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.pad == "reflection" and self.pad_size > 0:
            pad_width = (
                (0, 0),
                (self.pad_size, self.pad_size),
                (self.pad_size, self.pad_size),
                (0, 0),
            )
            x = jnp.pad(x, pad_width, mode="reflect")
        elif self.pad not in ("zero", "reflection"):
            raise ValueError(f"Unsupported padding mode: {self.pad}")

        y = self.conv(x)

        if self.downsample_mode == "avg" and self.stride != 1:
            y = nnx.avg_pool(y, (self.stride, self.stride), (self.stride, self.stride), "VALID")
        elif self.downsample_mode == "max" and self.stride != 1:
            y = nnx.max_pool(y, (self.stride, self.stride), (self.stride, self.stride), "VALID")
        elif self.downsample_mode not in ("stride", "avg", "max"):
            raise ValueError(f"Unsupported downsample mode: {self.downsample_mode}")

        return y


class SkipLevel(nnx.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        down_channels: int,
        deeper_channels: int,
        up_channels: int,
        filter_size_down: int,
        filter_size_up: int,
        filter_skip_size: int,
        *,
        pad: str,
        need_bias: bool,
        downsample_mode: str,
        upsample_mode: str,
        need1x1_up: bool,
        rngs: nnx.Rngs,
    ):
        self.has_skip = skip_channels > 0
        self.need1x1_up = need1x1_up
        self.upsample_mode = upsample_mode

        if self.has_skip:
            self.skip_conv = ConvLayer(
                input_channels,
                skip_channels,
                filter_skip_size,
                bias=need_bias,
                pad=pad,
                rngs=rngs,
            )
            self.skip_bn = nnx.BatchNorm(skip_channels, rngs=rngs)
        else:
            self.skip_conv = None
            self.skip_bn = None

        self.down_conv1 = ConvLayer(
            input_channels,
            down_channels,
            filter_size_down,
            stride=2,
            bias=need_bias,
            pad=pad,
            downsample_mode=downsample_mode,
            rngs=rngs,
        )
        self.down_bn1 = nnx.BatchNorm(down_channels, rngs=rngs)

        self.down_conv2 = ConvLayer(
            down_channels,
            down_channels,
            filter_size_down,
            bias=need_bias,
            pad=pad,
            rngs=rngs,
        )
        self.down_bn2 = nnx.BatchNorm(down_channels, rngs=rngs)

        concat_channels = (skip_channels if self.has_skip else 0) + deeper_channels
        self.concat_bn = nnx.BatchNorm(concat_channels, rngs=rngs)

        self.up_conv = ConvLayer(
            concat_channels,
            up_channels,
            filter_size_up,
            bias=need_bias,
            pad=pad,
            rngs=rngs,
        )
        self.up_bn = nnx.BatchNorm(up_channels, rngs=rngs)

        if need1x1_up:
            self.up1x1_conv = ConvLayer(
                up_channels,
                up_channels,
                1,
                bias=need_bias,
                pad=pad,
                rngs=rngs,
            )
            self.up1x1_bn = nnx.BatchNorm(up_channels, rngs=rngs)
        else:
            self.up1x1_conv = None
            self.up1x1_bn = None


class SkipNet(nnx.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_channels_down: Sequence[int],
        num_channels_up: Sequence[int],
        num_channels_skip: Sequence[int],
        filter_size_down: Sequence[int],
        filter_size_up: Sequence[int],
        filter_skip_size: int | Sequence[int],
        *,
        need_sigmoid: bool,
        need_tanh: bool,
        need_bias: bool,
        pad: str,
        upsample_mode: Sequence[str] | str,
        downsample_mode: Sequence[str] | str,
        act_fun: ActivationFn | str,
        need1x1_up: bool,
        rngs: nnx.Rngs,
    ):
        num_channels_down = list(num_channels_down)
        num_channels_up = list(num_channels_up)
        num_channels_skip = list(num_channels_skip)

        n_scales = len(num_channels_down)
        if not (len(num_channels_up) == len(num_channels_skip) == n_scales):
            raise ValueError("num_channels_down/up/skip must have the same length.")

        filter_size_down = _broadcast_to_list(filter_size_down, n_scales)
        filter_size_up = _broadcast_to_list(filter_size_up, n_scales)
        filter_skip_size = _broadcast_to_list(filter_skip_size, n_scales)
        upsample_mode = _broadcast_to_list(upsample_mode, n_scales)
        downsample_mode = _broadcast_to_list(downsample_mode, n_scales)

        self.act_fn = _get_activation(act_fun)
        self.need_sigmoid = need_sigmoid
        self.need_tanh = need_tanh
        self.need1x1_up = need1x1_up
        self.last_level = n_scales - 1

        levels = []
        current_channels = input_channels
        for i in range(n_scales):
            deeper_channels = (
                num_channels_down[i] if i == self.last_level else num_channels_up[i + 1]
            )
            level = SkipLevel(
                input_channels=current_channels,
                skip_channels=num_channels_skip[i],
                down_channels=num_channels_down[i],
                deeper_channels=deeper_channels,
                up_channels=num_channels_up[i],
                filter_size_down=filter_size_down[i],
                filter_size_up=filter_size_up[i],
                filter_skip_size=filter_skip_size[i],
                pad=pad,
                need_bias=need_bias,
                downsample_mode=downsample_mode[i],
                upsample_mode=upsample_mode[i],
                need1x1_up=need1x1_up,
                rngs=rngs,
            )
            levels.append(level)
            current_channels = num_channels_down[i]

        self.levels = nnx.List(levels)
        self.final_conv = ConvLayer(
            num_channels_up[0],
            output_channels,
            1,
            bias=need_bias,
            pad=pad,
            rngs=rngs,
        )

    def _forward_level(self, idx: int, x: jax.Array) -> jax.Array:
        level: SkipLevel = self.levels[idx]
        act_fn = self.act_fn

        skip_out = None
        if level.has_skip:
            skip_out = level.skip_conv(x)
            skip_out = level.skip_bn(skip_out)
            skip_out = act_fn(skip_out)

        deeper = level.down_conv1(x)
        deeper = level.down_bn1(deeper)
        deeper = act_fn(deeper)

        deeper = level.down_conv2(deeper)
        deeper = level.down_bn2(deeper)
        deeper = act_fn(deeper)

        if idx < self.last_level:
            deeper = self._forward_level(idx + 1, deeper)

        deeper = _upsample(deeper, level.upsample_mode)

        if skip_out is not None:
            combined = _concat_with_crop(skip_out, deeper)
        else:
            combined = deeper

        combined = level.concat_bn(combined)
        out = level.up_conv(combined)
        out = level.up_bn(out)
        out = act_fn(out)

        if level.need1x1_up:
            out = level.up1x1_conv(out)
            out = level.up1x1_bn(out)
            out = act_fn(out)

        return out

    def __call__(self, x: jax.Array) -> jax.Array:
        out = self._forward_level(0, x)
        out = self.final_conv(out)
        if self.need_sigmoid:
            out = jax.nn.sigmoid(out)
        elif self.need_tanh:
            out = jnp.tanh(out)
        return out


@dataclass
class TokenInvConfig:
    batch_size: int = 64
    seed: int = 42

    learning_rate: float = 0.01
    iterations: int = 20_000
    reduce_noise_stage_1_iter: int = 10_000
    reduce_noise_stage_2_iter: int = 15_000
    input_depth: int = 32

    scale_1: float = 10.0
    scale_2: float = 2.0
    scale_3: float = 0.5

    num_channels_down: tuple[int, ...] = (16, 32, 64, 128, 128, 128)
    num_channels_up: tuple[int, ...] = (16, 32, 64, 128, 128, 128)
    num_channels_skip: tuple[int, ...] = (4, 4, 4, 4, 4, 4)
    filter_size_down: tuple[int, ...] = (7, 7, 5, 5, 3, 3)
    filter_size_up: tuple[int, ...] = (7, 7, 5, 5, 3, 3)
    filter_skip_size: int | tuple[int, ...] = 1
    need_sigmoid: bool = True
    need_tanh: bool = False
    need_bias: bool = True
    pad: str = "reflection"
    upsample_mode: str | tuple[str, ...] = "bilinear"
    downsample_mode: str | tuple[str, ...] = "stride"
    act_fun: Literal["LeakyReLU", "Swish", "ELU"] = "LeakyReLU"
    need1x1_up: bool = True


def invert_token(
    cfg: TokenInvConfig,
    token_fn: Callable[[jax.Array], jax.Array],
    target_token: jax.Array,
    h: int,
    w: int,
) -> jax.Array:
    """Inverts a given token back into the model input that explains that token."""

    @nnx.jit
    def iteration_step(
        unet: SkipNet,
        optim: nnx.Optimizer,
        noise: jax.Array,
        cls_: jax.Array,
        key: jax.Array,
        noise_reg_scale: float,
    ):
        def loss_fn(unet: SkipNet, noise: jax.Array, cls_: jax.Array):
            noisy_input = (
                noise + jax.random.normal(key, noise.shape, dtype=noise.dtype) * noise_reg_scale
            )
            image = unet(noisy_input)
            pred_token = token_fn(image)
            return jnp.mean(jnp.square(pred_token - cls_))

        loss, grads = nnx.value_and_grad(loss_fn)(unet, noise, cls_)
        optim.update(unet, grads)
        return loss

    mesh = jax.make_mesh((1, 1), ("data", "model"))
    jax.set_mesh(mesh)
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("data", None, None, None)
    )

    rngs = nnx.Rngs(cfg.seed)
    unet = SkipNet(
        input_channels=cfg.input_depth,
        output_channels=3,
        num_channels_down=cfg.num_channels_down,
        num_channels_up=cfg.num_channels_up,
        num_channels_skip=cfg.num_channels_skip,
        filter_size_down=cfg.filter_size_down,
        filter_size_up=cfg.filter_size_up,
        filter_skip_size=cfg.filter_skip_size,
        need_sigmoid=cfg.need_sigmoid,
        need_tanh=cfg.need_tanh,
        need_bias=cfg.need_bias,
        pad=cfg.pad,
        upsample_mode=cfg.upsample_mode,
        downsample_mode=cfg.downsample_mode,
        act_fun=cfg.act_fun,
        need1x1_up=cfg.need1x1_up,
        rngs=rngs,
    )
    optim = nnx.Optimizer(
        unet,
        optax.chain(optax.clip_by_global_norm(10.0), optax.adamw(cfg.learning_rate)),
        wrt=nnx.Param,
    )

    noisy_input = jax.random.normal(rngs(), (1, h, w, cfg.input_depth), dtype=target_token.dtype)
    noisy_input = jax.device_put(noisy_input, sharding)

    pbar = tqdm(range(cfg.iterations), desc="inversion", ncols=80)
    for step in pbar:
        if step < cfg.reduce_noise_stage_1_iter:
            scale = cfg.scale_1
        elif step < cfg.reduce_noise_stage_2_iter:
            scale = cfg.scale_2
        else:
            scale = cfg.scale_3

        loss = iteration_step(unet, optim, noisy_input, target_token, rngs(), scale)
        pbar.set_postfix({"loss": loss.item()})

    unet.eval()
    return unet(noisy_input)

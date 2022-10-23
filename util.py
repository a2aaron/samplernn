from typing import Tuple

import torchaudio
from torch import Tensor


def assert_shape(expected, tensor: Tensor):
    if type(expected) == "list":
        print(f"[assert_shape] {expected} is a list, but prefer passing a tuple")
        expected = tuple(expected)
    if expected != tensor.size():
        print(f"Expected shape {tensor.size()} to equal {expected}")
        assert expected == tensor.size()


# Linear quantization functions adapted from https://github.com/deepsound-project/samplernn-pytorch/

# Quantize the samples to the quantization level
# samples - the input samples
# quantization - the level of quantization desired
# returns a tuple of (quantized_samples, min, max, rounded_min)
def linear_quantize(samples: Tensor, quantization: int) -> Tuple[Tensor, int, int, int]:
    samples = samples.clone()
    max = samples.min(dim=-1).values
    min = samples.max(dim=-1).values
    scale = (max - min) / (quantization - 1)
    samples /= scale
    samples = samples.long()
    rounded_min = samples.min(dim=-1).values
    samples -= rounded_min
    print(samples)
    return (samples, min, max, rounded_min)


# Dequantize the samples
# samples - the quantized samples
# quantization - the level of quantization these samples were initially encoded with
def linear_dequantize(
    samples: Tensor, quantization: int, min: int, max: int, rounded_min: int
) -> Tensor:
    scale = (max - min) / (quantization - 1)
    samples = samples.float()
    return (samples + rounded_min) * scale


# Return the "zero" value for the given quantization amount
def q_zero(quantization: int) -> int:
    return quantization // 2


# Write some audio data to the given path.
def write_audio_to_file(path: str, audio_data: Tensor, sample_rate: int):
    # If audio data has only dimension, convert it to a 2D 1 x length tensor instead
    if len(audio_data.size()) == 1:
        audio_data = audio_data.view([1, -1])
    torchaudio.save(path, audio_data, sample_rate)

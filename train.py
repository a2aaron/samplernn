from typing import Tuple
import torchaudio
import torch

from torch import Tensor
from torch.utils.data import Dataset

import model


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


class SongDataset(Dataset):
    # Create a new SongDataset from the given path.
    # `path` must be to a file with sound.
    # `frame_size` must be an integer specifying the size of the frames returned, in samples
    # Note that only the left channel is used, if there is more than one channel.
    def __init__(self, path, frame_size, quantization) -> None:
        super().__init__()
        (audio, self.sample_rate) = torchaudio.load(path)

        # If there are two channels, only take the left channel.
        audio = audio[0]
        print(audio.size())

        (self.audio, self.min, self.max, self.rounded_min) = linear_quantize(
            audio, quantization
        )
        print(self.audio)
        self.length = self.audio.size()[0]
        self.frame_size = frame_size
        self.quantization = quantization

    def __len__(self) -> int:
        pass

    def __getitem__(self, index) -> Tensor:
        return super().__getitem__(index)

    def dequantize_with(self, data: Tensor) -> Tensor:
        return linear_dequantize(
            data, self.quantization, self.min, self.max, self.rounded_min
        )


MODEL = model.SampleRNN()

if __name__ == "__main__":
    dataset = SongDataset("inputs/drumloop.wav", model.FRAME_SIZE, model.QUANTIZATION)
    write_audio_to_file(
        "ground_truth.wav",
        dataset.dequantize_with(dataset.audio),
        dataset.sample_rate,
    )
    print(dataset.audio.size())
    print(MODEL)

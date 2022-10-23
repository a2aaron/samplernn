from typing import Tuple
import torchaudio
import torch

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import model
from model import BATCH_SIZE, FRAME_SIZE, HIDDEN_SIZE, NUM_FRAMES, QUANTIZATION
from util import linear_dequantize, linear_quantize, write_audio_to_file


class SongDataset(Dataset):
    # Create a new SongDataset from the given path.
    # `path` must be to a file with sound.
    # `num_frames` - the number of frames per returned element
    # `frame_size` - the size of the frames per returned element, in samples
    # `quantization` - the quantization level to use
    # Note that only the left channel is used, if there is more than one channel.
    def __init__(self, path: str, num_frames: int, frame_size: int, quantization: int):
        super().__init__()
        (audio, self.sample_rate) = torchaudio.load(path)

        # If there are two channels, only take the left channel.
        audio = audio[0]

        (self.audio, self.min, self.max, self.rounded_min) = linear_quantize(
            audio, quantization
        )
        self.length = self.audio.size()[0]
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.quantization = quantization

    def __len__(self) -> int:
        return self.length - (self.frame_size * (self.num_frames + 1))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        seq_len = self.frame_size * self.num_frames
        input = self.audio[index : index + seq_len]
        overlap = self.audio[index : index + seq_len + self.frame_size]
        target = self.audio[index + self.frame_size : index + seq_len + self.frame_size]
        return input, overlap, target

    def dequantize_with(self, data: Tensor) -> Tensor:
        return linear_dequantize(
            data, self.quantization, self.min, self.max, self.rounded_min
        )

    def write_to_file_with(self, path, data: Tensor):
        dequantized = self.dequantize_with(data)
        write_audio_to_file(path, dequantized, self.sample_rate)


MODEL = model.SampleRNN()

if __name__ == "__main__":

    if torch.backends.mps.is_available():
        print("Using MPS device ")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA device ")
        device = torch.device("cuda")
    else:
        print("Using CPU device ")
        device = torch.device("cpu")

    PREFIX = "outputs/drumloop/drum"
    dataset = SongDataset(
        path="inputs/low/drumloop.wav",
        num_frames=NUM_FRAMES,
        frame_size=FRAME_SIZE,
        quantization=QUANTIZATION,
    )
    dataset.write_to_file_with(f"{PREFIX}_ground_truth.wav", dataset.audio)
    dataset.write_to_file_with(f"{PREFIX}_input_example.wav", dataset.__getitem__(0)[0])
    dataset.write_to_file_with(
        f"{PREFIX}_overlap_example.wav", dataset.__getitem__(0)[1]
    )
    dataset.write_to_file_with(
        f"{PREFIX}_target_example.wav", dataset.__getitem__(0)[2]
    )

    print(MODEL)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optim = torch.optim.Adam(MODEL.parameters())

    epoch_i = 0
    while True:
        for batch_i, batch in enumerate(dataloader):
            (hidden, cell) = MODEL.frame_level_rnn.init_state(BATCH_SIZE)
            (input, overlap, target) = batch

            input = dataset.dequantize_with(input) * 2.0
            input = input.view(BATCH_SIZE, NUM_FRAMES, FRAME_SIZE)
            (conditioning, new_hidden, new_cell) = MODEL.frame_level_rnn.forward(
                input, hidden, cell
            )
            unfold = overlap.unfold(1, FRAME_SIZE, 1)[:, :-1, :]
            unfold = unfold.reshape([BATCH_SIZE * NUM_FRAMES * FRAME_SIZE, FRAME_SIZE])
            conditioning = conditioning.reshape(
                [BATCH_SIZE * NUM_FRAMES * FRAME_SIZE, HIDDEN_SIZE]
            )
            logits = MODEL.sample_predictor.forward(unfold, conditioning)
            logits = logits.softmax(dim=1)

            target = target.reshape([BATCH_SIZE * NUM_FRAMES * FRAME_SIZE])
            loss = torch.nn.CrossEntropyLoss()
            loss = loss(logits, target)
            loss.backward()
            optim.step()

            print(f"Batch {batch_i}/{len(dataloader)}")
        print(f"Epoch {epoch_i}")
        epoch_i += 1

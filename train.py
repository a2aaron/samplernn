from typing import Tuple

import os
import argparse
import torchaudio
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import model
from util import linear_dequantize, linear_quantize, q_zero, write_audio_to_file


class SongDataset(Dataset):
    # Create a new SongDataset from the given path.
    # `path` must be to a file with sound.
    # `num_frames` - the number of frames per returned element
    # `frame_size` - the size of the frames per returned element, in samples
    # `quantization` - the quantization level to use
    # Note that only the left channel is used, if there is more than one channel.
    def __init__(
        self,
        path: str,
        num_frames: int,
        frame_size: int,
        quantization: int,
        device: torch.device,
    ):
        super().__init__()
        (audio, self.sample_rate) = torchaudio.load(path)

        # If there are two channels, only take the left channel.
        audio = audio[0]

        (audio, self.min, self.max, self.rounded_min) = linear_quantize(
            audio, quantization
        )

        unfold = audio.unfold(0, frame_size, 1)

        self.audio = audio.to(device)
        self.unfold = unfold.to(device)
        self.length = self.audio.size()[0]
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.quantization = quantization

    def __len__(self) -> int:
        return self.length - (self.frame_size * (self.num_frames + 1))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        seq_len = self.frame_size * self.num_frames
        input = self.audio[index : index + seq_len]
        unfold = self.unfold[index : index + seq_len]
        target = self.audio[index + self.frame_size : index + seq_len + self.frame_size]
        return input, unfold, target

    def dequantize_with(self, data: Tensor) -> Tensor:
        return linear_dequantize(
            data, self.quantization, self.min, self.max, self.rounded_min
        )

    def write_to_file_with(self, path, data: Tensor):
        dequantized = self.dequantize_with(data)
        write_audio_to_file(path, dequantized, self.sample_rate)


def generate(
    model: model.SampleRNN, device: torch.device, dataset: SongDataset, length: int
):
    frame_size = model.frame_size
    quantization = model.quantization
    hidden_size = model.hidden_size

    samples = torch.zeros(length, dtype=torch.long, device=device)
    # set first frame to zeros.
    samples[0:frame_size] = q_zero(quantization)

    (hidden, cell) = model.frame_level_rnn.init_state(1, device)
    conditioning = None
    for t in range(frame_size, length):
        if t % frame_size == 0:
            frame = samples[t - frame_size : t]
            frame = dataset.dequantize_with(frame) * 2.0
            frame = frame.reshape([1, 1, frame_size])
            conditioning, hidden, cell = model.frame_level_rnn.forward(
                frame, hidden, cell, 1, 1
            )
            conditioning = conditioning.reshape([frame_size, hidden_size])
        frame = samples[t - frame_size : t]
        frame = frame.reshape([1, frame_size])
        this_conditioning = conditioning[t % frame_size].reshape([1, hidden_size])
        logits = model.sample_predictor.forward(frame, this_conditioning, 1)
        sample = logits.softmax(-1).multinomial(1, replacement=False)

        samples[t] = sample
        if t % 10000 == 0:
            print(f"Generated {t}/{length} samples ({100.0 * t/length:.2f})%")
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("-o", "--out", required=True, help="output folder name")
    parser.add_argument(
        "-i", "--in", dest="in_path", required=True, help="path to input music file"
    )
    parser.add_argument(
        "--frame_size", type=int, default=16, help="size of frames, in samples"
    )
    parser.add_argument(
        "--rnn_layers", type=int, default=5, help="number of RNN layers in the LSTM"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1024,
        help="number of neurons in RNN and MLP layers",
    )
    parser.add_argument(
        "--embed_size",
        type=int,
        default=256,
        help="size of embedding in MLP layer",
    )
    parser.add_argument(
        "--quantization",
        type=int,
        default=256,
        help="number of quantization values for samples",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=64,
        help="number of frames to use in truncated BPTT training",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--sample_rate",
        type=int,
        help="sample rate of the training data and generated sound",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=10.0,
        help="length of samples, in seconds, to generate",
    )
    parser.add_argument(
        "--generate_every",
        type=int,
        default=100,
        help="how often to generate samples, in batch iterations",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="how often to checkpoint, in batch iterations",
    )

    args = parser.parse_args()

    if torch.backends.mps.is_available():
        # print("Using MPS device")
        # DEVICE = torch.device("mps")
        print("NOT using MPS device--Using CPU instead!")
        print("Currently, the LSTMs in pytorch are broken on MPS (as of Oct 22, 2022)")
        print("See the following issues")
        print("https://github.com/pytorch/pytorch/issues/80306")
        print("https://github.com/pytorch/pytorch/issues/83144")
        DEVICE = torch.device("cpu")
    elif torch.cuda.is_available():
        print("Using CUDA device ")
        DEVICE = torch.device("cuda")
    else:
        print("Using CPU device ")
        DEVICE = torch.device("cpu")

    the_model = model.SampleRNN(
        frame_size=args.frame_size,
        hidden_size=args.hidden_size,
        rnn_layers=args.rnn_layers,
        embed_size=args.embed_size,
        quantization=args.quantization,
    )
    the_model.to(DEVICE)

    batch_size = args.batch_size
    num_frames = args.num_frames
    frame_size = args.frame_size
    hidden_size = args.hidden_size

    PREFIX = f"outputs/{args.out}/{args.out}"
    os.makedirs(f"outputs/{args.out}/", exist_ok=True)

    dataset = SongDataset(
        path=args.in_path,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        quantization=args.quantization,
        device=DEVICE,
    )
    dataset.write_to_file_with(f"{PREFIX}_ground_truth.wav", dataset.audio)
    dataset.write_to_file_with(f"{PREFIX}_input_example.wav", dataset.__getitem__(0)[0])
    dataset.write_to_file_with(
        f"{PREFIX}_overlap_example.wav", dataset.__getitem__(0)[1].flatten()
    )
    dataset.write_to_file_with(
        f"{PREFIX}_target_example.wav", dataset.__getitem__(0)[2]
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optim = torch.optim.Adam(the_model.parameters())

    epoch_i = 0
    while True:
        for batch_i, batch in enumerate(dataloader):
            (hidden, cell) = the_model.frame_level_rnn.init_state(batch_size, DEVICE)
            (input, unfold, target) = batch

            input = dataset.dequantize_with(input) * 2.0
            input = input.view(batch_size, num_frames, frame_size)
            (conditioning, new_hidden, new_cell) = the_model.frame_level_rnn.forward(
                input, hidden, cell, batch_size, num_frames
            )
            unfold = unfold.reshape([batch_size * num_frames * frame_size, frame_size])

            conditioning = conditioning.reshape(
                [batch_size * num_frames * frame_size, hidden_size]
            )
            logits = the_model.sample_predictor.forward(
                unfold, conditioning, batch_size * num_frames * frame_size
            )

            target = target.reshape([batch_size * num_frames * frame_size])
            loss = torch.nn.CrossEntropyLoss()
            loss = loss(logits, target)
            num_correct = logits.argmax(-1, keepdim=False).eq(target).count_nonzero()
            accuracy = num_correct / (batch_size * num_frames * frame_size)

            optim.zero_grad()
            loss.backward()
            optim.step()

            print(
                f"Batch {batch_i}/{len(dataloader)}, loss: {loss:.4f}, accuracy: {100.0 * accuracy:.2f}%"
            )

            if batch_i % args.generate_every == 0:
                length = int(args.length * dataset.sample_rate)
                samples = generate(the_model, DEVICE, dataset, length)
                dataset.write_to_file_with(
                    f"{PREFIX}_epoch_{epoch_i}_batch_{batch_i}.wav", samples
                )
        print(f"Epoch {epoch_i}")
        epoch_i += 1

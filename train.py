import random
import time
from typing import Tuple, Union

import json
import os
import argparse
import torchaudio
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from sample_rnn import SampleRNN
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

    def get_frame(self, index: int) -> Tensor:
        input = self.audio[index : index + self.frame_size]
        return input

    def dequantize_with(self, data: Tensor) -> Tensor:
        return linear_dequantize(
            data, self.quantization, self.min, self.max, self.rounded_min
        )

    def write_to_file_with(self, path, data: Tensor):
        dequantized = self.dequantize_with(data)
        write_audio_to_file(path, dequantized, self.sample_rate)


GENERATION_GRAPH = None

# Wrapper struct for the CUDAGraph which can generate audio of files.
# We use a CUDAGraph in order to speed up generation on GPU--generating sample
# by sample normally requires communicating between the CPU and GPU for each
# sample. This incurs a ton of overhead and is very slow. The CUDAGraph enables
# us to generate audio in batches, skipping the CPU-GPU overhead.
class GenerationGraph:
    # Initalize a GenerationGraph. The model must not be deallocated while the
    # GenerationGraph is alive.
    # `num_frames` controls how many frames of audio are generated per call of
    # `replay`. A higher value means more frames are generated at once, but also
    # requires more memory and compute.
    # In particular, the number of samples generated when replay is called is
    # equal to num_frames * frame_size.
    # The CUDAGraph is extremely memory intensive--you probably cannot set num_frames
    # to an extremely large value.
    def __init__(self, model: SampleRNN, device: torch.device, num_frames: int):
        frame_size = model.frame_size
        hidden_size = model.hidden_size
        length = num_frames * frame_size

        self.generation_num_frames = num_frames
        self.generation_size = num_frames * frame_size

        self.samples = torch.zeros(length, dtype=torch.long, device=device)
        (self.hidden, self.cell) = model.frame_level_rnn.init_state(1, device)

        # Create the CUDAGraph. Think of the `with` block as a closure which
        # automatically captures any variables. This means that any variables it
        # captures also need to remain alive (in particular: these are `self.samples`,
        # `self.hidden`, `self.cell`, and the model)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            hidden = self.hidden
            cell = self.cell

            for t in range(frame_size, length):
                if t % frame_size == 0:
                    frame = self.samples[t - frame_size : t]
                    frame = dataset.dequantize_with(frame) * 2.0
                    frame = frame.reshape([1, 1, frame_size])
                    conditioning, hidden, cell = model.frame_level_rnn.forward(
                        frame, hidden, cell, 1, 1
                    )
                    conditioning = conditioning.reshape([frame_size, hidden_size])

                frame = self.samples[t - frame_size : t]
                frame = frame.reshape([1, frame_size])
                # Note that conditioning vectors are shaped like
                # [batch_size, frame_size, frame_size * hidden_size]
                # or, in this case, [1, frame_size, frame_size * hidden_size]
                # Each vector of [frame_size * hidden_size] is for each of the
                # of the sliding window frames. Since there are `frame_size` sliding
                # windows for each frame, there are also `frame_size` conditioning
                # vectors. Hence ,we pick out the right conditioning vector
                # (This explains why we do conditioning[t % frame_size])
                this_conditioning = conditioning[t % frame_size].reshape(
                    [1, hidden_size]
                )
                logits = model.sample_predictor.forward(frame, this_conditioning, 1)
                sample = logits.softmax(-1)
                # Note that replacement here doesn't actually matter, since we
                # only draw from one sample. However, CUDAGraph doesn't like it
                # when we disallow replacement (probably because this requires
                # some amount of CPU communication, which isn't possible in a
                # CUDAGraph). Hence, we set it to false here.
                sample = sample.multinomial(1, replacement=True)

                self.samples[t] = sample
            self.hidden.copy_(hidden)
            self.cell.copy_(cell)
        self.graph = g

    # Set the initial state for generation. This sets `self.hidden` and `self.cell`
    # to the initial zero-state, as well as setting up `self.samples` to have the
    # given prompt.
    def init_state(
        self, model: SampleRNN, device: torch.device, prompt_samples: Tensor
    ):
        self.samples.fill_(0)
        self.samples[0:frame_size].copy_(prompt_samples)

        (init_hidden, init_cell) = model.frame_level_rnn.init_state(1, device)
        self.hidden.copy_(init_hidden)
        self.cell.copy_(init_cell)

    # Set the subseqent states for generation. This sets `self.samples` to have
    # the prompt samples at the beginning, followed entirely by zeros.
    # `prompt_samples` should be a 1D tensor of length `frame_size``
    def set_prompt(self, prompt_samples: Tensor):
        self.samples.fill_(0)
        self.samples[0:frame_size].copy_(prompt_samples)

    # Replay the CUDAGraph. This will result in two things:
    # First, `self.samples` will be filled with the generated samples (note that
    # the first `frame_size` samples are the prompt and are unchanged--the rest
    # of the tensor contains the generated part)
    # Second, `self.hidden` and `self.cell` are updated to the last hidden/cell
    # states produced by generation
    def replay(self):
        self.graph.replay()


# Run generation on the model. This returns a 1D Tensor containing approximately
# `length` samples. (The tensor may not be exactly length samples). The tensor's
# first `frame_size` samples consist of a prompt from passed in `dataset`
def generate(
    model: SampleRNN, device: torch.device, dataset: SongDataset, length: int
) -> Tensor:
    global GENERATION_GRAPH
    # Intialize the CUDAGraph if it is not already initalized
    if GENERATION_GRAPH is None:
        now = time.time()
        # TODO: Is 128 frames the optimal number of frames?
        GENERATION_GRAPH = GenerationGraph(model, device, 128)
        print(f"Took {pretty_elapsed(now)} to build graph.")

    frame_size = model.frame_size
    hidden_size = model.hidden_size

    # The number of times we will run the GENERATION_GRAPH.
    num_generation_frames = length // GENERATION_GRAPH.generation_size

    # The tensor containing the returned samples.
    out_samples = torch.zeros(
        [num_generation_frames, GENERATION_GRAPH.generation_size - frame_size],
        dtype=torch.long,
        device=device,
    )

    now = time.time()

    prompt = dataset.get_frame(random.randrange(0, dataset.length - dataset.frame_size))
    GENERATION_GRAPH.init_state(model, device, prompt)

    for i in range(0, num_generation_frames):
        GENERATION_GRAPH.replay()
        # Copy out the samples--we can't just assign since GENERATION_GRAPH.samples
        # will be overwritten, so we need to fully extract the samples
        out_samples[i].copy_(GENERATION_GRAPH.samples[:-frame_size])
        # We need to clone GENERATION_GRAPH.samples here because otherwise will
        # attempting to copy a Tensor to itself, which PyTorch doesn't like
        new_prompt = GENERATION_GRAPH.samples[-frame_size:].clone()
        GENERATION_GRAPH.set_prompt(new_prompt)

    print(f"Took {pretty_elapsed(now)} to replay graph")
    # Include the prompt with the generateds samples.
    return torch.cat((prompt, out_samples.flatten()))


# Write the updatable args from the command line arguments object
# This writes a JSON file to the given path. If the file cannot be written, an
# exception is not thrown--instead we just print to console.
def write_args(path: str, args):
    try:
        with open(path, "wt") as file:
            args = {
                "length": args.length,
                "generate_every": args.generate_every,
                "checkpoint_every": args.checkpoint_every,
                "num_generated": args.num_generated,
            }
            json.dump(
                args,
                file,
                sort_keys=True,
                indent=4,
            )
    except Exception as e:
        print(f"Couldn't write {args} to file {path}, Reason:", e)


# Read the updatable args from the given path. This returns a dictionary containing
# `length`, `generate_every`, `checkpoint_every`, and `num_generated` as keys.
# If the path cannot be read, `None` is returned instead
def read_args(path) -> Union[dict, None]:
    try:
        with open(path, "rt") as file:
            args = json.load(file)
            for arg in [
                "length",
                "generate_every",
                "checkpoint_every",
                "num_generated",
            ]:
                if arg not in args.keys():
                    print(f"[Warning] Parsed json {args} does not have argument {arg}!")
            return args
    except Exception as e:
        print(f"Couldn't read from file {path}, Reason:", e)
        return None


# Write a CSV file containign the losses
def write_csv(path, losses):
    try:
        losses_str = "iter,loss\n"
        for loss_i, loss in enumerate(losses):
            losses_str += f"{loss_i}, {loss}\n"
        with open(path, "wt") as file:
            file.write(losses_str)
    except Exception as e:
        print(f"Couldn't write losses to file {path}, Reason:", e)


# Pretty print the elapsed time.
def pretty_elapsed(now: float) -> str:
    elapsed = time.time() - now
    if elapsed < 1:
        return f"{int(elapsed * 1000)}ms"
    if elapsed < 10:
        return f"{elapsed:.2f}s"
    return f"{elapsed:.1f}s"


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
        "--length",
        type=float,
        default=10.0,
        help="length of samples, in seconds, to generate",
    )
    parser.add_argument(
        "--num_generated",
        type=int,
        default=1,
        help="how many files to generate at once",
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="if passed, load model weights from a checkpoint",
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

    model = SampleRNN(
        frame_size=args.frame_size,
        hidden_size=args.hidden_size,
        rnn_layers=args.rnn_layers,
        embed_size=args.embed_size,
        quantization=args.quantization,
    ).to(DEVICE)

    generate_every = args.generate_every
    checkpoint_every = args.checkpoint_every
    length = args.length
    num_generated = args.num_generated

    PREFIX = f"outputs/{args.out}/{args.out}"
    ARGS_FILE = f"{PREFIX}_args.json"
    os.makedirs(f"outputs/{args.out}/", exist_ok=True)
    write_args(ARGS_FILE, args)

    dataset = SongDataset(
        path=args.in_path,
        num_frames=args.num_frames,
        frame_size=model.frame_size,
        quantization=model.quantization,
        device=DEVICE,
    )
    dataset.write_to_file_with(f"{PREFIX}_ground_truth.wav", dataset.audio)
    rand_i = random.randrange(0, len(dataset))
    dataset.write_to_file_with(
        f"{PREFIX}_input_example.wav", dataset.__getitem__(rand_i)[0]
    )
    dataset.write_to_file_with(
        f"{PREFIX}_overlap_example.wav", dataset.__getitem__(rand_i)[1].flatten()
    )
    dataset.write_to_file_with(
        f"{PREFIX}_target_example.wav", dataset.__getitem__(rand_i)[2]
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optim = torch.optim.Adam(model.parameters())

    epoch_i = 0
    iter_i = 0
    losses = []

    if args.resume is not None:
        resume_info = torch.load(args.resume)
        model.load_state_dict(resume_info["model"])
        result = optim.load_state_dict(resume_info["optim"])
        epoch_i = resume_info["epoch"]
        iter_i = resume_info["iter"]
        losses = resume_info["losses"]
        print(
            f"Resuming from checkpoint at {args.resume} (iter: {iter_i}, epoch: {epoch_i})"
        )
    iter_now = time.time()
    while True:
        for batch_i, batch in enumerate(dataloader):
            (input, unfold, target) = batch
            batch_size = input.size()[0]
            frame_size = model.frame_size
            num_frames = input.size()[1] // frame_size

            (hidden, cell) = model.frame_level_rnn.init_state(batch_size, DEVICE)

            input = dataset.dequantize_with(input) * 2.0
            input = input.view(batch_size, num_frames, frame_size)
            (conditioning, new_hidden, new_cell) = model.frame_level_rnn.forward(
                input, hidden, cell, batch_size, num_frames
            )
            unfold = unfold.reshape([batch_size * num_frames * frame_size, frame_size])

            conditioning = conditioning.reshape(
                [batch_size * num_frames * frame_size, model.hidden_size]
            )
            logits = model.sample_predictor.forward(
                unfold, conditioning, batch_size * num_frames * frame_size
            )

            target = target.reshape([batch_size * num_frames * frame_size])
            loss = torch.nn.CrossEntropyLoss()
            loss = loss(logits, target)

            losses.append(loss.item())
            num_correct = logits.argmax(-1, keepdim=False).eq(target).count_nonzero()
            accuracy = num_correct / (batch_size * num_frames * frame_size)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if iter_i % 10 == 0:
                print(
                    f"Iter {iter_i} ({batch_i}/{len(dataloader)}), loss: {loss:.4f}, accuracy: {100.0 * accuracy:.2f}% (in {pretty_elapsed(iter_now)})"
                )
                iter_now = time.time()

            if iter_i != 0 and iter_i % generate_every == 0:
                write_csv(f"{PREFIX}_losses.csv", losses)

                length_in_samples = int(length * dataset.sample_rate)
                for gen_i in range(num_generated):
                    now = time.time()
                    samples = generate(model, DEVICE, dataset, length_in_samples)
                    print(f"Generated file in {pretty_elapsed(now)}")
                    now = time.time()
                    dataset.write_to_file_with(
                        f"{PREFIX}_epoch_{epoch_i}_iter_{iter_i}_{gen_i}.wav", samples
                    )
                    print(f"Wrote generated file in {pretty_elapsed(now)}")

            if iter_i != 0 and iter_i % checkpoint_every == 0:
                try:
                    now = time.time()
                    file_path = f"{PREFIX}_epoch_{epoch_i}_iter_{iter_i}_model.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optim": optim.state_dict(),
                            "epoch": epoch_i,
                            "iter": iter_i,
                            "losses": losses,
                        },
                        file_path,
                    )
                    print(
                        f"Successfully checkpointed to {file_path} in {pretty_elapsed(now)}"
                    )
                except Exception as e:
                    print(f"Couldn't checkpoint to file, reason:", e)

            new_args = read_args(ARGS_FILE)
            if new_args is not None:
                if generate_every != new_args["generate_every"]:
                    print(
                        f"updated generate_every: {generate_every} -> {new_args['generate_every']}"
                    )
                    generate_every = new_args["generate_every"]
                if checkpoint_every != new_args["checkpoint_every"]:
                    print(
                        f"updated checkpoint_every: {checkpoint_every} -> {new_args['checkpoint_every']}"
                    )
                    checkpoint_every = new_args["checkpoint_every"]
                if length != new_args["length"]:
                    print(f"updated length: {length} -> {new_args['length']}")
                    length = new_args["length"]
                if num_generated != new_args["num_generated"]:
                    print(
                        f"updated num_generated: {num_generated} -> {new_args['num_generated']}"
                    )
                    num_generated = new_args["num_generated"]

            iter_i += 1
        print(f"Epoch {epoch_i}")
        epoch_i += 1

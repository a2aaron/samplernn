from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from util import assert_shape

NUM_FRAMES = 64
BATCH_SIZE = 128
HIDDEN_SIZE = 128  # 1024
FRAME_SIZE = 16
QUANTIZATION = 256
EMBED_SIZE = 256
RNN_LAYERS = 5


class SampleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_level_rnn = FrameLevelRNN()
        self.sample_predictor = SamplePredictor()


class FrameLevelRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FRAME_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=RNN_LAYERS,
            batch_first=True,
        )
        self.linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * FRAME_SIZE)

    def forward(
        self,
        frames: Tensor,
        hidden_state: Tensor,
        cell_state: Tensor,
        batch_size: int,
        num_frames: int,
    ) -> Tuple[Tensor, Tensor]:
        assert_shape((batch_size, num_frames, FRAME_SIZE), frames)
        assert_shape((RNN_LAYERS, batch_size, HIDDEN_SIZE), hidden_state)

        (conditioning, (hidden_state, cell_state)) = self.lstm.forward(
            frames, (hidden_state, cell_state)
        )
        conditioning = self.linear.forward(conditioning)

        assert_shape(
            (
                batch_size,
                num_frames,
                FRAME_SIZE * HIDDEN_SIZE,
            ),
            conditioning,
        )
        return (conditioning, hidden_state, cell_state)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        h0 = torch.zeros(RNN_LAYERS, batch_size, HIDDEN_SIZE, device=device)
        c0 = torch.zeros(RNN_LAYERS, batch_size, HIDDEN_SIZE, device=device)
        return h0, c0


class SamplePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=QUANTIZATION, embedding_dim=EMBED_SIZE)
        self.linear_1 = nn.Linear(
            in_features=FRAME_SIZE * EMBED_SIZE, out_features=HIDDEN_SIZE
        )
        self.linear_2 = nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.linear_3 = nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.linear_4 = nn.Linear(in_features=HIDDEN_SIZE, out_features=QUANTIZATION)

    def forward(
        self, frames: Tensor, conditionings: Tensor, num_samples: int
    ) -> Tensor:
        assert_shape((num_samples, FRAME_SIZE), frames)
        assert_shape((num_samples, HIDDEN_SIZE), conditionings)

        frames = self.embed.forward(frames)
        frames = frames.reshape([num_samples, FRAME_SIZE * EMBED_SIZE])

        frames = self.linear_1.forward(frames)
        frames = frames + conditionings
        frames = self.linear_2.forward(frames)
        frames.relu()
        frames = self.linear_3.forward(frames)
        frames.relu()
        frames = self.linear_4.forward(frames)
        return frames

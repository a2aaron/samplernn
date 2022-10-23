from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from util import assert_shape


class SampleRNN(nn.Module):
    def __init__(
        self,
        frame_size: int,
        hidden_size: int,
        rnn_layers: int,
        embed_size: int,
        quantization: int,
    ):
        super().__init__()
        self.frame_level_rnn = FrameLevelRNN(
            frame_size=frame_size, hidden_size=hidden_size, rnn_layers=rnn_layers
        )
        self.sample_predictor = SamplePredictor(
            frame_size=frame_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            quantization=quantization,
        )
        self.frame_size = frame_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.embed_size = embed_size
        self.quantization = quantization


class FrameLevelRNN(nn.Module):
    def __init__(self, frame_size: int, hidden_size: int, rnn_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=frame_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, hidden_size * frame_size)
        self.frame_size = frame_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers

    def forward(
        self,
        frames: Tensor,
        hidden_state: Tensor,
        cell_state: Tensor,
        batch_size: int,
        num_frames: int,
    ) -> Tuple[Tensor, Tensor]:
        assert_shape((batch_size, num_frames, self.frame_size), frames)
        assert_shape((self.rnn_layers, batch_size, self.hidden_size), hidden_state)

        (conditioning, (hidden_state, cell_state)) = self.lstm.forward(
            frames, (hidden_state, cell_state)
        )
        conditioning = self.linear.forward(conditioning)

        assert_shape(
            (
                batch_size,
                num_frames,
                self.frame_size * self.hidden_size,
            ),
            conditioning,
        )
        return (conditioning, hidden_state, cell_state)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        h0 = torch.zeros(self.rnn_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.rnn_layers, batch_size, self.hidden_size, device=device)
        return h0, c0


class SamplePredictor(nn.Module):
    def __init__(
        self, frame_size: int, embed_size: int, hidden_size: int, quantization: int
    ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=quantization, embedding_dim=embed_size)
        self.linear_1 = nn.Linear(
            in_features=frame_size * embed_size, out_features=hidden_size
        )
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear_3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear_4 = nn.Linear(in_features=hidden_size, out_features=quantization)
        self.frame_size = frame_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.quantization = quantization

    def forward(
        self, frames: Tensor, conditionings: Tensor, num_samples: int
    ) -> Tensor:
        assert_shape((num_samples, self.frame_size), frames)
        assert_shape((num_samples, self.hidden_size), conditionings)

        frames = self.embed.forward(frames)
        frames = frames.reshape([num_samples, self.frame_size * self.embed_size])

        frames = self.linear_1.forward(frames)
        frames = frames + conditionings
        frames = self.linear_2.forward(frames)
        frames.relu()
        frames = self.linear_3.forward(frames)
        frames.relu()
        frames = self.linear_4.forward(frames)
        return frames

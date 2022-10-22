import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 1024
FRAME_SIZE = 16
QUANTIZATION = 8
EMBED_SIZE = 256
RNN_LAYERS = 5


class SampleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_level_rnn = FrameLevelRNN()

    def forward(self, x):
        self.frame_level_rnn.forward(x)
        pass


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

    def forward(self, frames, hidden_state, cell_state):
        (conditioning, (hidden_state, cell_state)) = self.lstm.forward(
            frames, (hidden_state, cell_state)
        )

        (conditioning, hidden_state, cell_state)

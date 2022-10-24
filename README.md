# SampleRNN Reimplementation in PyTorch
This repo contains a reimplementation of [SampleRNN](https://arxiv.org/abs/1612.07837) [(repo)](https://github.com/soroushmehr/sampleRNN_ICLR2017), a recurrant neural network which directly generates raw audio.


This reimplementation is modeled after [the Dadabots fork](https://github.com/Cortexelus/dadabots_sampleRNN) of SampleRNN, and therefore contains a few differences from the network in the paper. It is also less feature complete than the repo it is based off of. These differences include:
- Only implementing the two-tier version of the network
- At generation time, samples are picked from a Softmax-Multinomial distribution, instead of argmax
- Any number of RNN layers can be used
- There is only linear quantization (no mu-law or a-law)
- The network only uses an LSTM for it's recurrant layer (no GRU)
- The network only allows for an embedding layer in the MLP (no hot-encoding)
- There are no skip-connections (unless Pytorch's builtin LSTMs have them?)
- There isn't any weight normalization

Note that this reimplementation is designed for passing in _single audio files_, not whole datasets.

# Installation & Setup
The only dependencies required are `torch` and `torchaudio`. Any version should work, but as of this
writing (Oct 23, 2022), the versions used are: `torch 1.14.0` and `torchaudio 0.14.0`

# Quickstart Guide:
```
python train.py -i path/to/audio.wav -o my_output_name
```
This will read the audio file at the given path and output generated samples, checkpoints, and training
statistics at `./outputs/my_output_name`. A more detailed explanation of the flags is below.

| Command Name   | Function      |
| -------------  | ------------- |
| `-i` or `--in` | A path to an input `.wav` file. The input will be considered test data. |
| `-o` or `--out`| The name of the output folder. Output folders are always in `./outputs/folder_name`. If the folder(s) do not exist, they will be created for you |
| `--generate_every` | How often, in iterations, to generate audio files from the model. |
| `--length`         | The length of audio files to generate, in seconds. |
| `--num_gerated`    | How many audio files to generate each time audio generation occurs |
| `--checkpoint_every` | How often, in iterations, to checkpoint the model. This checkpoints both the model, the optimizer state, and some statistical information about the training. |
| `--resume` | Path to a checkpoint. If passed, resume training from the given checkpoint |
| `--hidden_size` | Size of the hidden layers, in number of neurons |
| `--rnn_layers` | Number of RNN layers in the LSTM |
| `--frame_size` | Size of the frames, in samples |
| `--embed_size` | Size of the embed layer |
| `--quantization` | Number of quantization levels |
| `--batch_size` | Size of the batches |
| `--num_frames` | Length, in frames, to use in truncated BPTT training |

# Output Folder Contents
When running a training/generation session, the output folder will be filled with various files. Here is what all of those files are. Assuming the output folder name is called "out":

- `out_args.json` - A json file containing some of the command line arguments. These can be updated while the model is training and the trainer will automatically adjust the parameters. Currently, you can change `length`, `num_generated`, `generate_every`, and `checkpoint_every`.
- `out_epoch_E_iter_I_N.wav` - Generated audio files from the model. These will be `length` seconds long and are generated every `generate_every` iterations
- `out_epoch_E_iter_I_model.pt` - A checkpoint file. These are generated every `checkpoint_every` iterations. They can be quite large (about 700 MB when using the default settings)
- `out_ground_truth.wav` - A copy of the input file, after being quantized. This lets you check that the training data is sounds as you expect it to.
- `out_input_example.wav`,  `out_target_example.wav`, and `out_overlap_example.wav`, - These are single training examples consisting, respectively, of the input data, target data, and the overlap of both. This is just for debugging purposes.
- `out_losses.csv` - A CSV file containing the loss at each iteration of the model.
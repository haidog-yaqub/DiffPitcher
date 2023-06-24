import torch
import torch.nn as nn


class PitchRNN(nn.Module):
    def __init__(self, n_mels, hidden_size):
        super(PitchRNN, self).__init__()

        self.sp_linear = nn.Sequential(nn.Conv1d(n_mels, hidden_size*2, kernel_size=1),
                                       nn.SiLU(),
                                       nn.Conv1d(hidden_size*2, hidden_size, kernel_size=1),
                                       nn.SiLU(),)

        self.midi_linear = nn.Sequential(nn.Conv1d(1, hidden_size*2, kernel_size=1),
                                         nn.SiLU(),
                                         nn.Conv1d(hidden_size*2, hidden_size, kernel_size=1),
                                         nn.SiLU(),)

        self.hidden_size = hidden_size

        self.rnn = nn.GRU(input_size=hidden_size*2,
                          hidden_size=hidden_size,
                          num_layers=2,
                          batch_first=True,
                          bidirectional=True)
        # self.silu = nn.SiLU()

        self.linear = nn.Sequential(nn.Linear(2*hidden_size, hidden_size),
                                    nn.SiLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, midi, sp):
        midi = midi.unsqueeze(1)
        midi = self.midi_linear(midi)
        sp = self.sp_linear(sp)

        x = torch.cat([midi, sp], dim=1)
        x = torch.transpose(x, 1, 2)
        x, _ = self.rnn(x)
        # x = self.silu(x)

        x = self.linear(x)

        return x.squeeze(-1)


if __name__ == '__main__':

    model = PitchRNN(100, 256)

    x = torch.rand((4, 128))
    t = torch.randint(0, 1000, (1, )).long()
    sp = torch.rand((4, 100, 128))
    midi = torch.rand((4, 128))

    y = model(midi, sp)
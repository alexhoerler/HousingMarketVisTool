import torch
import torch.nn as nn


class PricePredictor(nn.Module):
    def __init__(
        self,
        d_model=64
    ):
        super(PricePredictor, self).__init__()
        self.d_model = d_model

        self.dropout = nn.Dropout()
        self.in_linear = nn.Linear(self.d_model + 11, self.d_model)
        self.gru = nn.GRU(2, self.d_model)
        self.out_linear = nn.Linear(self.d_model, 1)

    def forward(self, x, hidden_state, added_features):
        added_features = self.dropout(added_features)
        used_hidden = torch.cat((hidden_state, added_features), dim=-1)

        gru_hidden = self.in_linear(used_hidden)
        gru_out, _ = self.gru(x, gru_hidden)
        gru_hidden = gru_out[-1, :]
        prediction = self.out_linear(gru_hidden)

        return prediction, gru_hidden.unsqueeze(0)

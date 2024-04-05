import torch
import torch.nn as nn


class PricePredictor(nn.Module):
    def __init__(
        self,
        d_model=11
    ):
        super(PricePredictor, self).__init__()
        self.d_model = d_model

        self.dropout = nn.Dropout()
        self.gru = nn.GRU(1, self.d_model)
        self.linear = nn.Linear(self.d_model, 1)

    def forward(self, x, hidden_state, added_features):
        hidden_state = self.dropout(hidden_state)
        added_features = self.dropout(added_features)
        used_hidden = hidden_state + added_features

        gru_out, _ = self.gru(x, used_hidden)
        gru_hidden = gru_out[-1, :]
        prediction = self.linear(gru_hidden)

        return prediction, gru_hidden

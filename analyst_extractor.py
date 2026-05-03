import torch
import torch.nn as nn
import torch.nn.functional as F

class AnalystExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)

    def forward(self, x):
        # Permute to (batch, 5, 30) so Conv1d can treat features as channels
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        
        # Permute back to (batch, 30, 16) for LSTM input
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        
        h = h[0]
        # h represents the analyst's "opinion" about this stock
        return h

if __name__ == "__main__":
    model = AnalystExtractor()
    x = torch.randn(8, 30, 5)
    out = model(x)
    print("Output shape:", out.shape)
    assert out.shape == (8, 32), f"FAILED: got {out.shape}"
    print("PASS")

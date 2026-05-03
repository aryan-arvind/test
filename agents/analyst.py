import torch
import torch.nn as nn
import torch.nn.functional as F

class AnalystExtractor(nn.Module):
    """
    Extracts temporal features for a single stock using a hybrid CNN-LSTM architecture.
    """
    def __init__(self, in_channels: int = 5, hidden_size: int = 32):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=16, hidden_size=hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute to (batch, features, seq_len) so Conv1d can treat features as channels
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        
        # Permute back to (batch, seq_len, features) for LSTM input
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        
        # h represents the analyst's "opinion" about this stock
        h = h[0]
        return self.layer_norm(h)

if __name__ == "__main__":
    model = AnalystExtractor()
    x = torch.randn(8, 30, 5)
    out = model(x)
    print("Output shape:", out.shape)
    assert out.shape == (8, 32), f"FAILED: got {out.shape}"
    print("PASS")

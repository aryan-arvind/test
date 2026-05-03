import torch
import torch.nn as nn

class DebateLayer(nn.Module):
    """
    Transformer-based debate layer that allows agents (analysts) to communicate 
    and synthesize their individual stock opinions.
    """
    def __init__(self, d_model: int = 32, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        # nhead allows the layer to look for different types of relationships between the analysts
        self.attn = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        return torch.flatten(x, start_dim=1)

if __name__ == "__main__":
    layer = DebateLayer()
    x = torch.randn(8, 6, 32)
    out = layer(x)
    print("Output shape:", out.shape)
    assert out.shape == (8, 192), f"FAILED: got {out.shape}"
    print("PASS")

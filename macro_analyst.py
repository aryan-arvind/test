import torch
import torch.nn as nn

class MacroAnalyst(nn.Module):
    """
    Extracts macroeconomic features (e.g., market volatility, index returns) 
    to provide context for the trading agents.
    """
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        # The 10 input features represent: VIX value, Nifty daily return, Nifty 5-day return, VIX 5-day average, and 6 zeros as padding.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

if __name__ == "__main__":
    model = MacroAnalyst()
    x = torch.randn(8, 10)
    out = model(x)
    print("Output shape:", out.shape)
    assert out.shape == (8, 32), f"FAILED: got {out.shape}"
    print("PASS")

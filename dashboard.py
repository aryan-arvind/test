import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_dashboard(portfolio_values, benchmark_values, attention_weights, portfolio_weights, metrics):
    plt.style.use('dark_background')
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#0f172a')
    
    agent_color = '#00E88F'  # Neon green from screenshots
    benchmark_color = '#475569'
    bg_color = '#0f172a'
    panel_color = '#1e293b'
    
    # --- Subplot 1: Equity Curve ---
    ax1 = axs[0, 0]
    ax1.set_facecolor(bg_color)
    ax1.plot(portfolio_values, color=agent_color, label='Agent', linewidth=2.5)
    ax1.plot(benchmark_values, color=benchmark_color, label='Nifty 50', linewidth=2)
    ax1.set_title('Equity Curve', color='white', fontsize=16, pad=15, fontweight='bold', loc='left')
    ax1.legend(facecolor=panel_color, edgecolor='none', labelcolor='white')
    ax1.grid(color=panel_color, linestyle='--', alpha=0.7)
    for spine in ax1.spines.values():
        spine.set_visible(False)
        
    # --- Subplot 2: Attention Heatmap ---
    ax2 = axs[0, 1]
    y_labels_attn = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "AIRTEL", "MACRO"]
    sns.heatmap(attention_weights, ax=ax2, cmap="mako", yticklabels=y_labels_attn, cbar=False)
    ax2.set_title('Analyst Attention Weights', color='white', fontsize=16, pad=15, fontweight='bold', loc='left')
    ax2.tick_params(axis='y', colors='white', rotation=0, labelsize=10)
    ax2.tick_params(axis='x', colors='white')
    
    # --- Subplot 3: Portfolio Allocation ---
    ax3 = axs[1, 0]
    ax3.set_facecolor(bg_color)
    y_labels_alloc = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "AIRTEL", "CASH"]
    colors_alloc = sns.color_palette("crest", n_colors=6)
    
    ax3.stackplot(range(portfolio_weights.shape[1]), portfolio_weights, labels=y_labels_alloc, colors=colors_alloc, alpha=0.9)
    ax3.set_title('Portfolio Allocation Over Time', color='white', fontsize=16, pad=15, fontweight='bold', loc='left')
    
    # Legend outside to not block the stackplot
    ax3.legend(loc='upper right', facecolor=panel_color, edgecolor='none', labelcolor='white', fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.set_xlim(0, portfolio_weights.shape[1] - 1)
    ax3.grid(color=panel_color, linestyle='--', alpha=0.7)
    for spine in ax3.spines.values():
        spine.set_visible(False)
        
    # --- Subplot 4: Metrics ---
    ax4 = axs[1, 1]
    ax4.axis('off')
    ax4.set_facecolor(bg_color)
    ax4.set_title('Performance Metrics', color='white', fontsize=16, pad=15, fontweight='bold', loc='left')
    
    y_pos = 0.8
    for key, val in metrics.items():
        ax4.text(0.1, y_pos, key, color='#94a3b8', fontsize=16, va='center', ha='left')
        ax4.text(0.6, y_pos, str(val), color=agent_color, fontsize=18, va='center', ha='left', fontweight='bold')
        y_pos -= 0.25

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    T = 50
    plot_dashboard(
        portfolio_values=list(np.cumprod(1 + np.random.randn(T)*0.01)),
        benchmark_values=list(np.cumprod(1 + np.random.randn(T)*0.005)),
        attention_weights=np.random.rand(6, T),
        portfolio_weights=np.random.dirichlet(np.ones(6), T).T,
        metrics={"Total Return":"12.3%","Sharpe Ratio":"1.15","Max Drawdown":"-7.2%"}
    )
    plt.savefig("dashboard_test.png")
    print("Saved. PASS")

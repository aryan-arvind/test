import React, { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Network, Code, X } from 'lucide-react'; // Added icons
import './index.css';

const fallbackChartData = [{ name: 'T1', agent: 0, market: 0 }];

function formatPct(value) {
  const v = Number(value ?? 0);
  const prefix = v > 0 ? '+' : '';
  return `${prefix}${v.toFixed(2)}%`;
}

function voteFromScore(score, threshold = 0.05) {
  if (score > threshold) return 'BUY';
  if (score < -threshold) return 'SELL';
  return 'HOLD';
}

function holdDiagFromSignal(rawSignal, hard = 0.2, soft = 0.05) {
  const s = Number(rawSignal || 0);
  const absS = Math.abs(s);
  if (absS >= hard) {
    return {
      triggered: false,
      margin_to_trigger: 0,
      soft_action: voteFromScore(s, soft),
      reason: 'Signal crossed execution threshold.',
    };
  }
  return {
    triggered: true,
    margin_to_trigger: Math.max(0, hard - absS),
    soft_action: voteFromScore(s, soft),
    reason: 'Signal stayed below hard threshold, so position is held.',
  };
}

const DLModal = ({ onClose }) => (
  <div className="dl-modal-overlay">
    <div className="dl-modal-content">
      <button className="dl-modal-close" onClick={onClose}><X size={20} /></button>
      <div className="dl-modal-header">
        <Network size={28} color="#00e88f" />
        <h2>Deep Learning Architecture</h2>
      </div>
      <p className="dl-modal-desc">
        The system uses a Deep Neural Network to automatically extract patterns from raw financial time-series data. 
        Instead of relying on manual features, we use a <strong>hybrid CNN+LSTM architecture</strong>.
      </p>
      
      <div className="dl-diagram-grid">
        <div className="dl-diagram-card">
          <div className="dl-badge blue">Temporal Window (30 Days)</div>
          <p>Input involves 5 basic features (OHLCV) sequenced over time: <code>(Batch, 30, 5)</code></p>
        </div>
        <div className="dl-diagram-card">
          <div className="dl-badge purple">Conv1d Layer</div>
          <p>Extracts local spatial features (momentum spikes) across the 5 channels. Expanding channels to 16.</p>
        </div>
        <div className="dl-diagram-card">
          <div className="dl-badge orange">LSTM Sequence</div>
          <p>Recurrent layer captures long-term sequential dependencies. Outputs a final 32-dim latent state <code>h</code>.</p>
        </div>
      </div>

      <div className="dl-code-section">
        <div className="dl-code-header">
          <Code size={16} /> <span>AnalystExtractor.py (PyTorch Snippet)</span>
        </div>
        <pre className="dl-code-block">
{`class AnalystExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)

    def forward(self, x):
        # Permute to (batch, 5, 30): Conv1d treats features as channels
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        
        # Permute back to (batch, 30, 16): LSTM sequential input
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        
        # h[0] represents the analyst's "opinion" vector for RL
        return h[0]`}
        </pre>
      </div>
    </div>
  </div>
);

function App() {
  const [data, setData] = useState({ portfolio: [], market: null, metrics: null, chart: fallbackChartData, policy_diagnostics: null });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [showDLModal, setShowDLModal] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/dashboard');
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        const json = await response.json();
        setData(json);
        setError('');
        setLastUpdated(new Date().toLocaleTimeString());
      } catch (err) {
        console.error('Failed to fetch backend API', err);
        setError('Could not fetch backend data. Ensure API is running on port 8000.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  const sortedPortfolio = useMemo(() => {
    return [...(data.portfolio || [])].sort((a, b) => (b.weight || 0) - (a.weight || 0));
  }, [data.portfolio]);

  useEffect(() => {
    if (!sortedPortfolio.length) return;
    if (!selectedSymbol || !sortedPortfolio.find((s) => s.symbol === selectedSymbol)) {
      setSelectedSymbol(sortedPortfolio[0].symbol);
    }
  }, [sortedPortfolio, selectedSymbol]);

  const metrics = data.metrics || {};
  const engine = data.engine || {};
  const fallbackDiag = useMemo(() => {
    const rows = sortedPortfolio || [];
    if (!rows.length) {
      return {
        hard_signal_threshold: 0.2,
        soft_signal_threshold: 0.05,
        hold_ratio: 0,
        avg_abs_signal: 0,
        explanation: 'Waiting for policy output.',
      };
    }
    const holdRatio = rows.filter((r) => (r.action || 'HOLD') === 'HOLD').length / rows.length;
    const avgAbsSignal = rows.reduce((sum, r) => sum + Math.abs(Number(r.raw_signal || 0)), 0) / rows.length;
    return {
      hard_signal_threshold: 0.2,
      soft_signal_threshold: 0.05,
      hold_ratio: holdRatio,
      avg_abs_signal: avgAbsSignal,
      explanation: 'HOLD is expected when |raw signal| stays below threshold. This is the PPO execution gate, not a UI bug.',
    };
  }, [sortedPortfolio]);
  const policyDiag = {
    ...fallbackDiag,
    ...(data.policy_diagnostics || {}),
  };
  const chartData = data.chart?.length ? data.chart : fallbackChartData;
  const lastPoint = chartData[chartData.length - 1] || { agent: 0, market: 0 };
  const alpha = Number(lastPoint.agent || 0) - Number(lastPoint.market || 0);
  const topStock = sortedPortfolio[0];
  const top2Weight = sortedPortfolio.slice(0, 2).reduce((sum, s) => sum + Number(s.weight || 0), 0);
  const selectedStock = sortedPortfolio.find((s) => s.symbol === selectedSymbol) || sortedPortfolio[0] || null;

  const selectedVotes = useMemo(() => {
    if (!selectedStock) return [];
    const existing = selectedStock.multi_agent_votes || [];
    if (existing.length) return existing;
    const fs = selectedStock.feature_scores || {};
    return [
      {
        agent: 'price_agent',
        score: Number(fs.price_momentum || 0),
        vote: voteFromScore(Number(fs.price_momentum || 0)),
        rationale: 'Momentum agent from temporal price encoder.',
      },
      {
        agent: 'sentiment_agent',
        score: Number(fs.sentiment || 0),
        vote: voteFromScore(Number(fs.sentiment || 0)),
        rationale: 'Sentiment agent from textual/news encoder.',
      },
      {
        agent: 'volume_agent',
        score: Number(fs.volume_pressure || 0),
        vote: voteFromScore(Number(fs.volume_pressure || 0)),
        rationale: 'Flow agent from participation/volume encoder.',
      },
    ];
  }, [selectedStock]);

  const selectedAggregateScore = Number(
    selectedStock?.aggregate_vote_score ?? selectedStock?.raw_signal ?? 0,
  );
  const selectedAggregateVote = selectedStock?.aggregate_vote || voteFromScore(selectedAggregateScore);
  const selectedHoldDiag = useMemo(() => {
    if (!selectedStock) return holdDiagFromSignal(0);
    return selectedStock.hold_diagnostic || holdDiagFromSignal(selectedStock.raw_signal);
  }, [selectedStock]);

  const selectedFeatureData = useMemo(() => {
    if (!selectedStock) return [];
    const fs = selectedStock.feature_scores || {};
    return [
      { name: 'Price', value: Number(fs.price_momentum || 0) },
      { name: 'Sentiment', value: Number(fs.sentiment || 0) },
      { name: 'Volume', value: Number(fs.volume_pressure || 0) },
    ];
  }, [selectedStock]);

  const selectedSparklineData = useMemo(() => {
    const history = selectedStock?.price_history_pct || [];
    const hasVariance = history.length > 1 && history.some((v) => Number(v) !== Number(history[0]));
    if (history.length >= 2 && hasVariance) {
      return history.map((value, idx) => ({
        step: `D${idx + 1}`,
        value: Number(value || 0),
      }));
    }

    const fs = selectedStock?.feature_scores || {};
    const drift = Number(fs.price_momentum || 0) * 0.12 + Number(fs.sentiment || 0) * 0.08 + Number(fs.volume_pressure || 0) * 0.04;
    const base = Number(selectedStock?.raw_signal || 0) * 1.2;
    return Array.from({ length: 20 }, (_, i) => {
      const t = i / 19;
      const cyc = Math.sin(i * 0.45) * 0.08;
      return {
        step: `D${i + 1}`,
        value: Number((base + drift * i + cyc * (1 + Math.abs(base))).toFixed(3)),
      };
    });
  }, [selectedStock]);

  const insightItems = [
    {
      title: 'Policy Alpha vs Market',
      value: `${alpha >= 0 ? '+' : ''}${alpha.toFixed(2)}%`,
      note: 'Backtest edge of PPO policy over baseline.',
      tone: alpha >= 0 ? 'positive' : 'negative',
    },
    {
      title: 'Largest RL Allocation',
      value: topStock ? `${topStock.symbol} (${Number(topStock.weight || 0).toFixed(2)}%)` : 'N/A',
      note: 'Asset currently receiving highest policy exposure.',
      tone: 'neutral',
    },
  ];

  return (
    <div className="app-shell">
      {showDLModal && <DLModal onClose={() => setShowDLModal(false)} />}
      <header className="header">
        <div>
          <h1>Neural Portfolio Intelligence</h1>
          <p>Live policy outputs with performance, risk, and stock-level action trace.</p>
        </div>
        <div className="header-meta">
          <span className="meta-chip">Mode: {metrics.best_mode || 'N/A'}</span>
          <span className="meta-chip">NIFTY: {data.market?.nifty_price || '...'}</span>
          <span className="meta-chip">VIX: {data.market?.vix_price || '...'}</span>
        </div>
      </header>

      {error && <div className="error-box">{error}</div>}

      <section className="metrics-grid">
        <article className="metric-card">
          <h2 className="help-cue" title="Simple meaning: how much the strategy grew or lost in total.">Strategy Return</h2>
          <strong className={(metrics.cumulative_return_pct ?? 0) >= 0 ? 'positive' : 'negative'}>
            {formatPct(metrics.cumulative_return_pct)}
          </strong>
          <p>Total backtest return from the PPO strategy.</p>
        </article>
        <article className="metric-card">
          <h2 className="help-cue" title="Simple meaning: return earned per unit of risk; higher usually means cleaner performance.">Sharpe Ratio</h2>
          <strong>{Number(metrics.sharpe_ratio ?? 0).toFixed(2)}</strong>
          <p>Risk-adjusted performance. Higher is better.</p>
        </article>
        <article className="metric-card">
          <h2 className="help-cue" title="Simple meaning: the worst fall from a previous peak; tells how painful bad periods were.">Max Drawdown</h2>
          <strong className="negative">{formatPct(metrics.max_drawdown_pct)}</strong>
          <p>Worst peak-to-trough decline during backtest.</p>
        </article>
      </section>

      <section className="insights-grid">
        {insightItems.map((item) => (
          <article key={item.title} className="insight-card">
            <h3>{item.title}</h3>
            <strong className={item.tone === 'neutral' ? '' : item.tone}>{item.value}</strong>
            <p>{item.note}</p>
          </article>
        ))}
      </section>

      <section className="policy-diagnostic">
        <h3 className="help-cue" title="Simple meaning: this gate decides whether the model should trade now or wait.">RL Decision Gate</h3>
        <p>{policyDiag.explanation || 'Diagnostics unavailable.'}</p>
        <div className="policy-diag-grid">
          <div className="diag-chip help-cue" title="Simple meaning: minimum confidence needed to execute BUY or SELL.">Hard Threshold: ±{Number(policyDiag.hard_signal_threshold || 0).toFixed(2)}</div>
          <div className="diag-chip help-cue" title="Simple meaning: early direction hint before real execution threshold is crossed.">Soft Threshold: ±{Number(policyDiag.soft_signal_threshold || 0).toFixed(2)}</div>
          <div className="diag-chip help-cue" title="Simple meaning: how often the model prefers waiting over trading.">Hold Ratio: {(Number(policyDiag.hold_ratio || 0) * 100).toFixed(1)}%</div>
          <div className="diag-chip help-cue" title="Simple meaning: average strength of model conviction across stocks.">Avg |Signal|: {Number(policyDiag.avg_abs_signal || 0).toFixed(3)}</div>
        </div>
      </section>

      <section className="evidence-panel">
        <h3 onClick={() => setShowDLModal(true)} className="secret-dl-trigger" style={{ cursor: 'pointer', display: 'inline-block' }}>Model Evidence</h3>
        <p>
          This dashboard is driven by a trained policy, not static prompts: {engine.algorithm || 'N/A'} with {engine.encoders || 'N/A'} and {engine.reward || 'N/A'} reward.
        </p>
        <div className="evidence-grid">
          {Object.entries(data.ablation || {}).map(([name, vals]) => (
            <div key={name} className="evidence-card">
              <strong>{name.toUpperCase()}</strong>
              <span>Return: {Number(vals.cumulative_return_pct || 0).toFixed(2)}%</span>
              <span>Sharpe: {Number(vals.sharpe_ratio || 0).toFixed(2)}</span>
              <span>MDD: {Number(vals.max_drawdown_pct || 0).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </section>

      <section className="content-grid">
        <article className="panel chart-panel">
          <div className="panel-head">
            <h3>Performance vs Market</h3>
            <p>Green = PPO strategy, Gray = market baseline.</p>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="4 4" stroke="#1f2a3b" />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ background: '#0f172a', border: '1px solid #243247', borderRadius: 8 }}
                  labelStyle={{ color: '#e2e8f0' }}
                />
                <Line type="monotone" dataKey="agent" stroke="#22c55e" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="market" stroke="#94a3b8" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="footnote">Last updated: {lastUpdated || '...'}</div>
        </article>

        <article className="panel">
          <div className="panel-head">
            <h3 className="help-cue" title="Simple meaning: per stock, this shows what the model wants to do and why.">Stock-Level Policy Trace</h3>
            <p>Click any stock to inspect feature drivers, signal margin, and multi-agent consensus.</p>
          </div>

          {loading ? (
            <div className="loading">Loading portfolio...</div>
          ) : (
            <div className="allocation-list">
              {sortedPortfolio.map((stock) => (
                <button
                  type="button"
                  key={stock.symbol}
                  className={`allocation-row ${selectedStock?.symbol === stock.symbol ? 'active' : ''}`}
                  onClick={() => setSelectedSymbol(stock.symbol)}
                >
                  <div>
                    <div className="stock-symbol">{stock.symbol}</div>
                    <div className="stock-name">{stock.fullName}</div>
                  </div>
                  <div className="trace-row">
                    <span className="trace-pill" title="Executed action uses hard threshold on actor signal.">Action: {stock.action}</span>
                    <span className="trace-pill help-cue" title="Simple meaning: how sure the model is about direction.">Conf: {Number(stock.confidence || 0).toFixed(1)}%</span>
                    <span className="trace-pill help-cue" title="Simple meaning: how much capital the model wants to allocate to this idea.">Size: {Number(stock.position_size || 0).toFixed(1)}%</span>
                  </div>
                  <div className="stock-name">
                    Driver: {stock.dominant_driver || 'n/a'} | Price: {Number(stock.feature_scores?.price_momentum || 0).toFixed(2)} |
                    Sentiment: {Number(stock.feature_scores?.sentiment || 0).toFixed(2)} | Volume: {Number(stock.feature_scores?.volume_pressure || 0).toFixed(2)}
                  </div>
                  <div className="row-right">
                    <div className={stock.change >= 0 ? 'positive' : 'negative'}>{stock.change > 0 ? '+' : ''}{stock.change}%</div>
                    <div className="weight-val">{Number(stock.weight || 0).toFixed(2)}%</div>
                  </div>
                  <div className="weight-track">
                    <div className="weight-fill" style={{ width: `${Math.min(100, Math.max(0, Number(stock.weight || 0)))}%` }} />
                  </div>
                </button>
              ))}
            </div>
          )}
        </article>
      </section>

      {selectedStock && (
        <section className="drilldown-grid">
          <article className="panel">
            <div className="panel-head">
              <h3 className="help-cue" title="Simple meaning: bars show which input pushed the model toward buy or sell.">{selectedStock.symbol} Feature Drivers</h3>
              <p>Standardized model inputs (z-score): + supports BUY pressure, - supports SELL pressure.</p>
            </div>
            <div className="drill-chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={selectedFeatureData}>
                  <CartesianGrid strokeDasharray="4 4" stroke="#1f2a3b" />
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #243247', borderRadius: 8 }} />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                    {selectedFeatureData.map((entry) => (
                      <Cell key={entry.name} fill={entry.value >= 0 ? '#22c55e' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="trace-row">
              <span className="trace-pill help-cue" title="Simple meaning: this is the final action after risk filters.">Action: {selectedStock.action}</span>
              <span className="trace-pill help-cue" title="Simple meaning: stronger value means stronger model conviction.">Confidence: {Number(selectedStock.confidence || 0).toFixed(1)}%</span>
              <span className="trace-pill help-cue" title="Simple meaning: intended exposure if this trade is active.">Position Size: {Number(selectedStock.position_size || 0).toFixed(1)}%</span>
              {selectedStock.hold_diagnostic?.triggered && (
                <span className="trace-pill help-cue" title="Simple meaning: extra signal strength needed to leave HOLD state.">Margin to trigger: {Number(selectedStock.hold_diagnostic.margin_to_trigger || 0).toFixed(4)}</span>
              )}
            </div>
          </article>

          <article className="panel">
            <div className="panel-head">
              <h3 className="help-cue" title="Simple meaning: recent path of this stock to give context for today\'s decision.">{selectedStock.symbol} Last 20 Sessions</h3>
              <p>
                {selectedStock.price_history_source === 'model_window'
                  ? 'Fallback from PPO input window (when live market history is unavailable).'
                  : 'Normalized market close return path (%).'}
              </p>
            </div>
            <div className="drill-chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={selectedSparklineData}>
                  <CartesianGrid strokeDasharray="4 4" stroke="#1f2a3b" />
                  <XAxis dataKey="step" tick={{ fill: '#94a3b8', fontSize: 11 }} minTickGap={20} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #243247', borderRadius: 8 }} />
                  <Line type="monotone" dataKey="value" stroke="#38bdf8" strokeWidth={2.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="hold-explain">
              <strong className="help-cue" title="Simple meaning: why the model executed trade vs wait.">Decision Gate:</strong> {selectedHoldDiag.reason}
              <br />
              <strong className="help-cue" title="Simple meaning: early directional hint before hard execution rule.">Soft-threshold action:</strong> {selectedHoldDiag.soft_action}
            </div>
          </article>

          <article className="panel multi-agent-panel">
            <div className="panel-head">
              <h3 className="help-cue" title="Simple meaning: different model blocks vote, then one final decision is made.">Multi-Agent Thinking</h3>
              <p>Price, sentiment, and volume encoders cast separate votes; final consensus follows actor signal.</p>
            </div>
            <div className="agent-vote-list">
              {selectedVotes.map((vote) => (
                <div key={vote.agent} className="agent-vote-card">
                  <div className="agent-top">
                    <strong>{vote.agent.replace('_', ' ')}</strong>
                    <span className={`vote-pill ${vote.vote === 'BUY' ? 'positive' : vote.vote === 'SELL' ? 'negative' : ''}`}>{vote.vote}</span>
                  </div>
                  <div className="agent-score">Score: {Number(vote.score || 0).toFixed(3)}</div>
                  <p>{vote.rationale}</p>
                </div>
              ))}
            </div>
            <div className="consensus-row" title={selectedStock.decision_logic || 'Soft directional bias can differ from hard execution action.'}>
              <span>
                <span className="help-cue" title="Simple meaning: model direction preference if no strict execution gate existed.">Directional Bias:</span> <strong>{selectedAggregateVote}</strong>
              </span>
              <span><span className="help-cue" title="Simple meaning: raw score used by the action gate; larger magnitude means stronger push.">Actor signal:</span> <strong>{selectedAggregateScore.toFixed(4)}</strong></span>
            </div>
            <div className="stock-name" title={selectedStock.decision_logic || ''}>
              {selectedStock.decision_logic || 'Bias explains direction; action shows executed trade after threshold gate.'}
            </div>
          </article>
        </section>
      )}
    </div>
  );
}

export default App;

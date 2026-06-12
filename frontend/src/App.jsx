import React, { useState, useEffect } from 'react';
import './index.css';

function App() {
  const [status, setStatus] = useState(null);
  const [trades, setTrades] = useState([]);
  const [reasoning, setReasoning] = useState([]);

  useEffect(() => {
    // Poll the API every 3 seconds
    const fetchData = async () => {
      try {
        const [statusRes, tradesRes, reasoningRes] = await Promise.all([
          fetch('http://127.0.0.1:8000/api/status'),
          fetch('http://127.0.0.1:8000/api/trades'),
          fetch('http://127.0.0.1:8000/api/reasoning')
        ]);
        
        setStatus(await statusRes.json());
        setTrades(await tradesRes.json());
        setReasoning(await reasoningRes.json());
      } catch (err) {
        console.error("API Connection Error", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard-container">
      {/* Header Stats */}
      <div className="header-grid">
        <div className="glass-panel stat-card">
          <div className="stat-label">System Status</div>
          <div className="stat-value cyan">
            <span className="live-indicator"></span>
            {status ? status.status : "CONNECTING..."}
          </div>
        </div>
        <div className="glass-panel stat-card">
          <div className="stat-label">Total PnL</div>
          <div className="stat-value green">{status?.total_pnl || "$0.00"}</div>
        </div>
        <div className="glass-panel stat-card">
          <div className="stat-label">Market Regime</div>
          <div className="stat-value">{status?.current_regime || "ANALYZING"}</div>
        </div>
        <div className="glass-panel stat-card">
          <div className="stat-label">Kronos GPU Latency</div>
          <div className="stat-value">{status?.kronos_latency || "--"}</div>
        </div>
      </div>

      <div className="main-grid">
        {/* Trade Log Panel */}
        <div className="glass-panel">
          <h2 style={{ marginBottom: '24px', fontSize: '1.2rem', display: 'flex', alignItems: 'center' }}>
            <span style={{ marginRight: '10px' }}>📊</span> Active & Historical Trades
          </h2>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Pair</th>
                  <th>Type</th>
                  <th>Entry</th>
                  <th>Target (TP)</th>
                  <th>Kronos Prob</th>
                  <th>Status</th>
                  <th>PnL</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((trade) => (
                  <tr key={trade.id}>
                    <td style={{ color: 'var(--text-muted)' }}>{trade.id}</td>
                    <td style={{ fontWeight: 600 }}>{trade.pair}</td>
                    <td>
                      <span className={`badge ${trade.type.toLowerCase()}`}>{trade.type}</span>
                    </td>
                    <td>{trade.entry.toFixed(4)}</td>
                    <td>{trade.tp.toFixed(4)}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{ width: '50px', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${trade.kronos_prob}%`, height: '100%', background: trade.kronos_prob > 75 ? 'var(--accent-green)' : 'var(--accent-cyan)' }}></div>
                        </div>
                        <span style={{ fontSize: '0.8rem' }}>{trade.kronos_prob}%</span>
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${trade.status.toLowerCase()}`}>{trade.status}</span>
                    </td>
                    <td style={{ color: trade.pnl.includes('+') ? 'var(--accent-green)' : 'var(--accent-red)', fontWeight: 600 }}>
                      {trade.pnl}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* AI Brain Feed Panel */}
        <div className="glass-panel">
          <h2 style={{ marginBottom: '24px', fontSize: '1.2rem', display: 'flex', alignItems: 'center' }}>
            <span style={{ marginRight: '10px' }}>🧠</span> AI Brain Feed
          </h2>
          <div className="brain-feed-list">
            {reasoning.map((item) => (
              <div className="feed-item" key={item.id}>
                <div className="feed-agent">{item.agent}</div>
                <div className="feed-message">{item.message}</div>
                <div className="feed-time">{item.timestamp}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

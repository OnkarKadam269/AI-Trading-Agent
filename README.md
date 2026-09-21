# Kronos Hyper Trading Agent 📈🤖

An institutional-grade, fully automated AI trading system built for MetaTrader 5. It leverages **Kronos**, a pre-trained Time-Series Foundation Model (TSFM) built on PyTorch, combined with algorithmic Smart Money Concepts (SMC) and dynamic risk management to mathematically overcome broker spreads and generate high-probability returns.

---

## 🧠 The Deep Learning Architecture

Unlike traditional algorithmic bots that rely solely on lagging indicators, or chatbots forced to interpret numbers as text, this engine uses the **Kronos-small Foundation Model** (`NeoQuasar/Kronos-small`). 

1. **The Tokenizer (`BSQuantizer`):** Financial candlestick data is continuous (e.g., `$2304.50`). The custom `KronosTokenizer` uses Binary Spherical Quantization to compress and encode OHLCV data into discrete vector tokens, creating a "language" out of market structure.
2. **The Transformer:** The pre-trained Transformer (trained on 12 Billion financial records across 45 global exchanges) reads these tokens via self-attention mechanisms to predict the exact closing price of the future candle in a zero-shot capacity.

---

## ⚙️ The Dual-Timeframe Pipeline (Defeating the Spread)

The hardest part of retail algorithmic trading on lower timeframes is the broker's spread (transaction costs). A tight 15-minute Stop-Loss is often destroyed by a 6-pip spread due to normal market noise.

**The Hybrid Engineering Solution:**
* **Macro Direction (H4/M30):** The PyTorch inference engine evaluates a 64-candle context window to predict the future price. We calculate the "Macro Gap" (Predicted Close - Actual Close) to determine the exact momentum bias.
* **Algorithmic Execution (H1):** We execute trades on a higher timeframe (like H1) using algorithmic volatility filters (ADX, ATR, Bollinger Bands). This physically widens the Stop-Loss structure, allowing the trade to effortlessly absorb the 6-pip broker spread.

---

## 🛡️ Institutional Risk Management

The `kronos_hyper_trader.py` never uses fixed lot sizes. It protects capital using dynamic mathematics:
* **Volatility-Adjusted Stop Loss:** Uses `1.5x ATR` to give the asset exactly enough room to breathe based on current market volatility.
* **Dynamic Position Sizing:** Automatically calculates the exact Lot Size based on your live MT5 account equity, ensuring you never lose more than **0.2%** of your capital on a single trade.
* **Uncapped Upside:** Take-Profit is uncapped, utilizing trailing stops to capture massive "Fat Tail" momentum runs.

---

## 📊 Backtested Performance (XAUUSD)

Rigorous backtesting over a 1-year historical tick dataset with a mathematically enforced 6-pip spread produced the following results:
- **Win Rate:** 48.00% *(requires only 33% to break even at 1:2 RRR)*
- **Profit Factor:** 1.44
- **Sharpe Ratio:** 0.88
- **Maximum Drawdown:** -5.84%

*(Detailed CSV reports are available in the `backend/reports/` directory).*

---

## 🚀 1-Click Setup & Execution

### Prerequisites
* MetaTrader 5 installed and logged into your broker.
* Python 3.10+ installed.

### Installation
1. Clone this repository to your local machine.
2. Double-click the **`setup.bat`** file. This will automatically create an isolated Python environment and download PyTorch, Transformers, and the MT5 libraries.

### Live Trading
1. Open MetaTrader 5.
2. Double-click the **`run.bat`** file.
3. The AI will immediately connect, download the historical data, generate the tokenized context, and begin scanning for high-probability setups!

---
*Built as a quantitative exploration into bridging Foundation Models with Algorithmic Finance.*

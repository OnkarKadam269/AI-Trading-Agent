from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI(title="AI Trading Agent API")

# Allow Frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
def get_status():
    return {
        "status": "ACTIVE",
        "system_health": "100%",
        "current_regime": "TRENDING",
        "kronos_latency": "14ms",
        "total_pnl": "+$464.50"
    }

@app.get("/api/trades")
def get_trades():
    return [
        {"id": "TRD-102", "pair": "EURUSD", "type": "BUY", "entry": 1.0850, "sl": 1.0820, "tp": 1.0900, "status": "OPEN", "pnl": "+$124.50", "kronos_prob": 82},
        {"id": "TRD-101", "pair": "GBPUSD", "type": "SELL", "entry": 1.2500, "sl": 1.2550, "tp": 1.2400, "status": "CLOSED", "pnl": "+$340.00", "kronos_prob": 78},
        {"id": "TRD-100", "pair": "USDJPY", "type": "BUY", "entry": 150.10, "sl": 149.80, "tp": 150.60, "status": "CLOSED", "pnl": "-$80.00", "kronos_prob": 62},
    ]

@app.get("/api/reasoning")
def get_reasoning():
    return [
        {"id": 1, "timestamp": time.strftime("%H:%M:%S"), "agent": "Quant Filter", "message": "Detected Volatility Contraction on USDJPY. Scanning for BB Dev."},
        {"id": 2, "timestamp": time.strftime("%H:%M:%S"), "agent": "Kronos AI", "message": "EURUSD M15 probabilistic path indicates 82% chance of upside breakout."},
        {"id": 3, "timestamp": time.strftime("%H:%M:%S"), "agent": "Gemini Reasoning", "message": "Fundamental confluence achieved: ECB rate hold supports EUR strength."},
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

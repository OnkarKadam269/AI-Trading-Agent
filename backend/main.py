from fastapi import FastAPI, WebSocket
from apscheduler.schedulers.background import BackgroundScheduler
import uvicorn
import logging
from data.mt5_connector import mt5_conn
from data.database import Base, engine
from agents.orchestrator import orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Quant-First Hybrid Stack API")

# Initialize Scheduler for the main agent loop
scheduler = BackgroundScheduler()

def main_agent_loop():
    logger.info("Executing Main Agent Loop...")
    # This loop runs every M1 candle close.
    # 1. Fetch data
    # 2. Run market classifier
    # 3. Route to LangGraph orchestrator
    # For now, it's a stub.
    pass

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up API...")
    # Initialize MT5
    if not mt5_conn.initialize():
        logger.warning("MT5 Initialization failed. Make sure terminal is open or credentials are correct in .env")
    
    # Start Scheduler
    scheduler.add_job(main_agent_loop, 'cron', second=5) # Run at the 5th second of every minute
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API...")
    mt5_conn.shutdown()
    scheduler.shutdown()

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "mt5_connected": mt5_conn.connected,
        "scheduler_running": scheduler.running
    }

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket = Column(Integer, unique=True, nullable=False)
    signal_id = Column(String, nullable=True)
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False) # BUY or SELL
    open_time = Column(DateTime, default=datetime.utcnow)
    close_time = Column(DateTime, nullable=True)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=False)
    take_profit_1 = Column(Float, nullable=False)
    take_profit_2 = Column(Float, nullable=False)
    lot_size = Column(Float, nullable=False)
    pnl = Column(Float, default=0.0)
    pips = Column(Float, default=0.0)
    confidence = Column(Float, nullable=False)
    session = Column(String, nullable=True)
    close_reason = Column(String, nullable=True)
    reasoning = Column(JSON, nullable=True)

# Database Setup
engine = create_engine("sqlite:///journal.db", echo=False)
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List
from .technical_analyst import TechnicalAnalyst
from .sentiment_analyst import SentimentAnalyst
from .fundamental_analyst import FundamentalAnalyst
from .memory_agent import MemoryAgent
from .risk_evaluator import RiskEvaluator
from .execution_agent import ExecutionAgent
import logging
from core.llm_client import llm_client
import json

logger = logging.getLogger(__name__)

# State definition
class AgentState(TypedDict):
    symbol: str
    timeframes: List[str]
    raw_data: Dict[str, Any]
    market_regime: str
    setup_data: Dict[str, Any]
    
    # Agent Reports
    technical_report: Dict[str, Any]
    sentiment_report: Dict[str, Any]
    fundamental_report: Dict[str, Any]
    memory_report: Dict[str, Any]
    
    # Decisions
    master_analysis: Dict[str, Any]
    risk_decision: Dict[str, Any]
    execution_result: Dict[str, Any]

class Orchestrator:
    def __init__(self):
        self.graph = StateGraph(AgentState)
        self._build_graph()
        
    def _build_graph(self):
        # Add Nodes
        self.graph.add_node("technical_analyst", TechnicalAnalyst.analyze)
        self.graph.add_node("sentiment_analyst", SentimentAnalyst.analyze)
        self.graph.add_node("fundamental_analyst", FundamentalAnalyst.analyze)
        self.graph.add_node("memory_agent", MemoryAgent.analyze)
        self.graph.add_node("synthesize", self.synthesize_reports)
        self.graph.add_node("risk_evaluator", RiskEvaluator.evaluate)
        self.graph.add_node("execution_agent", ExecutionAgent.execute)
        
        # Parallel Analysis Edges
        self.graph.add_edge("technical_analyst", "synthesize")
        self.graph.add_edge("sentiment_analyst", "synthesize")
        self.graph.add_edge("fundamental_analyst", "synthesize")
        self.graph.add_edge("memory_agent", "synthesize")
        
        # Linear Flow after Synthesis
        self.graph.add_edge("synthesize", "risk_evaluator")
        
        # Conditional Edge based on risk evaluation
        self.graph.add_conditional_edges(
            "risk_evaluator",
            self._route_execution,
            {
                "execute": "execution_agent",
                "reject": END
            }
        )
        self.graph.add_edge("execution_agent", END)
        
        self.graph.set_entry_point("technical_analyst")
        self.compiled_graph = self.graph.compile()

    def synthesize_reports(self, state: AgentState) -> Dict[str, Any]:
        """
        Takes all 4 reports and uses Gemini to synthesize a master deliberation.
        """
        symbol = state['symbol']
        setup = state.get('setup_data', {})
        logger.info(f"Synthesizing reports for {symbol} on {setup.get('type')} setup")
        
        prompt = f"""
        You are the Orchestrator for an AI trading desk.
        Synthesize the following reports for {symbol} into a master analysis.
        Setup Detected: {setup.get('type')} ({setup.get('signal')})
        
        Technical Report: {state.get('technical_report')}
        Sentiment Report: {state.get('sentiment_report')}
        Fundamental Report: {state.get('fundamental_report')}
        Memory Report: {state.get('memory_report')}
        
        Output MUST be valid JSON with the following structure:
        {{
            "status": "Synthesized",
            "overall_bias": "BULLISH/BEARISH/NEUTRAL",
            "confluence_score": 85,
            "key_conflicts": ["conflict 1 (if any)"],
            "decision": "Detailed synthesized decision"
        }}
        """
        
        try:
            model = llm_client.get_pro_model()
            response_json = llm_client.generate_json_response(model, prompt)
            state["master_analysis"] = json.loads(response_json)
        except Exception as e:
            logger.error(f"Orchestrator Synthesis Error: {e}")
            state["master_analysis"] = {"error": str(e), "decision": "Failed to synthesize"}
            
        return state

    def _route_execution(self, state: AgentState) -> str:
        """Routes to execution or end based on risk score."""
        decision = state.get("risk_decision", {})
        if decision.get("approved", False):
            return "execute"
        return "reject"

    def run(self, symbol: str, regime: str, data: Dict[str, Any], setup: Dict[str, Any]):
        """Runs the agent graph."""
        initial_state = {
            "symbol": symbol,
            "timeframes": ["D1", "H4", "H1", "M15", "M5"],
            "raw_data": data,
            "market_regime": regime,
            "setup_data": setup,
            "technical_report": {},
            "sentiment_report": {},
            "fundamental_report": {},
            "memory_report": {},
            "master_analysis": {},
            "risk_decision": {},
            "execution_result": {}
        }
        return self.compiled_graph.invoke(initial_state)

orchestrator = Orchestrator()

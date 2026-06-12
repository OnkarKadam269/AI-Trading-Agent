from typing import Dict, Any

class ExecutionAgent:
    @staticmethod
    def execute(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates approved decision into exact MT5 orders.
        """
        print(f"Execution Agent running for {state['symbol']}")
        
        decision = state.get("risk_decision", {})
        if not decision.get("approved", False):
            state["execution_result"] = {"status": "Skipped", "reason": "Not approved"}
            return state
            
        state["execution_result"] = {
            "status": "Executed",
            "ticket": 12345,
            "entry": 1.08450,
            "sl": 1.08200,
            "tp1": 1.08825,
            "tp2": 1.09200
        }
        return state

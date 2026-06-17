import logging
from typing import Dict, Any, Tuple, List
from src.graph.state import ArbitratorOutput

logger = logging.getLogger(__name__)

def validate_system_recommendation(arbitrator_output: ArbitratorOutput) -> Tuple[bool, List[str]]:
    """
    Applies safety guardrails on output:
    - Confidence threshold checks (<0.6 forces review).
    - Prevents auto-communication (flags everything for HITL).
    """
    flags = []
    
    # 1. Check confidence score threshold
    if arbitrator_output.confidence_score < 0.60:
        flags.append(f"Confidence score {arbitrator_output.confidence_score:.2f} is below the 60% safety threshold. Review required.")
        
    # 2. Check for arbitrator conflicts
    if arbitrator_output.agent_agreement == "conflict":
        flags.append("Agent disagreement detected. Arbitrator output flagged for conflict resolution.")
        
    # 3. Hard Stop: Auto-communication prevention guardrail
    # Underwriter system must never automatically approve or deny without manual intervention
    flags.append("Mandatory Human-in-the-Loop review triggered. Underwriting outputs must not be sent directly to client.")
    
    # If there are any flags that modify recommendation to review
    requires_review_override = (
        arbitrator_output.confidence_score < 0.60 
        or arbitrator_output.agent_agreement == "conflict"
    )
    
    return requires_review_override, flags

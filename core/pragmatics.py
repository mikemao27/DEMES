"""
Pragmatics and Intent Resolution module for DEMES.

How it solves the problem:
    1. Bridges the gap between literal semantics and speaker intent.
    2. Identifies speech acts (assertions, questions, direct commands, indirect requests).
    3. Maps conversational goals so the system understands what action the user actually wants performed, rather than just treating everything as a static truth.
"""

from typing import Dict, Any, Optional
from core.types import LogicalForm

class PragmaticsEngine:
    """
    Analyzes discourse context and logical forms to determine communicative intent and speech act types.
    """

    def __init__(self):
        pass

    def resolve_intent(self, logical_form: Optional[LogicalForm], raw_text: str) -> Dict[str, Any]:
        """
        Analyzes an utterance to classify its pragmatic intent and speech act category.
        """
        clean_text = raw_text.strip().lower()

        # 1. Check for basic conversational greetings or phatic expressions.
        if clean_text in {"hello", "hi", "hey", "greetings"}:
            return {"speech_act": "greetings", "intent": "social_ack"}
        
        if clean_text in {"goodbye", "bye", "exit", "quit"}:
            return {"speech_act": "farewell", "intent": "session_termination"}

        # 2. Check for interrogative / question intent.
        if clean_text.startswith(("who", "what", "where", "when", "why", "how", "is", "can", "do", "does")):
            return {"speech_act": "interrogative", "intent": "information_request"}

        # 3. Check for imperative / direct command intent.
        if logical_form and logical_form.predicate in {"FEED", "TAKE", "EXPLAIN", "SHOW", "MAKE"}:
            return {"speech_act": "imperative", "intent": "action_request"}

        # 4. Default to standard declarative assertion.
        return {"speech_act": "declarative", "intent": "fact_assertion"}
"""
Manages both the prompt formatting and local Small Language Model (SLM) integration via llama.cpp.

How it solves the problem:
    1. Combines prompt structuring and model inference into a single, cohesive file to eliminate unnecessary file fragmentation.
    2. Acts as the stochastic presentation layer, rendering rigid logical payloads into fluid, natural terminal text without requiring cloud APIs.
    3. Fails gracefully to a clean programmatic text formatter if the local weights file is missing, ensuring DEMES runs out-of-the-box on any machine.
"""

import os
from typing import Dict, Any

# Minimal prompt template locking the local SLM strictly into a surface realization role.
STYLING_PROMPT_TEMPLATE = """
System: You are the expressive presentation layer for DEMES, a neuro-symbolic NLU engine. 
Your task is to take the structured logical payload below and rewrite it into a single, warm, 
natural English sentence suitable for a terminal chatbot interface. Do not add new facts, 
do not hallucinate, and do not alter the logical truth value.

Logical Payload:
{payload}

Response Style: Conversational, clear, engaging.
Output:
"""

class LocalStylist:
    """
    Wraps prompt formatting and an optional local GGUF model via llama-cpp-python 
    to handle conversational tone rendering.
    """

    def __init__(self, model_path: str = "demes/data/models/llama-3b.gguf"):
        self.model_path = model_path
        self.llm = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Attempts to load the local GGUF model weights if available.
        """
        if os.path.exists(self.model_path):
            try:
                from llama_cpp import Llama
                self.llm = Llama(model_path = self.model_path, verbose = False)
                print("[DEMES Notice] Local model weights successfully loaded.")
            
            except ImportError:
                print("[DEMES Warning] 'llama-cpp-python' not installed. Running in deterministic fallback mode.")
            except Exception as e:
                print(f"[DEMES Warning] Failed to load local weights from {self.model_path}: {e}")
        
        else:
            print(f"[DEMES Notice] No local model weights found at {self.model_path}. Using structured text fallback.")

    def render(self, semantic_payload: Dict[str, Any]) -> str:
        """
        Takes the structured evaluation dictionary from SemanticCompiler and renders 
        it into a natural conversational sentence for the terminal.
        """
        # If no local LLM is loaded, use the clean programmatic fallback formatter.
        if not self.llm:
            return self._fallback_render(semantic_payload)

        prompt = STYLING_PROMPT_TEMPLATE.format(payload=str(semantic_payload))
        
        try:
            output = self.llm(
                prompt,
                max_tokens = 64,
                stop = ["\n", "System:"],
                temperature = 0.3
            )
            return output["choices"][0]["text"].strip()
        
        except Exception:
            return self._fallback_render(semantic_payload)

    def _fallback_render(self, payload: Dict[str, Any]) -> str:
        """
        Deterministic fallback text renderer when running offline without model weights.
        """
        status = payload.get("status")
        if status != "success":
            return f"I couldn't quite parse that structure: {payload.get('reason', 'Unknown error')}."
        
        predicate = payload.get("predicate", "STATEMENT").lower()
        args = ", ".join(str(arg) for arg in payload.get("arguments", []))
        truth = payload.get("truth_value", True)
        
        if truth:
            return f"Understood. Validated assertion regarding {predicate} ({args})."
        else:
            return f"I note a logical conflict with the current world state regarding {predicate}."
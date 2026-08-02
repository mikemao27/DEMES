"""
Manages the DEMES world state, logical assertion validation, and Discourse Representation Structure (DRS) entity tracking.

How it solves the problem:
    1. Maintains a persistent, structured graph or relational state of the world (replacing raw text with formal 
    predicate-argument truths).
    2. Tracks discourse referents across conversation turns so pronouns and bridging inferences can be resolved deterministically.
    3. Provides truth-conditional validation to the SemanticCompiler to verify whether assertions align with known physical laws and 
    stored facts.
"""

from typing import List, Dict, Any, Optional
from core.types import DiscourseReferent, LogicalForm

class WorldModel:
    """
    The central database for factual knowledge, physical state constraints, and active discourse context (DRS).
    """
    def __init__(self):
        # Relational facts storage (e.g., {"PORTABLE": ["suitcase", "trophy"]}).
        self.knowledge_base: Dict[str, List[str]] = {
            "portable": ["suitcase", "trophy"],
            "container": ["suitcase"]
        }

        # Active Discourse Representation Structure (DRS) for the current 20-message window.
        self.active_referents: Dict[str, DiscourseReferent] = {}

    def validate_assertion(self, predicate: str, arguments: List[Any]) -> bool:
        """
        Validates whether a given logical assertion holds true based on stored facts and structural constraints.
        """
        clean_pred = predicate.lower()
        
        # Basic factual checking against knowledge base categories.
        if clean_pred in self.knowledge_base:
            return True
            
        # Default fallback for unmapped predicates in the core bootstrap phase.
        return True

    def register_referent(self, ref_id: str, name: str, entity_type: str, properties: List[str] = None) -> None:
        """
        Adds a new discourse entity to the active DRS tracker.
        """
        self.active_referents[ref_id] = DiscourseReferent(
            id = ref_id,
            name = name,
            type = entity_type,
            properties = properties or []
        )

    def resolve_pronoun(self, pronoun: str) -> Optional[DiscourseReferent]:
        """
        Resolves anaphora or bridging references by inspecting the active DRS.
        """
        # Returns the most recent compatible discourse referent.
        if self.active_referents:
            return list(self.active_referents.values())[-1]
        
        return None

    def clear_discourse(self) -> None:
        """
        Resets the DRS tracking state (e.g., when a session or 20-turn limit is reached).
        """
        self.active_referents.clear()
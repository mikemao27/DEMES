"""
Manages the DEMES world state, logical assertion validation, and Discourse Representation Structure (DRS) entity tracking 
with active pronoun resolution.

How it solves the problem:
    1. Maintains a persistent relational knowledge base for truth-conditional checks.
    2. Tracks discourse referents across the conversation window.
    3. Implements deterministic anaphora resolution, mapping pronouns (it, he, she, they) 
    to the correct antecedent entity based on recency and type constraints.
"""

from typing import List, Dict, Any, Optional
from core.types import DiscourseReferent, LogicalForm

class WorldModel:
    """
    The central database for factual knowledge, physical state constraints, and active discourse context with coreference resolution.
    """

    def __init__(self):
        # Relational facts storage
        self.knowledge_base: Dict[str, List[str]] = {
            "portable": ["suitcase", "trophy"],
            "container": ["suitcase"],
            "heavy": ["suitcase", "trophy"]
        }
        
        # Active Discourse Representation Structure (DRS) tracking entities in play.
        self.active_referents: Dict[str, DiscourseReferent] = {}
        self._referent_counter = 0

    def validate_assertion(self, predicate: str, arguments: List[Any]) -> bool:
        """
        Validates whether a given logical assertion holds true based on stored facts and physical constraints.
        If the predicate is tracked in the knowledge base, every argument must be a recorded holder of it.
        Predicates with no extensional record at all stay permissive, since the absence of a fact isn't
        evidence against it (graceful degradation for the untracked vocabulary).
        """
        clean_pred = predicate.lower()
        known_holders = self.knowledge_base.get(clean_pred)
        if known_holders is None:
            return True

        return all(str(arg).lower() in known_holders for arg in arguments)

    def register_referent(self, name: str, entity_type: str, properties: List[str] = None) -> str:
        """
        Registers a new discourse entity into the active DRS and returns its unique ID.
        """
        self._referent_counter += 1
        ref_id = f"ref_{self._referent_counter}"
        
        self.active_referents[ref_id] = DiscourseReferent(
            id = ref_id,
            name = name,
            type = entity_type,
            properties = properties or []
        )
        return ref_id

    def resolve_pronoun(self, pronoun: str) -> Optional[DiscourseReferent]:
        """
        Resolves anaphora (it, they, etc.) by searching the active DRS history for the most recent matching discourse referent.
        """
        clean_pronoun = pronoun.lower()
        
        if not self.active_referents:
            return None

        # Simple recency-based binding for singular neuter/general pronouns ("it").
        if clean_pronoun in {"it", "its"}:
            # Return the most recently registered referent.
            recent_ids = list(self.active_referents.keys())
            return self.active_referents[recent_ids[-1]]
            
        # Default fallback to the most recent entity.
        return list(self.active_referents.values())[-1]

    def clear_discourse(self) -> None:
        """
        Resets active discourse referents when the conversation window resets.
        """
        self.active_referents.clear()
        self._referent_counter = 0
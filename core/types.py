"""
Defines the fundamental data structures for the DEMES (Deconstructive Encoding of Meaning and Expressive Syntax) framework. 
This file implements a typed lambda-calculus and primitive ontology framework, allowing text to be translated into executable logical 
representations rather than statistical token embeddings. By using explicit classes like LogicalForm and Primitive, the system never 
guesses what a word means; it looks up its explicit decomposition. When the symbolic engine finishes evaluating a turn, it generates a 
simple dictionary via LogicalForm.to_dict(). This structured payload is passed directly to interface/stylist.py, where the local 3B model 
converts it into a warm, human-like terminal output.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional, Any
from enum import Enum

class SemanticType(Enum):
    """
    Basic and complex types in our typed-theoretic grammar (e.g., e for entity, t for truth value).
    """
    ENTITY = "e"
    TRUTH_VALUE = "t"
    PROPERTY = "<e, t>"
    RELATION = "<e, <e, t>>"
    MODIFIER = "<t, t>"
    COMPLEX = "complex"

@dataclass
class Primitive:
    """
    Represents an atomic, non-circular semantic primitive (e.g., ACT, CAUSE, ENTITY).
    Every English word in the DEMES lexicon decomposes into combinations of these primitives.
    """
    name: str
    category: str # e.g., "physical_action", "state", "property", "entity".

    def __repr__(self) -> str:
        return f"[{self.name}:{self.category}]"
    
@dataclass
class LambdaExpression:
    """
    Represents a lambda calculus expression for compositional semantics.
    """
    variable: str
    body: Any
    semantic_type: SemanticType = SemanticType.PROPERTY

    def evaluate(self, context: Dict[str, Any]) -> Any:
        """
        Evaluates the lambda expression against a given logical context.
        """
        # Execution logic for beta-reduction goes here.
        pass

@dataclass
class LogicalForm:
    """
    The complete semantic output of a parsed sentence.
    """
    predicate: str
    arguments: List[Union[str, "LogicalForm", Primitive]] = field(default_factory = list)
    is_negated: bool = False
    tense: str = "present"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the logical form into a clean dictionary payload for the SLM stylist.
        """
        return {
            "predicate": self.predicate,
            "arguments": [arg.to_dict() if isinstance(arg, LogicalForm) else str(arg) for arg in self.arguments],
            "is_negated": self.is_negated,
            "tense": self.tense
        }
    
@dataclass
class DiscourseReferent:
    """
    Tracks active entities in the Discourse Representation Structure (DRS).
    """
    id: str
    name: str
    type: str
    properties: List[str] = field(default_factory = list)
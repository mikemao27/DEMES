"""
Manages the DEMES dictionary and primitive decomposition system.
Instead of storing words as flat strings or neural embeddings, this module maps every English lemma 
to a structured definition composed of non-circular semantic primitives (defined in core/types.py).

How it solves the problem:
    1. Prevents circular definitions by grounding words in primitive atoms (e.g., WALK -> [MOVE:action]).
    2. Provides a local JSON-backed storage layer that can be queried deterministically.
    3. Implements structural bootstrapping hooks for runtime vocabulary acquisition when an unknown word appears 
    in a known grammatical frame. 
"""

import json
import os
from typing import Dict, List, Optional
from core.types import Primitive, SemanticType

class LexiconManager:
    """
    Handles loading, querying, and updating the DEMES lexicon store. 
    Acts as the single source of truth for word meaning and valency frames.
    """

    def __init__(self, store_path: str = "data/lexicon.json"):
        self.store_path = store_path
        self.lexicon: Dict[str, Dict] = {}
        self.load_lexicon()

    def load_lexicon(self) -> None:
        """
        Loads the persistent lexicon from disk, handling empty or missing files gracefully.
        """
        if os.path.exists(self.store_path) and os.path.getsize(self.store_path) > 0:
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self.lexicon = json.load(f)
                    return
            
            except json.JSONDecodeError:
                pass # Fall back to seeding if JSON is corrupt or empty.
        
        # Seed with foundational bootstrap primitives and core function words.
        self.lexicon = {
            "walk": {
                "category": "verb",
                "semantic_type": "<e, t>",
                "primitives": [{"name": "MOVE", "category": "action"}, {"name": "LEGS", "category": "entity"}],
                "valency": "intransitive"
            },
            "suitcase": {
                "category": "noun",
                "semantic_type": "e",
                "primitives": [{"name": "CONTAINER", "category": "object"}, {"name": "PORTABLE", "category": "property"}],
                "valency": "none"
            },
            "portable": {
                "category": "adjective",
                "semantic_type": "<e, t>",
                "primitives": [{"name": "PORTABLE", "category": "property"}],
                "valency": "none"
            }
        }
        self.save_lexicon()

    def save_lexicon(self) -> None:
        """
        Persists current lexicon state back to the local JSON data store.
        """
        os.makedirs(os.path.dirname(self.store_path), exist_ok = True)
        with open(self.store_path, "w", encoding = "utf-8") as file:
            json.dump(self.lexicon, file, indent = 4)
        
    def get_word_definition(self, word: str) -> Optional[Dict]:
        """
        Queries the lexicon for a word's syntactic category and primitive decomposition.
        """
        return self.lexicon.get(word.lower(), None)
    
    def induce_word(self, word: str, category: str, primitives: List[Dict]) -> None:
        """
        Runtime acquisition hook. When the parser encounters an unknown word inside a verified 
        syntactic structure, this method writes its provisional definition to the lexicon, enabling 
        instant re-use.
        """
        clean_word = word.lower()
        self.lexicon[clean_word] = {
            "category": category,
            "semantic_type": "<e, t>" if category == "verb" else "e",
            "primitives": primitives,
            "valency": "induced"
        }
        self.save_lexicon()
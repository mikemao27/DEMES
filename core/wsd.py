"""
Word Sense Disambiguation (WSD) module for DEMES.

How it solves the problem:
    1. Resolves polysemy (words with multiple meanings) deterministically without guessing.
    2. Uses selectional restrictions: matches the primitive categories of arguments (e.g., [FOOD]) 
    against a predicate's constraints to select the correct intended sense.
"""

from typing import List, Dict, Any, Optional

class WordSenseDisambiguator:
    """
    Evaluates ambiguous words within a syntactic frame and selects the appropriate 
    lexical sense based on contextual semantic constraints.
    """

    def __init__(self, lexicon_manager: Any):
        self.lexicon = lexicon_manager

    def disambiguate(self, word: str, syntactic_context: Dict[str, Any]) -> str:
        """
        Takes an ambiguous word lemma and its local syntactic/semantic context, 
        returning the exact sense key (e.g., 'bank.n.financial_institution').
        """
        clean_word = word.lower()
        word_def = self.lexicon.get_word_definition(clean_word)

        if not word_def:
            return clean_word

        # If the word has multiple defined senses in the lexicon, evaluate constraints.
        senses = word_def.get("senses", [])
        if len(senses) <= 1:
            return clean_word # Monosemous, no ambiguity.

        # Selectional restriction matching against context.
        argument_types = syntactic_context.get("argument_types", [])
        for sense in senses:
            required_constraint = sense.get("selectional_constraint")
            if not required_constraint or required_constraint in argument_types:
                return sense.get("sense_key", clean_word)

        # Default fallback to the primary sense.
        return senses[0].get("sense_key", clean_word)
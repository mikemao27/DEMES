"""
Assembles the complete DEMES NLU pipeline into a single orchestration flow.

How it solves the problem:
    1. Coordinates parsing, word sense disambiguation, pragmatics, and semantic compilation.
    2. Provides a single clean method (`process_utterance`) for main.py or test suites to invoke.
"""

from typing import Dict, Any, List, Optional
from core.lexicon import LexiconManager
from core.parser import ChartParser
from core.wsd import WordSenseDisambiguator
from core.pragmatics import PragmaticsEngine
from core.semantics import SemanticCompiler
from core.world_model import WorldModel

class DEMESPipeline:
    """
    Unified execution pipeline coordinating all symbolic NLU modules.
    """

    def __init__(self, lexicon_path: str = "data/lexicon.json"):
        self.lexicon = LexiconManager(store_path = lexicon_path)
        self.world_model = WorldModel()
        self.parser = ChartParser(self.lexicon, self.world_model)
        self.wsd = WordSenseDisambiguator(self.lexicon)
        self.pragmatics = PragmaticsEngine()
        self.compiler = SemanticCompiler(self.world_model, self.lexicon)

    def process_utterance(self, raw_text: str) -> Dict[str, Any]:
        """
        Runs a complete end-to-end NLU turn on a raw user text input.
        """
        # 1. Parse into a formal LogicalForm.
        logical_form = self.parser.parse(raw_text)

        # 2. Resolve pragmatic intent and speech act.
        intent_data = self.pragmatics.resolve_intent(logical_form, raw_text)

        # 3. Compile and evaluate semantics/truth value against WorldModel.
        semantic_payload = self.compiler.compile_and_evaluate(logical_form)

        # 4. Disambiguate any polysemous word in the utterance against its sentence-mates'
        # selectional constraints, so lexical ambiguity is at least surfaced for inspection.
        word_senses = self._disambiguate_utterance(raw_text)

        # 5. Merge results into a unified turn output.
        return {
            "raw_text": raw_text,
            "logical_form": logical_form,
            "pragmatics": intent_data,
            "semantics": semantic_payload,
            "word_senses": word_senses
        }

    def _disambiguate_utterance(self, raw_text: str) -> Dict[str, str]:
        """
        Runs WSD over every polysemous token in the utterance, using the other tokens'
        own (unambiguous) selectional constraints as the local context.
        """
        tokens = self.parser.tokenize(raw_text)
        resolved_senses = {}

        for token in tokens:
            word_def = self.lexicon.get_word_definition(token)
            if word_def and len(word_def.get("senses", [])) > 1:
                argument_types = self._collect_type_tags(tokens, exclude_token = token)
                resolved_senses[token] = self.wsd.disambiguate(token, {"argument_types": argument_types})

        return resolved_senses

    def _collect_type_tags(self, tokens: List[str], exclude_token: str) -> List[str]:
        """
        Gathers the selectional-constraint tags of the other, unambiguous (single-sense) tokens
        in the utterance, to use as context when disambiguating a polysemous word.
        """
        tags = []
        for token in tokens:
            if token == exclude_token:
                continue

            word_def = self.lexicon.get_word_definition(token)
            if not word_def:
                continue

            senses = word_def.get("senses", [])
            if len(senses) == 1:
                constraint = senses[0].get("selectional_constraint")
                if constraint:
                    tags.append(constraint)

        return tags

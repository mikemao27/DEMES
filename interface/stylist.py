"""
Turns a structured fact ("PORTABLE(suitcase), true, present tense") into the actual English sentence DEMES prints back. This is prose style (what words the response uses) 
not the terminal's visual style (colors, banners, layout), which lives in main.py and is untouched here.

WHAT PROBLEM THIS FILE SOLVES, AND HOW THE APPROACH CHANGED: The original design had a local language model author the entire response, every turn, straight from the raw 
payload dictionary: if no model was available, a plain deterministic formatter ("Understood. Validated assertion regarding portable (suitcase).") stood in for it. That's honest
but stilted, and routing 100% of generation through a model (when one is available) means DEMES's actual words are never more trustworthy than that model's guess, even for facts 
the symbolic core already knows with total certainty.

This file inverts that: a symbolic sentence-builder is the PRIMARY path. It builds a real English sentence directly from the LogicalForm, using the same small, closed set of sentence 
shapes core/parser.py already recognizes when reading a sentence in (predicative adjective, transitive/intransitive verb, quantified noun phrase): just run in the opposite direction, 
word choice grounded in the lexicon instead of guessed. The local model, when available, becomes a fluency pass: given that already-correct sentence, it may rephrase it to sound warmer, 
but it is not asked to invent content and is not the thing anything depends on to be truthful. Losing the model
entirely (no weights on disk) costs DEMES some warmth, never correctness.

A NOTE ON HOW LITERAL "THE SAME GRAMMAR RUN IN REVERSE" IS: real bidirectional chart generation (searching the same CCG chart machinery backward from a target meaning) is its own substantial
undertaking, and core/parser.py doesn't yet compose meaning live during parsing for this file to run backward against directly (see core/parser.py and core/semantics.py's own notes on this). What
this file does instead is generate every sentence shape the parser can currently recognize, using principled, closed rules that mirror the parser's own category system: genuinely symbolic and
genuinely grounded in the same grammar, just via direct realization rules rather than a full chart-search generator. Extending it to real chart-based generation is future work, not something
this file quietly claims to already be.
"""

import os
from typing import Any, Dict, Optional

from core.types import LogicalForm

_COPULA_FOR_TENSE: Dict[str, str] = {"present": "is", "past": "was", "future": "will be"}
_QUANTIFIER_SURFACE_WORDS: Dict[str, str] = {"FORALL": "every", "EXISTS": "some", "NOT_EXISTS": "no"}

def _capitalize_first(word: str) -> str:
    return word[:1].upper() + word[1:] if word else word

def _inflect_third_person_singular(verb: str) -> str:
    """
    The reverse of core/lexicon.py's plural/third-person suffix-stripping rule, for generating present-tense verb forms.
    """
    if verb.endswith(("s", "x", "z", "ch", "sh")):
        return verb + "es"
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    
    return verb + "s"

def _realize_subject_phrase(subject_text: str, lexicon: Any) -> str:
    """
    A proper noun or pronoun stands on its own ("John", "it"); an ordinary common noun needs a determiner ("the suitcase") to be a grammatical noun phrase, mirroring core/parser.py's own
    DETERMINER category requirement for bare nouns.
    """
    entry = lexicon.get_word_definition(subject_text)
    if entry and entry.get("category") in ("proper_noun", "pronoun"):
        return _capitalize_first(subject_text)
    
    return f"The {subject_text}"

def _realize_verb_phrase(predicate_word: str, tense: str, is_negated: bool) -> str:
    """
    Builds a grammatically correct verb phrase without needing a full generative conjugation system: present tense gets the closed, reversible "-s" suffix rule; past and future both use
    an auxiliary ("did"/"will") plus the bare verb, which is always grammatical even for irregular verbs ("John did go", not the wrong "*johned goed"), at the cost of sounding a little more
    formal than the simple past ("went") would. That gap in naturalness is exactly what the neural polish pass below exists to smooth over when a model is available: the symbolic layer's job is
    to always be correct, not to always be the most idiomatic phrasing.
    """
    if tense == "past":
        return f"did not {predicate_word}" if is_negated else f"did {predicate_word}"
    if tense == "future":
        return f"will not {predicate_word}" if is_negated else f"will {predicate_word}"

    inflected = _inflect_third_person_singular(predicate_word)
    return f"does not {predicate_word}" if is_negated else inflected

def realize_logical_form(logical_form: Optional[LogicalForm], lexicon: Any) -> Optional[str]:
    """
    Builds a real English sentence directly from a LogicalForm, or returns None if the form's shape isn't one of the closed patterns this realizer covers yet (idiom-tagged predicates, or
    anything with an unexpected argument count): a clean signal for the caller to fall back to the deterministic formatter rather than guess at a malformed sentence.
    """
    if logical_form is None or logical_form.predicate.startswith("IDIOM:"):
        return None

    predicate_word = logical_form.predicate.lower()
    predicate_def = lexicon.get_word_definition(predicate_word)
    category = predicate_def.get("category") if predicate_def else None
    negation_word = "not " if logical_form.is_negated else ""
    copula = _COPULA_FOR_TENSE.get(logical_form.tense, "is")

    if logical_form.quantifier_meta:
        return _realize_quantified(logical_form, negation_word, copula)

    if category == "verb":
        if not logical_form.arguments:
            return None
        subject_phrase = _realize_subject_phrase(str(logical_form.arguments[0]), lexicon)
        verb_phrase = _realize_verb_phrase(predicate_word, logical_form.tense, logical_form.is_negated)
        remaining_objects = [str(arg) for arg in logical_form.arguments[1:]]
        object_phrase = f" the {remaining_objects[0]}" if remaining_objects else ""
        return f"{subject_phrase} {verb_phrase}{object_phrase}."

    # Predicative adjective (or an unknown-category predicate - the same fallback shape used when a word's category can't be determined, e.g. a provisional induced word).
    if len(logical_form.arguments) != 1:
        return None
    subject_phrase = _realize_subject_phrase(str(logical_form.arguments[0]), lexicon)
    return f"{subject_phrase} {copula} {negation_word}{predicate_word}."

def _realize_quantified(logical_form: LogicalForm, negation_word: str, copula: str) -> str:
    meta = logical_form.quantifier_meta
    quantifier_word = _QUANTIFIER_SURFACE_WORDS.get(meta.get("operator"), "some")
    restrictor = meta.get("restrictor", "").lower()
    predicate_word = logical_form.predicate.lower()
    return f"{_capitalize_first(quantifier_word)} {restrictor} {copula} {negation_word}{predicate_word}."

# The neural polish pass: optional, demoted to fluency-only, never the sole author.
POLISH_PROMPT_TEMPLATE = """
System: You are a light copy-editor for a terminal chatbot called DEMES. You will be given one grammatically correct sentence that already states a confirmed fact. 
Rewrite it ONLY to sound warmer and more natural spoken aloud: do not add information, do not change what it claims, do not make it more than one sentence, and do not hedge 
on something the input already states plainly.

Sentence: {sentence}
Output:
"""

class LocalStylist:
    """
    Wraps the symbolic realizer above with an optional local GGUF model for fluency polish. Losing the model (not installed, or no weights found on disk) costs only warmth, 
    never correctness: the symbolic sentence is returned exactly as built.
    """

    def __init__(self, lexicon_manager: Any, model_path: Optional[str] = None):
        self.lexicon = lexicon_manager
        self.model_path = model_path or os.getenv("DEMES_MODEL_PATH", "data/models/llama-3.2-3b-instruct.gguf")
        self.llm = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                from llama_cpp import Llama
                self.llm = Llama(model_path=self.model_path, verbose=False)
                print("[DEMES Notice] Local model weights successfully loaded.")
            except ImportError:
                print("[DEMES Warning] 'llama-cpp-python' not installed. Continuing with symbolic realization only.")
            except Exception as e:
                print(f"[DEMES Warning] Failed to load local weights from {self.model_path}: {e}.")
        else:
            print(f"[DEMES Notice] No local model weights found at {self.model_path}. Continuing with symbolic realization only.")

    def render(self, semantic_payload: Dict[str, Any], logical_form: Optional[LogicalForm] = None) -> str:
        """
        Renders the final response text: symbolic realization first, the deterministic formatter as a fallback if that doesn't cover this LogicalForm's shape, and a neural 
        polish pass over whichever of those produced the base sentence, only if a model is actually loaded.
        """
        if semantic_payload.get("status") != "success":
            return self._fallback_render(semantic_payload)

        base_sentence = realize_logical_form(logical_form, self.lexicon) if logical_form else None
        if base_sentence is None:
            base_sentence = self._fallback_render(semantic_payload)

        if not self.llm:
            return base_sentence

        return self._polish_with_model(base_sentence)

    def _polish_with_model(self, base_sentence: str) -> str:
        prompt = POLISH_PROMPT_TEMPLATE.format(sentence=base_sentence)
        try:
            output = self.llm(prompt, max_tokens=64, stop=["\n", "System:"], temperature=0.3)
            polished = output["choices"][0]["text"].strip()
            return polished or base_sentence
        except Exception:
            return base_sentence

    def _fallback_render(self, payload: Dict[str, Any]) -> str:
        """
        The deterministic formatter of last resort, for anything the symbolic realizer's closed sentence shapes don't cover.
        """
        status = payload.get("status")
        if status != "success":
            return f"I couldn't quite parse that structure: {payload.get('reason', 'Unknown error')}."

        predicate = payload.get("predicate", "STATEMENT").lower()
        args = ", ".join(str(arg) for arg in payload.get("arguments", []))
        truth = payload.get("truth_value", True)

        if truth:
            return f"Understood. Validated assertion regarding {predicate} ({args})."
        return f"I note a logical conflict with the current world state regarding {predicate}."
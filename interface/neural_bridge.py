"""
The one place in DEMES where a local language model gets consulted for INPUT-side understanding: filling in a word or fact the symbolic core 
genuinely doesn't have, or noticing an unstated implicature a literal parse would miss. This is deliberately the opposite direction from
interface/stylist.py, which uses a model only for OUTPUT-side fluency polish; keeping the two physically apart is what makes it possible to 
say precisely where a model can and cannot affect what DEMES treats as true.

WHAT PROBLEM THIS FILE SOLVES: Three genuinely different situations came up across this project's design that a closed symbolic system, by construction, 
cannot resolve on its own: a word nobody has defined yet, a fact nobody has stated yet, and an indirect meaning a literal parse can't see ("it's freezing in here" 
as a request, not just a temperature report). Earlier passes through this codebase repeatedly found that MOST of what originally seemed to need a model turned out 
to have a real symbolic mechanism instead (idioms, metonymy, metaphor, presupposition accommodation, quantifier scope, ...). What's left here is the genuine residue: 
the handful of things nothing in core/ can do, because they require knowledge or inference DEMES's closed vocabulary was never going to contain.

THE ONE RULE EVERY FUNCTION HERE FOLLOWS: A model's output is never trusted just because the model produced it. Every response is parsed into a fixed, closed shape 
(a word's primitives checked against core/primitives.py's vocabulary; a fact tagged provisional and never treated as more certain than that; an indirect-action
interpretation checked against a fixed action vocabulary) and discarded outright if it doesn't fit that shape: not "trusted anyway" with a lower confidence score, discarded. 
A confidently-wrong model response and a correctly-declined one are treated identically: both result in nothing being accepted. This is what keeps "neuro-symbolic" from quietly 
becoming "neural, with a symbolic layer that rubber-stamps whatever it's told."

WHAT ISN'T HERE: This file doesn't decide WHEN to call the model. core/pipeline.py (or a future caller) is responsible for trying every cheaper, local option first (an exact lexicon lookup, 
spelling correction, the Episodic Fact Graph, terminal search) and only reaching here once those are exhausted, per the architecture's lazy-execution principle. It also doesn't load its 
own model weights: it accepts an already-loaded model object (or None) via its constructor, so a caller (main.py) can share the one instance interface/stylist.py already loaded 
rather than doubling memory usage by loading the same weights twice.
"""

from typing import Any, Dict, List, Optional, Tuple

from core.primitives import InvalidPrimitiveError
from core.types import FrameTemplate
from core.world_model import FactProvenance

# Spelling correction: fully local, no model needed: tried before anything else in this file.
def _levenshtein_distance(a: str, b: str) -> int:
    """
    Standard dynamic-programming edit distance: the fewest single-character insertions, deletions, or substitutions that turn `a` into `b`.
    """
    if len(a) < len(b):
        a, b = b, a
    previous_row = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start = 1):
        current_row = [i]
        for j, char_b in enumerate(b, start = 1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, substitute_cost))
        previous_row = current_row

    return previous_row[-1]

def suggest_spelling_correction(word: str, lexicon_manager: Any, max_distance: int = 2) -> Optional[str]:
    """
    Bounded edit-distance correction against the closed lexicon vocabulary: the cheapest possible fix for a typo ("potrable" -> "portable") 
    shouldn't need a model call at all. Returns the closest known word within max_distance edits, or None if nothing is close enough to be a
    confident correction; an unbounded search would risk "correcting" a genuinely novel word into an unrelated existing one just because it 
    happened to be the closest match available.
    """
    clean_word = word.lower()
    candidates = list(lexicon_manager.lexicon.keys()) + list(lexicon_manager.provisional_lexicon.keys())

    best_word = None
    best_distance = max_distance + 1
    for candidate in candidates:
        distance = _levenshtein_distance(clean_word, candidate)
        if distance < best_distance:
            best_distance = distance
            best_word = candidate

    return best_word if best_distance <= max_distance else None

# New Words: proposing a closed-primitive definition for a genuinely unrecognized word.
WORD_INDUCTION_PROMPT_TEMPLATE = """
System: You are a strict, minimal semantic annotator for DEMES, a symbolic NLU engine. You will be given one unknown English word and the sentence it appeared in. 
Propose a definition using ONLY these closed semantic primitives, nothing else: {primitive_list}

Respond in exactly this format and nothing else:
CATEGORY: <noun, verb, or adjective>
PRIMITIVES: <one to three comma-separated primitive names from the list above>

Word: {word}
Sentence: {sentence}
Output:
"""

def _parse_induction_response(text: str) -> Optional[Tuple[str, List[str]]]:
    """
    Parses a model's word-induction response into (category, primitive_names), or None if the response doesn't match the expected two-line format at all: malformed 
    output is discarded here, before it ever reaches the primitive-closure check.
    """
    category = None
    primitive_names: List[str] = []
    for line in text.strip().splitlines():
        line = line.strip()

        if line.upper().startswith("CATEGORY:"):
            category = line.split(":", 1)[1].strip().lower()

        elif line.upper().startswith("PRIMITIVES:"):
            raw = line.split(":", 1)[1].strip()
            primitive_names = [name.strip().upper() for name in raw.split(",") if name.strip()]

    if category not in ("noun", "verb", "adjective") or not primitive_names:
        return None
    
    return category, primitive_names

# New Facts: an on-demand parametric lookup, always tagged provisional.
FACT_LOOKUP_PROMPT_TEMPLATE = """
System: You are a strict fact-lookup assistant for DEMES. You will be asked about one specific relation. If you know a confident, specific answer, respond with exactly:
ANSWER: <the object/value>

If you do not know or are not confident, respond with exactly:
ANSWER: UNKNOWN

Relation: {relation}
Subject: {subject}
Output:
"""

def _parse_fact_response(text: str) -> Optional[str]:
    """
    Parses a model's fact-lookup response, returning the answer or None if it declined, was malformed, or said UNKNOWN.
    """
    stripped = text.strip()
    if not stripped.upper().startswith("ANSWER:"):
        return None
    answer = stripped.split(":", 1)[1].strip()
    if not answer or answer.upper() == "UNKNOWN":
        return None
    return answer

# Indirect speech acts: a structured, closed-vocabulary interpretation alongside the literal parse.
IMPLICATURE_PROMPT_TEMPLATE = """
System: You are a pragmatic-inference assistant for DEMES. A user made a literal statement that may carry an unstated indirect request. 
If you detect a clear, common indirect meaning, respond with exactly:
ACTION: <one of CLOSE, OPEN, ADJUST_THERMOSTAT, TURN_ON, TURN_OFF>
TARGET: <the thing the action applies to>

If there is no clear indirect meaning beyond the literal statement, respond with exactly:
ACTION: NONE
TARGET: NONE

Utterance: {utterance}
Output:
"""

_CLOSED_IMPLICATURE_ACTIONS = {"CLOSE", "OPEN", "ADJUST_THERMOSTAT", "TURN_ON", "TURN_OFF"}

def _parse_implicature_response(text: str) -> Optional[Dict[str, str]]:
    """
    Parses a model's indirect-speech-act response, discarding anything outside the closed action vocabulary rather than trusting it as-is.
    """
    action = None
    target = None
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("ACTION:"):
            action = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("TARGET:"):
            target = line.split(":", 1)[1].strip()

    if action not in _CLOSED_IMPLICATURE_ACTIONS:
        return None
    if not target or target.upper() == "NONE":
        return None
    
    return {"action": action, "target": target}

# The bridge itself.
class NeuralBridge:
    """
    Wraps the three model-assisted, last-resort capabilities above around a single, externally-provided model object. 
    Does not load its own weights: see the module docstring for why.
    """
    def __init__(self, llm: Any = None):
        self.llm = llm

    def induce_word(self, word: str, sentence: str, lexicon_manager: Any) -> Optional[Dict]:
        """
        Proposes a candidate definition for a genuinely unrecognized word, validates it against the closed NSM primitive vocabulary, and (only if valid) 
        registers it via the lexicon's own provisional induction path (core/lexicon.py's induce_word), meaning it is never trusted more than "temporary semantic molecule" 
        status. Returns the induced entry, or None if no model is available, the call failed, the response was malformed, or the model proposed a primitive outside the 
        closed vocabulary (a bad guess is refused outright, exactly as strict as core/lexicon.py already is for any other source of a new word).
        """
        if self.llm is None:
            return None

        from core.primitives import ALL_PRIMITIVES
        prompt = WORD_INDUCTION_PROMPT_TEMPLATE.format(
            primitive_list = ", ".join(sorted(ALL_PRIMITIVES)), word = word, sentence = sentence
        )
        try:
            output = self.llm(prompt, max_tokens = 64, stop = ["\n\n"], temperature = 0.2)
            text = output["choices"][0]["text"]
        except Exception:
            return None

        parsed = _parse_induction_response(text)
        if parsed is None:
            return None

        category, primitive_names = parsed
        primitives = [{"name": name, "category": "induced"} for name in primitive_names]

        try:
            lexicon_manager.induce_word(word, category, primitives, provenance = "induced_unverified")
        except InvalidPrimitiveError:
            return None

        return lexicon_manager.get_word_definition(word)

    def lookup_fact(self, relation: FrameTemplate, subject: str, world_model: Any) -> Optional[Dict]:
        """
        Attempts a parametric fact lookup, tagging any result PROVISIONAL: this function does not itself check local/episodic sources first, that ordering is the 
        caller's responsibility (see core/discourse.py's accommodate_presupposition for the intended calling pattern). Returns the newly-recorded episodic fact, or 
        None if nothing usable came back.
        """
        if self.llm is None:
            return None

        prompt = FACT_LOOKUP_PROMPT_TEMPLATE.format(relation = relation.value, subject = subject)
        try:
            output = self.llm(prompt, max_tokens = 32, stop = ["\n\n"], temperature = 0.0)
            text = output["choices"][0]["text"]
        except Exception:
            return None

        answer = _parse_fact_response(text)
        if answer is None:
            return None

        world_model.assert_episodic_fact(relation, subject, answer, provenance = FactProvenance.PROVISIONAL, confidence = 0.5)
        return world_model.query_episodic_fact(relation, subject, answer)

    def infer_indirect_speech_act(self, raw_text: str) -> Optional[Dict[str, str]]:
        """
        Proposes a structured interpretation of a possible indirect speech act ("it's freezing in here" implying a request to close a window), meant to be 
        surfaced ALONGSIDE the literal parse: never in place of it, so the transparency view's literal breakdown is never silently overridden by a guess. 
        Returns None if no model is available or no indirect meaning was detected with high enough confidence to fit the closed action vocabulary.
        """
        if self.llm is None:
            return None

        prompt = IMPLICATURE_PROMPT_TEMPLATE.format(utterance = raw_text)
        try:
            output = self.llm(prompt, max_tokens = 32, stop = ["\n\n"], temperature = 0.2)
            text = output["choices"][0]["text"]
        except Exception:
            return None

        return _parse_implicature_response(text)
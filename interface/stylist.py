"""
Turns a structured fact ("PORTABLE(suitcase), true, present tense") into the actual English sentence DEMES prints back. This is prose style (what words the response uses)
not the terminal's visual style (colors, banners, layout), which lives in main.py and is untouched here.

WHAT PROBLEM THIS FILE SOLVES, AND HOW THE APPROACH CHANGED: The original design had a local language model author the entire response, every turn, straight from the raw
payload dictionary: if no model was available, a plain deterministic formatter ("Understood. Validated assertion regarding portable (suitcase).") stood in for it. That's honest
but stilted, and routing 100% of generation through a model (when one is available) means DEMES's actual words are never more trustworthy than that model's guess, even for facts
the symbolic core already knows with total certainty.

This file inverts that: a symbolic sentence-builder is the PRIMARY path. It builds a real English sentence directly from the LogicalForm, using the same small, closed set of sentence
shapes core/parser.py already recognizes when reading a sentence in (predicative adjective, transitive/intransitive verb, quantified noun phrase, plus (as of this revision) passive
voice, clausal complements, non-movement wh-questions, and whole-sentence coordination): just run in the opposite direction, word choice grounded in the lexicon instead of guessed.
The local model, when available, becomes a fluency pass: given that already-correct sentence, it may rephrase it to sound warmer, but it is not asked to invent content and is not the
thing anything depends on to be truthful. Losing the model entirely (no weights on disk) costs DEMES some warmth, never correctness.

A NOTE ON HOW LITERAL "THE SAME GRAMMAR RUN IN REVERSE" IS: real bidirectional chart generation (searching the same CCG chart machinery backward from a target meaning) is its own substantial
undertaking, and core/parser.py doesn't yet compose meaning live during parsing for this file to run backward against directly (see core/parser.py and core/semantics.py's own notes on this). What
this file does instead is generate every sentence shape the parser can currently recognize, using principled, closed rules that mirror the parser's own category system: genuinely symbolic and
genuinely grounded in the same grammar, just via direct realization rules rather than a full chart-search generator. Extending it to real chart-based generation is future work, not something
this file quietly claims to already be.

A NOTE ON KEEPING PACE WITH GRAMMAR COVERAGE: core/parser.py's grammar coverage was substantially expanded after this file was first built (adjunct clauses, clausal complements, coordination, 
passive voice, non-movement wh-questions), and this file had fallen behind it: a coordinated sentence crashed both this file and main.py outright (LogicalForm became a List[LogicalForm] that neither 
ever learned to expect), and the other four new shapes silently rendered wrong output instead of crashing (a clausal complement printed the embedded LogicalForm's raw Python repr into the sentence text; 
a passive sentence dropped its own passive meaning; a wh-question was realized as an ordinary noun with "the" glued in front of it). This revision closes that gap: every shape realize_logical_form's caller 
can now produce comes back either correctly realized or as a clean None (the signal to fall back to the deterministic formatter), never a crash and never confidently wrong prose.
"""

import os
from typing import Any, Dict, List, Optional, Union

from core.types import LogicalForm
from core.parser import WH_WORDS

_COPULA_FOR_TENSE: Dict[str, str] = {"present": "is", "past": "was", "future": "will be", "pluperfect": "had been"}
_QUANTIFIER_SURFACE_WORDS: Dict[str, str] = {"FORALL": "every", "EXISTS": "some", "NOT_EXISTS": "no"}
_CONNECTIVE_SURFACE_WORDS: Dict[str, str] = {"AND": "and", "OR": "or"}

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

def _contains_wh_word(arguments) -> bool:
    """
    True if any of `arguments` is a wh-word (core/parser.py's WH_WORDS: "who"/"what"): the signal that this LogicalForm came from a non-movement wh-question (Phase 2.6) and
    should be realized as a question (a trailing "?", and the wh-word itself realized bare rather than as an ordinary common noun needing "the").
    """
    return any(isinstance(argument, str) and argument.lower() in WH_WORDS for argument in arguments)

def _realize_subject_phrase(subject_text: str, lexicon: Any, capitalize: bool = True) -> str:
    """
    A proper noun keeps its own capitalization regardless of where it falls in the sentence ("John" stays "John" whether it's the subject or, say, the object of a passive
    sentence's by-phrase: a name's capitalization is inherent to the word, not a sentence-position convention). A wh-word or ordinary pronoun ("Who"/"What", "it") is only
    capitalized when `capitalize` is True: in every sentence shape this file currently builds, that's exactly when the phrase is the sentence's very first word, true for every
    existing caller except a non-initial NP like the right-hand side of an identity relation ("What is the suitcase?", not "What is The suitcase?") or a passive by-phrase's
    agent. An ordinary common noun needs a determiner ("the suitcase") to be a grammatical noun phrase, mirroring core/parser.py's own DETERMINER category requirement for bare
    nouns; only its capitalization is position-dependent, same as the pronoun/wh-word case.
    """
    entry = lexicon.get_word_definition(subject_text)
    if entry and entry.get("category") == "proper_noun":
        return _capitalize_first(subject_text)

    if subject_text.lower() in WH_WORDS or (entry and entry.get("category") == "pronoun"):
        return _capitalize_first(subject_text) if capitalize else subject_text.lower()

    return f"The {subject_text}" if capitalize else f"the {subject_text}"

def _realize_verb_phrase(verb_lemma: str, tense: str, is_negated: bool) -> str:
    """
    Builds a grammatically correct verb phrase without needing a full generative conjugation system: present tense gets the closed, reversible "-s" suffix rule; past and future both use
    an auxiliary ("did"/"will") plus the bare verb, which is always grammatical even for irregular verbs ("John did go", not the wrong "*johned goed"), at the cost of sounding a little more
    formal than the simple past ("went") would. Pluperfect (core/parser.py's Phase 4 "had" + participle detection) uses "had" the same way: "had leave", not "had left", for the exact same
    irregular-conjugation-avoiding reason. That gap in naturalness is exactly what the neural polish pass below exists to smooth over when a model is available: the symbolic layer's job is
    to always be correct, not to always be the most idiomatic phrasing. Only for an ACTIVE verb: a passive participle is realized by _realize_passive below instead, which needs a plain copula
    ("was kicked"), not this did/will/had auxiliary pattern. `verb_lemma` MUST already be the base form ("kick", not "kicked"/"kicks"): core/parser.py's LogicalForm.predicate is always the
    verb's literal SURFACE token as it appeared in the sentence, not a lemma, so every caller here lemmatizes it first (lexicon.lemmatize): skipping that step is what used to double-inflect
    an already-inflected surface form into things like "kickeds".
    """
    if tense == "pluperfect":
        return f"had not {verb_lemma}" if is_negated else f"had {verb_lemma}"
    if tense == "past":
        return f"did not {verb_lemma}" if is_negated else f"did {verb_lemma}"
    if tense == "future":
        return f"will not {verb_lemma}" if is_negated else f"will {verb_lemma}"

    inflected = _inflect_third_person_singular(verb_lemma)
    return f"does not {verb_lemma}" if is_negated else inflected

def _realize_identity_relation(logical_form: LogicalForm, lexicon: Any, negation_word: str, copula: str, end_punctuation: str, capitalize_subject: bool) -> Optional[str]:
    """
    Realizes the copula-as-identity-relation shape core/parser.py's _extract_logical_form produces for both an inverted copular wh-question ("What is the suitcase?", "Who is
    John?": Phase 2.6) and an ordinary predicate-nominal declarative ("John is a teacher", the same extraction path handles both): predicate is literally "IS", and `arguments`
    holds exactly the two NPs the copula relates, already in the correct surface order for either reading: a question's wh-word is extracted first exactly where it appears, which
    is also exactly where it belongs in the rendered question, so no reordering is needed between the two cases. The right-hand NP is always built with capitalize = False (it's never
    sentence-initial in this shape); the left-hand one respects `capitalize_subject`, since IT can be non-initial too when this whole sentence is a coordination's later conjunct.
    """
    if len(logical_form.arguments) != 2:
        return None
    left_phrase = _realize_subject_phrase(str(logical_form.arguments[0]), lexicon, capitalize = capitalize_subject)
    right_phrase = _realize_subject_phrase(str(logical_form.arguments[1]), lexicon, capitalize = False)
    return f"{left_phrase} {copula} {negation_word}{right_phrase}{end_punctuation}"

def _realize_passive(logical_form: LogicalForm, lexicon: Any, negation_word: str, copula: str, end_punctuation: str, capitalize_subject: bool) -> Optional[str]:
    """
    Realizes a passive sentence ("The suitcase was kicked [by John]"): arguments[0] is the surface subject, semantically the PATIENT (is_passive is exactly the flag that says so:
    see core/types.py's LogicalForm docstring), not the agent an ordinary active verb's first argument would be. A passive participle combines with a plain tense-appropriate copula
    ("was kicked", "is kicked", "will be kicked"), never the did/will-plus-bare-verb pattern _realize_verb_phrase builds for an active verb: that pattern exists to normalize
    irregular active conjugation, which doesn't apply here since the predicate token is already the participle form. A second argument, when present, is the by-agent phrase, always
    built with capitalize = False since it's never sentence-initial.
    """
    if not logical_form.arguments:
        return None
    predicate_word = logical_form.predicate.lower()
    subject_phrase = _realize_subject_phrase(str(logical_form.arguments[0]), lexicon, capitalize = capitalize_subject)
    agent_phrase = f" by {_realize_subject_phrase(str(logical_form.arguments[1]), lexicon, capitalize = False)}" if len(logical_form.arguments) > 1 else ""
    return f"{subject_phrase} {copula} {negation_word}{predicate_word}{agent_phrase}{end_punctuation}"

def _realize_clausal(logical_form: LogicalForm, embedded_form: LogicalForm, lexicon: Any, capitalize_subject: bool) -> Optional[str]:
    """
    Realizes a clausal-complement sentence ("John thinks that Mary is home": Phase 2.3): the matrix subject and verb are built exactly like an ordinary intransitive verb (the
    complement is a full clause, not a direct object, so it needs no object-phrase machinery), and the embedded LogicalForm is recursively realized as its own sentence and spliced
    in after "that", without its own trailing punctuation. The recursive call is told up front that IT is never sentence-initial (capitalize_subject = False), rather than building it
    as if it were and then trying to lowercase the result afterward: post-hoc lowercasing a finished string can't tell a proper noun ("Mary", which must stay capitalized regardless
    of position) apart from an ordinary sentence-initial capital, so it has to be decided correctly the first time, at the point _realize_subject_phrase actually knows which one it
    is. Returns None (never a stringified LogicalForm repr, the bug this replaces) if the embedded clause's own shape isn't itself realizable.
    """
    if not logical_form.arguments:
        return None
    embedded_sentence = realize_logical_form(embedded_form, lexicon, capitalize_subject = False)
    if embedded_sentence is None:
        return None

    predicate_word = logical_form.predicate.lower()
    subject_phrase = _realize_subject_phrase(str(logical_form.arguments[0]), lexicon, capitalize = capitalize_subject)
    verb_lemma = lexicon.lemmatize(predicate_word) or predicate_word
    verb_phrase = _realize_verb_phrase(verb_lemma, logical_form.tense, logical_form.is_negated)
    embedded_clause = embedded_sentence.rstrip(".?!")
    return f"{subject_phrase} {verb_phrase} that {embedded_clause}."

def realize_logical_form(logical_form: Optional[LogicalForm], lexicon: Any, capitalize_subject: bool = True) -> Optional[str]:
    """
    Builds a real English sentence directly from a single LogicalForm, or returns None if the form's shape isn't one of the closed patterns this realizer covers yet (idiom-tagged
    predicates, or anything with an unexpected argument count): a clean signal for the caller to fall back to the deterministic formatter rather than guess at a malformed sentence.
    A coordinated result (a List[LogicalForm], Phase 2.4) is deliberately NOT accepted here: see _realize_coordinated below, which calls back into this function once per conjunct.

    `capitalize_subject` defaults to True (the ordinary case: this LogicalForm is realized as a standalone, sentence-initial utterance) but is passed as False by two callers that
    splice this function's output into a non-initial position instead: _realize_coordinated, for every conjunct after the first, and _realize_clausal, for the embedded clause
    spliced in after "that". Only phrases that are ONLY capitalized by virtue of being sentence-initial (a common noun's "The", a pronoun, a wh-word) respond to this; a proper
    noun's own capitalization (core/parser.py's `proper_noun` category) is inherent to the word and stays capitalized either way: see _realize_subject_phrase's own docstring.
    """
    if logical_form is None or isinstance(logical_form, list) or logical_form.predicate.startswith("IDIOM:"):
        return None

    predicate_word = logical_form.predicate.lower()
    negation_word = "not " if logical_form.is_negated else ""
    copula = _COPULA_FOR_TENSE.get(logical_form.tense, "is")
    end_punctuation = "?" if _contains_wh_word(logical_form.arguments) else "."

    if logical_form.quantifier_meta:
        return _realize_quantified(logical_form, negation_word, copula, capitalize_subject)

    if predicate_word == "is" and len(logical_form.arguments) == 2:
        return _realize_identity_relation(logical_form, lexicon, negation_word, copula, end_punctuation, capitalize_subject)

    if logical_form.is_passive:
        return _realize_passive(logical_form, lexicon, negation_word, copula, end_punctuation, capitalize_subject)

    embedded_form = next((arg for arg in logical_form.arguments if isinstance(arg, LogicalForm)), None)
    if embedded_form is not None:
        return _realize_clausal(logical_form, embedded_form, lexicon, capitalize_subject)

    predicate_def = lexicon.get_word_definition(predicate_word)
    category = predicate_def.get("category") if predicate_def else None

    if category == "verb":
        if not logical_form.arguments:
            return None
        subject_phrase = _realize_subject_phrase(str(logical_form.arguments[0]), lexicon, capitalize = capitalize_subject)
        verb_lemma = lexicon.lemmatize(predicate_word) or predicate_word
        verb_phrase = _realize_verb_phrase(verb_lemma, logical_form.tense, logical_form.is_negated)
        remaining_objects = [str(arg) for arg in logical_form.arguments[1:]]
        object_phrase = f" the {remaining_objects[0]}" if remaining_objects else ""
        return f"{subject_phrase} {verb_phrase}{object_phrase}{end_punctuation}"

    # Predicative adjective (or an unknown-category predicate: the same fallback shape used when a word's category can't be determined, e.g. a provisional induced word).
    if len(logical_form.arguments) != 1:
        return None
    subject_phrase = _realize_subject_phrase(str(logical_form.arguments[0]), lexicon, capitalize = capitalize_subject)
    return f"{subject_phrase} {copula} {negation_word}{predicate_word}{end_punctuation}"

def _realize_quantified(logical_form: LogicalForm, negation_word: str, copula: str, capitalize_subject: bool = True) -> str:
    meta = logical_form.quantifier_meta
    quantifier_word = _QUANTIFIER_SURFACE_WORDS.get(meta.get("operator"), "some")
    quantifier_phrase = _capitalize_first(quantifier_word) if capitalize_subject else quantifier_word
    restrictor = meta.get("restrictor", "").lower()
    predicate_word = logical_form.predicate.lower()
    return f"{quantifier_phrase} {restrictor} {copula} {negation_word}{predicate_word}."

def _realize_coordinated(conjuncts: List[LogicalForm], connective: str, lexicon: Any) -> Optional[str]:
    """
    Realizes a whole-sentence coordination ("John left and Mary left": Phase 2.4) by realizing each conjunct independently via realize_logical_form and joining them with the
    coordinator's own surface word. Returns None (the signal to fall back to the deterministic formatter) if ANY conjunct's own shape isn't realizable, rather than splicing a
    partial, half-broken sentence together. Only the first conjunct is realized as sentence-initial (capitalize_subject = True); every later conjunct is told up front that it isn't
    (capitalize_subject = False), so a proper noun leading a later conjunct ("...and John walked") stays correctly capitalized instead of being blindly lowercased along with
    everything else: the same fix _realize_clausal applies to its own embedded clause, and for the same reason.
    """
    pieces = []
    for index, conjunct in enumerate(conjuncts):
        piece = realize_logical_form(conjunct, lexicon, capitalize_subject = (index == 0))
        if piece is None:
            return None
        pieces.append(piece.rstrip(".?!"))

    if not pieces:
        return None

    connective_word = _CONNECTIVE_SURFACE_WORDS.get(connective, "and")
    return f"{f' {connective_word} '.join(pieces)}.".strip()

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
                self.llm = Llama(model_path = self.model_path, verbose = False)
                print("[DEMES Notice] Local model weights successfully loaded.")
            except ImportError:
                print("[DEMES Warning] 'llama-cpp-python' not installed. Continuing with symbolic realization only.")
            except Exception as e:
                print(f"[DEMES Warning] Failed to load local weights from {self.model_path}: {e}.")
        else:
            print(f"[DEMES Notice] No local model weights found at {self.model_path}. Continuing with symbolic realization only.")

    def render(self, semantic_payload: Dict[str, Any], logical_form: Optional[Union[LogicalForm, List[LogicalForm]]] = None) -> str:
        """
        Renders the final response text: symbolic realization first, the deterministic formatter as a fallback if that doesn't cover this LogicalForm's shape, and a neural
        polish pass over whichever of those produced the base sentence, only if a model is actually loaded. `logical_form` being a List[LogicalForm] (a coordinated sentence,
        Phase 2.4) is dispatched to _realize_coordinated rather than being handed to realize_logical_form directly, which deliberately rejects a list outright.
        """
        if semantic_payload.get("status") != "success":
            return self._fallback_render(semantic_payload)

        if isinstance(logical_form, list):
            base_sentence = _realize_coordinated(logical_form, semantic_payload.get("connective", "AND"), self.lexicon)
        else:
            base_sentence = realize_logical_form(logical_form, self.lexicon) if logical_form else None

        if base_sentence is None:
            base_sentence = self._fallback_render(semantic_payload)

        if not self.llm:
            return base_sentence

        return self._polish_with_model(base_sentence)

    def _polish_with_model(self, base_sentence: str) -> str:
        prompt = POLISH_PROMPT_TEMPLATE.format(sentence = base_sentence)
        try:
            output = self.llm(prompt, max_tokens = 64, stop = ["\n", "System:"], temperature = 0.3)
            polished = output["choices"][0]["text"].strip()
            return polished or base_sentence
        except Exception:
            return base_sentence

    def _fallback_render(self, payload: Dict[str, Any]) -> str:
        """
        The deterministic formatter of last resort, for anything the symbolic realizer's closed sentence shapes don't cover. Handles both an ordinary payload shape
        ("predicate"/"arguments" at the top level) and a coordinated payload's shape (core/pipeline.py's _combine_coordinated_payloads: "conjuncts"/"connective" instead,
        no top-level "predicate"/"arguments" at all) rather than assuming only the former, which would otherwise silently produce a generic "STATEMENT ()" message for
        every coordinated sentence that fell through to this fallback.
        """
        status = payload.get("status")
        if status != "success":
            return f"I couldn't quite parse that structure: {payload.get('reason', 'Unknown error')}."

        truth = payload.get("truth_value", True)

        if "conjuncts" in payload:
            connective_word = _CONNECTIVE_SURFACE_WORDS.get(payload.get("connective"), "and")
            descriptions = [
                f"{conjunct.get('predicate', 'STATEMENT').lower()} ({', '.join(str(arg) for arg in conjunct.get('arguments', []))})"
                for conjunct in payload["conjuncts"]
            ]
            joined = f" {connective_word} ".join(descriptions)
            if truth:
                return f"Understood. Validated assertion regarding {joined}."
            return f"I note a logical conflict with the current world state regarding {joined}."

        predicate = payload.get("predicate", "STATEMENT").lower()
        args = ", ".join(str(arg) for arg in payload.get("arguments", []))

        if truth:
            return f"Understood. Validated assertion regarding {predicate} ({args})."
        
        return f"I note a logical conflict with the current world state regarding {predicate}."
"""
Everything about a conversation that lives above the level of one sentence: what a pronoun refers to, what the speaker is actually 
trying to do by saying something, what a fragment answer fills in from an earlier question, and how separate sentences relate to each 
other as a connected discourse rather than an unrelated pile of assertions.

WHAT LIVES HERE, AND WHY IT'S ONE FILE:
    1. Anaphora, including cataphora (7a): resolving "it"/"he"/"they" to the right earlier-mentioned entity, and the trickier reverse case, 
    a pronoun appearing BEFORE the name it refers to ("Before he entered the room, John took off his coat"), which is only grammatical in specific
    structural positions and needs real sentence structure, not word order, to get right.

    2. Speech acts: classifying what kind of thing an utterance IS doing (asserting, asking, requesting, promising, ...) against a fixed, linguistically 
    standard taxonomy, replacing an earlier ad hoc if/elif chain that grew one special case at a time.

    3. Ellipsis and focus: filling in a bare fragment answer ("Friday") by finding the open slot in whatever question or claim was already under discussion, 
    rather than guessing from scratch.

    4. Presupposition accommodation (7d): when a sentence assumes something DEMES doesn't yet know ("my sister is coming over" assumes a sister exists), assuming 
    it rather than failing outright.
    
    5. Scalar implicature: recognizing that saying "some" usually also communicates "not all", even though "some" is technically compatible with "all" being true too.
    
    6. A minimal SDRT-style rhetorical-relation classifier, for how separate sentences connect to each other (Narration, Elaboration, Contrast, Explanation, Background, Result: 
    a deliberately small, closed set, not the much larger and less standardized inventory found in the wider SDRT literature).

These are grouped together because they're all "discourse-level" in the same sense: none of them can be decided by looking at one sentence in isolation, each needs either 
earlier context (what's been said, what's under discussion) or a broader classification of communicative purpose that spans more than a single clause.

WHAT THIS FILE DOES NOT DO: Nothing here decides what a sentence's literal syntax or truth value is (core/parser.py, core/semantics.py), it consumes their output. 
And, honestly: core/parser.py doesn't currently support the sentence structures several of these mechanisms are demonstrated against (subordinate adjunct clauses, wh-questions, 
multi-sentence input), those mechanisms are tested here against directly-constructed structures (the same approach core/parser.py's own c-command tests used), correct and ready 
for when parser coverage catches up, not silently assumed to already be wired into a live end-to-end parse.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.types import (
    DerivationNode, Gender, GrammaticalNumber, DiscourseReferent, LogicalForm, Aktionsart, FrameTemplate,
)
from core.parser import collect_leaves, c_commands
from core.world_model import WorldModel

# Anaphora, including cataphora.
def resolve_pronoun_with_agreement(
    world_model: WorldModel,
    required_gender: Optional[Gender] = None,
    required_number: Optional[GrammaticalNumber] = None,
    required_animate: Optional[bool] = None,
) -> Optional[DiscourseReferent]:
    """
    Resolves an ordinary (backward-looking) pronoun against the world model's active discourse referents, honoring whatever 
    grammatical constraints are known: a candidate already tagged with a conflicting gender, number, or animacy is skipped 
    rather than grabbed just because it's recent. Searches from most to least recently introduced, returning the first referent 
    that doesn't conflict with any constraint actually supplied (an unset constraint places no restriction, and a referent whose 
    own field is unset is never treated as a conflict: only an explicit mismatch rules a candidate out).
    """
    for referent in reversed(list(world_model.active_referents.values())):
        if required_gender is not None and referent.gender is not None and referent.gender != required_gender:
            continue
        if required_number is not None and referent.number is not None and referent.number != required_number:
            continue
        if required_animate is not None and referent.animate is not None and referent.animate != required_animate:
            continue
        return referent
    
    return None

def attempt_cataphora_resolution(
    root: DerivationNode,
    pronoun_leaf: DerivationNode,
    candidate_leaves: List[DerivationNode],
) -> Optional[DerivationNode]:
    """
    Attempts to resolve a pronoun to a candidate antecedent that appears LATER in the same sentence ("Before he entered the room, John took off his coat"). 
    This is licensed only when Principle C holds: the pronoun's position must not c-command the candidate. Checking against the same c_commands() utility 
    core/parser.py's NPI licensing uses is what tells apart this sentence (fine: "he" sits inside a fronted adjunct clause that doesn't c-command "John") from
    "*He took off his coat before John entered the room" with the same intended coreference (bad: "he" is the matrix subject and does c-command the adjunct containing "John"). 
    Candidates are checked in order of appearance; the first structurally-licensed one wins.
    """
    for candidate in candidate_leaves:
        if candidate.span[0] <= pronoun_leaf.span[0]:
            continue # This candidate doesn't actually follow the pronoun: not a cataphora case.
        if not c_commands(root, pronoun_leaf, candidate):
            return candidate
        
    return None

def find_pronoun_and_name_leaves(root: DerivationNode, pronoun_tokens: set, lexicon: Any) -> Tuple[List[DerivationNode], List[DerivationNode]]:
    """
    Scans a derivation tree's leaves and splits them into pronoun leaves (tokens in `pronoun_tokens`) and proper-noun leaves. Name leaves are identified by the lexicon's own
    "proper_noun" category tag, not surface capitalization: core/parser.py's tokenizer lowercases every token before a leaf is ever built (needed so lexicon lookups are case-consistent), 
    so by the time a tree reaches this function, capitalization has already been destroyed; checking the lexicon category is both more reliable and doesn't depend on casing surviving at all.
    """
    leaves = collect_leaves(root)
    pronouns = [leaf for leaf in leaves if leaf.token and leaf.token.lower() in pronoun_tokens]
    names = []
    for leaf in leaves:
        if not leaf.token:
            continue
        
        entry = lexicon.get_word_definition(leaf.token)
        if entry and entry.get("category") == "proper_noun":
            names.append(leaf)

    return pronouns, names

# Speech acts: Searle's five-category taxonomy.
class SpeechActCategory(Enum):
    """
    Searle's standard, closed classification of what an utterance's illocutionary force actually is: replacing an ad hoc, ever-growing if/elif chain of situational labels with a fixed set
    every utterance gets sorted into.
    """
    ASSERTIVE = "assertive" # Commits the speaker to something being the case.
    DIRECTIVE = "directive" # Attempts to get the hearer to do something: including questions, which request the act of answering.
    COMMISSIVE = "commissive" # Commits the speaker to some future action.
    EXPRESSIVE = "expressive" # Expresses a psychological/social state (greetings, thanks, apologies).
    DECLARATION = "declaration" # Brings about a state of affairs by the utterance itself ("I resign").

_EXPRESSIVE_PHRASES = {"hello", "hi", "hey", "greetings", "goodbye", "bye", "thanks", "thank you", "sorry"}
_COMMISSIVE_VERBS = {"promise", "swear", "guarantee", "vow"}
_DECLARATION_VERBS = {"declare", "pronounce", "resign", "sentence", "appoint"}
_DIRECTIVE_VERBS = {"feed", "take", "explain", "show", "make", "give", "stop", "close", "open"}
_QUESTION_STARTERS = ("who", "what", "where", "when", "why", "how", "is", "can", "do", "does", "are", "will")

def classify_speech_act(logical_form: Optional[LogicalForm], raw_text: str) -> SpeechActCategory:
    """
    Classifies an utterance's speech act. Questions are correctly classified as DIRECTIVE, not their own category: in Searle's taxonomy a question is a request for the hearer to perform
    the act of informing, not a fundamentally different kind of speech act from an imperative.
    """
    clean_text = raw_text.strip().lower().rstrip("?.!")

    if clean_text in _EXPRESSIVE_PHRASES:
        return SpeechActCategory.EXPRESSIVE

    if raw_text.strip().endswith("?") or clean_text.startswith(_QUESTION_STARTERS):
        return SpeechActCategory.DIRECTIVE

    if logical_form:
        predicate_lower = logical_form.predicate.lower()
        if predicate_lower in _DECLARATION_VERBS:
            return SpeechActCategory.DECLARATION
        if predicate_lower in _COMMISSIVE_VERBS:
            return SpeechActCategory.COMMISSIVE
        if predicate_lower in _DIRECTIVE_VERBS:
            return SpeechActCategory.DIRECTIVE

    return SpeechActCategory.ASSERTIVE

# Ellipsis / QUD, and focus.
@dataclass
class QUDEntry:
    """
    One question-or-claim under discussion, kept just structured enough for a later fragment to fill in whatever's still open. open_slot_index names which position in `arguments` is the
    thing actually being asked about or left unresolved ("Are you free on Tuesday?" -> predicate FREE, arguments [you, tuesday], open_slot_index 1 - the day is what's under discussion).
    """
    predicate: str
    arguments: List[str]
    open_slot_index: Optional[int] = None

class QUDStack:
    """
    Tracks the current question/claim under discussion so bare fragment answers ("What about Friday?", "Friday?") can be resolved by filling the open slot of the most recent entry, rather
    than being treated as their own free-standing (and un-parseable) sentence.
    """
    def __init__(self):
        self._stack: List[QUDEntry] = []

    def push(self, entry: QUDEntry) -> None:
        self._stack.append(entry)

    def current(self) -> Optional[QUDEntry]:
        return self._stack[-1] if self._stack else None

    def resolve_fragment(self, fragment_tokens: List[str]) -> Optional[LogicalForm]:
        """
        Fills the current QUD entry's open slot with the fragment, returning a full LogicalForm, or None if there's nothing open to fill.
        """
        entry = self.current()
        if entry is None or entry.open_slot_index is None:
            return None
        
        filled_arguments = list(entry.arguments)
        filled_arguments[entry.open_slot_index] = " ".join(fragment_tokens)
        return LogicalForm(predicate = entry.predicate, arguments = filled_arguments)

FOCUS_PARTICLES = {"only", "even"}

def detect_focus(tokens: List[str]) -> Optional[str]:
    """
    Returns the token immediately following a focus particle ("only", "even"), if present: a minimal, honest treatment of focus: it identifies WHAT's 
    focused, not the fuller alternative-set/exhaustivity semantics real focus theory involves, which is real further work.
    """
    for index, token in enumerate(tokens):
        if token in FOCUS_PARTICLES and index + 1 < len(tokens):
            return tokens[index + 1]
    return None

# Presupposition accommodation.
def accommodate_presupposition(
    world_model: WorldModel,
    trigger_word: str,
    subject: str,
    related_relation: Optional[FrameTemplate] = None,
) -> bool:
    """
    When a presupposition trigger fires ("my sister is coming over") and its presupposed content isn't already known, this doesn't fail the sentence: 
    it checks the Episodic Fact Graph first (in case the fact was actually stated earlier and just not linked to this trigger), and failing
    that, assumes the presupposition holds and records the assumption so the main assertion can proceed. Returns True if something was newly accommodated, 
    False if it was already known (there was nothing to accommodate). In the full architecture, an external/model-assisted lookup would
    be tried before falling back to blind accommodation: that's interface/neural_bridge.py's job, not yet built; this function accommodates directly once its 
    own local checks are exhausted.
    """
    existing = world_model.check_presupposition(trigger_word, subject)
    if existing is not None:
        return False

    if related_relation is not None and world_model.query_episodic_fact(related_relation, subject) is not None:
        world_model.assert_presupposition(trigger_word, subject, holds = True)
        return True

    world_model.assert_presupposition(trigger_word, subject, holds = True)
    return True

# Scalar implicature: Horn scales.
HORN_SCALES: Tuple[Tuple[str, str], ...] = (
    ("some", "all"),
    ("good", "excellent"),
    ("warm", "hot"),
    ("or", "and"),
    ("might", "must"),
)

def generate_scalar_implicature(word: str) -> Optional[str]:
    """
    Given a weaker term from a Horn scale, returns the defeasible negative implicature its use generates (e.g. "some" -> "NOT_ALL"): defeasible meaning 
    it can be cancelled ("some, in fact all, of you passed" is coherent), which is exactly what distinguishes an implicature from a
    presupposition (7d's mechanism, which survives negation and isn't casually cancelled). Returns None for a word that isn't the weak member of any registered scale.
    """
    word_lower = word.lower()
    for weak_term, strong_term in HORN_SCALES:
        if word_lower == weak_term:
            return f"NOT_{strong_term.upper()}"
        
    return None

# SDRT-lite: a closed six-relation rhetorical structure classifier.
class RhetoricalRelation(Enum):
    """
    A deliberately small, closed core of rhetorical relations: pruned down from the much larger and less standardized inventories found across the wider SDRT literature, 
    specifically to keep this consistent with every other closed vocabulary in the system.
    """
    NARRATION = "narration"
    ELABORATION = "elaboration"
    CONTRAST = "contrast"
    EXPLANATION = "explanation"
    BACKGROUND = "background"
    RESULT = "result"

_CONNECTIVE_RELATIONS: Dict[str, RhetoricalRelation] = {
    "but": RhetoricalRelation.CONTRAST,
    "however": RhetoricalRelation.CONTRAST,
    "although": RhetoricalRelation.CONTRAST,
    "because": RhetoricalRelation.EXPLANATION,
    "since": RhetoricalRelation.EXPLANATION,
    "so": RhetoricalRelation.RESULT,
    "therefore": RhetoricalRelation.RESULT,
    "thus": RhetoricalRelation.RESULT,
    "then": RhetoricalRelation.NARRATION,
}

_EVENTIVE_AKTIONSART = {Aktionsart.ACHIEVEMENT, Aktionsart.ACCOMPLISHMENT}

def classify_rhetorical_relation(
    first_aktionsart: Optional[Aktionsart],
    second_aktionsart: Optional[Aktionsart],
    connective: Optional[str] = None,
) -> RhetoricalRelation:
    """
    Classifies how a second clause relates to a first one. Meant to run only across sentence boundaries (per the architecture's lazy-execution principle: this is not meant to 
    fire within a single utterance). An explicit connective, when present, is checked first against a closed cue-phrase table. Otherwise, this falls back to the standard SDRT 
    diagnostic: two event-like (achievement/accomplishment) clauses in sequence signal forward progression (Narration), while a state description following an event signals Background 
    rather than continued narration, this is exactly why aktionsart (core/semantics.py's derive_aktionsart) is a genuine input to this classifier, not an independent, unrelated mechanism.
    """
    if connective is not None and connective.lower() in _CONNECTIVE_RELATIONS:
        return _CONNECTIVE_RELATIONS[connective.lower()]

    if first_aktionsart in _EVENTIVE_AKTIONSART and second_aktionsart in _EVENTIVE_AKTIONSART:
        return RhetoricalRelation.NARRATION
    if second_aktionsart == Aktionsart.STATE:
        return RhetoricalRelation.BACKGROUND
    
    return RhetoricalRelation.ELABORATION
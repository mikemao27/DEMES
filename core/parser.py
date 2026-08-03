r"""
Implements DEMES's Combinatory Categorial Grammar (CCG) engine: a real chart parser that figures out sentence structure by combining 
words according to a small, fixed set of combination rules.

WHAT PROBLEM THIS FILE SOLVES: An earlier version of this parser worked by scanning a sentence left to right and guessing that
"the last word that looks like a predicate is the predicate." That happened to work for simple sentences ("the suitcase is portable") but 
broke on almost anything else: ordinary subject-verb-object sentences ("John kicked the ball") got their predicate picked backwards, and
there was no way at all to tell an adjective describing a noun ("the greater suitcase") apart from an adjective that IS the sentence's claim 
("the suitcase is greater"), because the old parser never built any actual sentence structure to check: just a flat list of words.

HOW THIS FILE FIXES THAT: Every word gets tagged with a syntactic category describing what it combines with and what it
produces, not unlike a jigsaw-puzzle piece shape. An intransitive verb like "walks" is tagged S\NP: "give me a noun phrase on my LEFT, and I'll produce a full sentence." 
A transitive verb like "kicked" is tagged (S\NP)/NP: "give me a noun phrase on my RIGHT first, then one on my left, and I'll produce a sentence." 
A handful of fixed combination rules (see CCGCombinators below) decide when two adjacent pieces fit together and what shape falls out. Critically, an adjective gets
offered MULTIPLE candidate categories, N/N (modifies a following noun, e.g. "the greater suitcase") and S\NP (is the sentence's predicate, e.g. "the suitcase is greater"), 
and which one actually gets used isn't guessed in advance; it falls out of which one lets the whole sentence combine into a complete S. This is what makes the grammar 
genuinely general instead of a pile of sentence-shape-specific rules: new sentence patterns (coordination, embedding, questions, ...) don't need new parser code as they 
get added later, only new category assignments for the words involved: the combination rules themselves stay fixed.

The tree of how words combined (a DerivationNode, from core/types.py) is kept around after parsing rather than thrown away, specifically so later checks, 
deciding whether a pronoun may refer to a name later in the same sentence, or whether a word like "any" is licensed by a nearby negation, have real 
sentence structure to check against.

WHAT THIS FILE DOES NOT YET DO, ON PURPOSE: This is real CCG composition for the sentence shapes DEMES has been built and tested against so far: simple declaratives,
quantified noun phrases (multiple per sentence), plural nouns, negation, tense marking, transitive/intransitive verbs, attributive/predicative/comparative adjectives,
adjunct/subordinate clauses, clausal complements ("John thinks that Mary is home"), whole-sentence "and"/"or" coordination (a dedicated, closed ternary chart rule,
deliberately narrower than general CCG coordination: see COORDINATORS' own docstring), passive voice ("The suitcase was kicked [by John]", is_passive on the resulting 
LogicalForm flagging that the surface subject is the PATIENT, not the AGENT), and non-movement wh-questions ("Who walked?", "What is the suitcase?": the wh-word fills 
an ordinary argument position directly; see WH_WORDS' own docstring for why real wh-movement, extracting a gap from an embedded position like "What did John kick?", is 
deliberately out of scope). type_raise remains implemented and tested here but not wired into the chart's automatic search (real wh-movement is the mechanism that would need it), 
to keep the search space from exploding before there's a real need to. Extending coverage later means adding more category data, not rewriting this engine.

This file also does not resolve pronouns or decide what a sentence's truth value is: producing the correct sentence STRUCTURE is this file's job; resolving 
what "it" refers to (core/discourse.py) and evaluating whether the sentence is true (core/semantics.py, core/world_model.py) are separate, later jobs 
that consume this file's output.
"""

from typing import List, Dict, Optional, Tuple, Any, Union

from core.types import LogicalForm, DerivationNode, Gender, GrammaticalNumber, StoredQuantifier

# Categories: the "jigsaw-puzzle piece shapes" words get tagged with.
class CCGCategory:
    r"""
    A CCG syntactic category. An atomic category (like S, NP, N, PP) is just a label: CCGCategory ("NP") means "a complete noun phrase." 
    A complex category describes something still waiting to combine with more material: CCGCategory(CCGCategory("S"), CCGCategory("NP"), "\\") is written
    S\NP and means "combine me with an NP to my LEFT and you get a full sentence": the category of a plain intransitive verb phrase like "walks". 
    The direction is "/" for "I want my argument to my RIGHT" and "\\" for "I want my argument to my LEFT", matching the sentence's actual word
    order once combined.
    """
    def __init__(self, result, argument=None, direction: Optional[str] = None):
        self.result = result
        self.argument = argument
        self.direction = direction

    def is_atomic(self) -> bool:
        return self.argument is None

    def __eq__(self, other) -> bool:
        if not isinstance(other, CCGCategory):
            return NotImplemented
        
        return (
            self.result == other.result
            and self.argument == other.argument
            and self.direction == other.direction
        )

    def __hash__(self) -> int:
        return hash((self.result, self.argument, self.direction))

    def __repr__(self) -> str:
        if self.is_atomic():
            return self.result
        return f"({self.result!r}{self.direction}{self.argument!r})"

# Atomic categories, defined once and reused everywhere.
S = CCGCategory("S")
NP = CCGCategory("NP")
N = CCGCategory("N")
PP = CCGCategory("PP")

# A coordinator ("and"/"or") deliberately gets an atomic category that is inert under every ordinary combinator (try_forward_application etc. all bail
# out immediately on an atomic left/right, per their own guards above): it can never combine with a neighbor via application or composition. It is
# consumed only by the dedicated ternary coordination rule in ChartParser._run_chart, so tagging it this way guarantees it can't spuriously participate
# in any other derivation.
COORD_ATOM = CCGCategory("COORD")

# Complex categories built out of the atomic ones, given readable names.
INTRANSITIVE_VERB = CCGCategory(S, NP, "\\") # S\NP.
TRANSITIVE_VERB = CCGCategory(INTRANSITIVE_VERB, NP, "/") # (S\NP)/NP.
DITRANSITIVE_VERB = CCGCategory(TRANSITIVE_VERB, NP, "/") # ((S\NP)/NP)/NP.
PREDICATIVE_ADJECTIVE = CCGCategory(S, NP, "\\") # S\NP (same shape as an intransitive verb).
ATTRIBUTIVE_ADJECTIVE = CCGCategory(N, N, "/") # N/N.
COMPARATIVE_PREDICATIVE = CCGCategory(PREDICATIVE_ADJECTIVE, PP, "/") # (S\NP)/PP: "greater [than X]".
COMPARATIVE_ATTRIBUTIVE = CCGCategory(ATTRIBUTIVE_ADJECTIVE, PP, "/") # (N/N)/PP.
DETERMINER = CCGCategory(NP, N, "/") # NP/N.
PREDICATE_MODIFIER = CCGCategory(PREDICATIVE_ADJECTIVE, PREDICATIVE_ADJECTIVE, "/") # (S\NP)/(S\NP): copula, negation, tense auxiliaries.
COMPARISON_PP = CCGCategory(PP, NP, "/") # PP/NP: "than".
SENTENCE_MODIFIER = CCGCategory(S, S, "/") # S/S: something that turns one following sentence into another, e.g. a FRONTED adjunct clause ("Before X, Y").

# A TRAILING adjunct ("Y before X") attaches at the VP level, (S\NP)\(S\NP): not S\S. This matters for more than tidiness: attaching at the whole-sentence 
# level would place the adjunct structurally OUTSIDE the subject's c-command domain, which gets Principle C backwards: a trailing adjunct is standardly a VP-adjunct, 
# and the subject genuinely does c-command into it (that's exactly why "*He arrived before John did" is bad for intended coreference, unlike the fronted "Before he arrived, John left"). 
# Attaching at the VP means the subject and the adjunct both end up under the same S node once the subject combines, giving c-command tests the right geometry to work with.
TRAILING_VP_MODIFIER = CCGCategory(INTRANSITIVE_VERB, INTRANSITIVE_VERB, "\\") # (S\NP)\(S\NP).
CLAUSAL_VERB = CCGCategory(INTRANSITIVE_VERB, S, "/") # (S\NP)/S: a mental-predicate verb like "think" taking a full embedded sentence as its complement, not an NP.
COMPLEMENTIZER = CCGCategory(S, S, "/") # S/S: "that", consuming its clause directly to produce a complete S in one step: unlike a subordinator, which needs an extra step (see ADJUNCT_TAKING) before it can attach to a matrix clause.
ADJUNCT_TAKING = CCGCategory(SENTENCE_MODIFIER, S, "/") # (S/S)/S: a subordinator like "before" in fronted position, consuming its own clause first.
TRAILING_ADJUNCT_TAKING = CCGCategory(TRAILING_VP_MODIFIER, S, "/") # ((S\NP)\(S\NP))/S: the same subordinator in trailing position.

# "By John" in a passive sentence ("The suitcase was kicked by John") reuses the same VP-level attachment shape a trailing adjunct does: ((S\NP)\(S\NP))/NP,
# taking the agent NP to its right and, once combined, backward-applying to an already-complete passive VP without changing its category, since that is
# simply what "a thing that attaches after a VP and gives back a VP" means categorially, the same shape any right-adjoining VP modifier must have. Because
# of that reuse, _matrix_clause_leaves's adjunct-exclusion check (below) can't tell the two apart by category label alone, and is refined accordingly.
BY_AGENT = CCGCategory(TRAILING_VP_MODIFIER, NP, "/") # ((S\NP)\(S\NP))/NP.

# Combinators: the small, fixed set of rules for combining two adjacent categories.
def try_forward_application(left: CCGCategory, right: CCGCategory) -> Optional[CCGCategory]:
    """
    X/Y combined with Y (to its right) yields X. E.g. "the"(NP/N) + "suitcase"(N) -> NP.
    """
    if left.is_atomic() or left.direction != "/":
        return None
    if left.argument == right:
        return left.result
    
    return None

def try_backward_application(left: CCGCategory, right: CCGCategory) -> Optional[CCGCategory]:
    r"""
    Y combined with X\Y (to its right) yields X. E.g. "suitcase"(NP) + "walks"(S\NP) -> S.
    """
    if right.is_atomic() or right.direction != "\\":
        return None
    if right.argument == left:
        return right.result
    
    return None

def try_forward_composition(left: CCGCategory, right: CCGCategory) -> Optional[CCGCategory]:
    """
    X/Y combined with Y/Z yields X/Z: lets two rightward-looking pieces chain together.
    """
    if left.is_atomic() or left.direction != "/":
        return None
    if right.is_atomic() or right.direction != "/":
        return None
    if left.argument == right.result:
        return CCGCategory(left.result, right.argument, "/")
    
    return None

def try_backward_composition(left: CCGCategory, right: CCGCategory) -> Optional[CCGCategory]:
    r"""
    Y\Z combined with X\Y yields X\Z: the mirror image of forward composition.
    """
    if left.is_atomic() or left.direction != "\\":
        return None
    if right.is_atomic() or right.direction != "\\":
        return None
    if left.result == right.argument:
        return CCGCategory(right.result, left.argument, "\\")
    
    return None

def type_raise(category: CCGCategory, result_type: CCGCategory) -> CCGCategory:
    r"""
    Turns a category X into T/(T\X): "instead of waiting to be some verb's argument, go look for a verb that wants an X and produce a T myself." 
    This combinator is implemented and tested here because the architecture plan commits to it, but it is not yet invoked by the chart's search
    below: applying it to every category everywhere would multiply the search space for a benefit this project's sentences don't need yet. 
    It's a tested building block, ready for when the chart's search strategy is extended to use it selectively (e.g. for coordination).
    """
    inner = CCGCategory(result_type, category, "\\")
    return CCGCategory(result_type, inner, "/")

_APPLICATION_PENALTY = 0
# CCG's well-known "spurious ambiguity": composition can derive the same span via a structurally different path than plain application would, even 
# when application alone already succeeds (e.g. an adjunct clause composing into the middle of a verb phrase instead of attaching to the finished sentence). 
# Penalizing composition relative to application is the standard fix: it doesn't disable composition (still explored, and still wins when it's the only way to succeed), 
# it just means a plain-application derivation is preferred over a composition-based one whenever the chart actually has a choice between two derivations of the same category.
_COMPOSITION_PENALTY = 1

# Coordination competes for the same span as ordinary application/composition would, but never redundantly (nothing else can produce a valid category
# for a span whose middle token is a bare coordinator: application/composition would need it to combine with a real category on one side, and
# COORD_ATOM is inert), so this penalty exists for consistency with the other two mechanisms' documented ranking discipline rather than to resolve any
# known ambiguity.
_COORDINATION_PENALTY = 1
_COMBINATOR_RULES: Tuple[Tuple[str, Any, int], ...] = (
    (">", try_forward_application, _APPLICATION_PENALTY),
    ("<", try_backward_application, _APPLICATION_PENALTY),
    (">B", try_forward_composition, _COMPOSITION_PENALTY),
    ("<B", try_backward_composition, _COMPOSITION_PENALTY),
)

# Closed function-word tables. Same role as an ordinary category assignment, but for words whose job is purely grammatical rather than being a lexicon 
# entry with its own decomposable meaning.
QUANTIFIERS: Dict[str, str] = {
    "every": "FORALL",
    "all": "FORALL",
    "some": "EXISTS",
    "a": "EXISTS",
    "no": "NOT_EXISTS",
}

DETERMINERS = {"the", "a", "an"}
COPULA_PRESENT = {"is", "are"}
COPULA_PAST = {"was", "were"}
NEGATION_WORDS = {"not", "never"}
FUTURE_MARKERS = {"will"}
PAST_AUX = {"did"}
COMPARISON_MARKERS = {"than"}
FOCUS_PARTICLES = {"only", "even"}

# Subordinating conjunctions introducing an adjunct clause ("Before he entered the room, ..."). A closed set, same discipline as every other function-word table 
# here: not an attempt to cover every English subordinator, just the ones needed to make adjunct clauses parseable at all.
SUBORDINATORS = {"before", "after", "while", "when"}

# The complementizer introducing a finite clausal complement ("John thinks THAT Mary is home"). Deliberately not extended to infinitival-complement verbs
# ("want to leave"): that's a genuinely different construction (an infinitive "to" + bare verb, not a finite embedded sentence) and isn't attempted here.
COMPLEMENTIZER_WORDS = {"that"}

# Coordinators, each mapped to the closed logical connective it represents. Ordinary CCG coordination needs a category variable ("X and X -> X" for any
# X), which the category algebra above deliberately doesn't support: see ChartParser._run_chart's dedicated coordination pass for how this closed,
# narrower mechanism (same concrete category on both sides of the coordinator) covers "John left and Mary left" / "the suitcase or the trophy" without it.
COORDINATORS: Dict[str, str] = {"and": "AND", "or": "OR"}

# The preposition introducing a passive sentence's agent phrase ("The suitcase was kicked BY John"). A closed, one-word set: the same discipline as every
# other function-word table here, not an attempt to cover prepositions generally.
PASSIVE_AGENT_MARKERS = {"by"}

# Question words, scoped deliberately to Phase 2.6's narrow, non-movement wh-question patterns: the wh-word fills an ordinary argument position directly 
# ("Who walked?": subject position, exactly where "John" would sit) rather than leaving a gap somewhere else in the tree that has to be tracked and passed up 
# through arbitrary structure (real wh-movement, a substantially bigger mechanism this phase deliberately doesn't attempt: see this file's own top docstring). 
# Deliberately NOT part of _FUNCTION_WORDS below, the same reasoning PRONOUNS itself documents: a wh-word IS the thing core/discourse.py's QUD stack needs to find 
# in `arguments` afterward, so it must survive extraction rather than being filtered out as noise.
WH_WORDS = {"who", "what"}

_DETERMINER_LIKE = set(QUANTIFIERS) | DETERMINERS
_PREDICATE_MODIFIER_WORDS = COPULA_PRESENT | COPULA_PAST | NEGATION_WORDS | FUTURE_MARKERS | PAST_AUX
_FUNCTION_WORDS = (
    _DETERMINER_LIKE | _PREDICATE_MODIFIER_WORDS | COMPARISON_MARKERS | FOCUS_PARTICLES | SUBORDINATORS
    | COMPLEMENTIZER_WORDS | set(COORDINATORS) | PASSIVE_AGENT_MARKERS
)

# Pronouns are closed-class function words syntactically (the same status as determiners or negation) so their category and agreement features come 
# from this fixed table, never from a lexicon.json entry. (An earlier design routed them through the ordinary lexicon, which required giving them a 
# primitive decomposition; "PRONOUN" was never a real NSM primitive, so "it"/"they" were silently rejected by core/lexicon.py's closure validation the 
# moment that validation went in: pronouns were unparseable until this table replaced that approach.) Deliberately NOT part of _FUNCTION_WORDS above: 
# unlike determiners, a pronoun IS the argument core/discourse.py needs to resolve later, so it must survive extraction rather than being filtered out as noise.
PRONOUNS: Dict[str, Dict] = {
    "it": {"gender": None, "number": GrammaticalNumber.SINGULAR, "animate": False},
    "he": {"gender": Gender.MASCULINE, "number": GrammaticalNumber.SINGULAR, "animate": True},
    "him": {"gender": Gender.MASCULINE, "number": GrammaticalNumber.SINGULAR, "animate": True},
    "she": {"gender": Gender.FEMININE, "number": GrammaticalNumber.SINGULAR, "animate": True},
    "her": {"gender": Gender.FEMININE, "number": GrammaticalNumber.SINGULAR, "animate": True},
    "they": {"gender": None, "number": GrammaticalNumber.PLURAL, "animate": None},
    "them": {"gender": None, "number": GrammaticalNumber.PLURAL, "animate": None},
}

def get_pronoun_features(word: str) -> Optional[Dict]:
    """
    Returns the closed agreement-feature record for a pronoun, or None if `word` isn't one.
    """
    return PRONOUNS.get(word.lower())

def get_coordination_conjuncts(root: Optional[DerivationNode]) -> Optional[Tuple[DerivationNode, DerivationNode]]:
    """
    If `root` was built by the coordination pass in ChartParser._run_chart, returns its (left_conjunct, right_conjunct) subtrees; None otherwise. This
    is a purely structural check (no special label marker needed on the node itself) precisely because a coordination node's arity (exactly three
    children, with the middle one a coordinator leaf) never arises any other way in this grammar: every other combinator in _COMBINATOR_RULES always
    builds a 2-child node. Used both by parse_with_derivation (to extract each conjunct as its own LogicalForm) and by core/pipeline.py (to resolve
    pronouns/cataphora independently within each conjunct's own subtree, never across them: see that file's own docstring for why crossing conjuncts
    would be wrong).
    """
    if root is None or len(root.children) != 3 or not root.children[1].is_leaf() or root.children[1].token not in COORDINATORS:
        return None
    return (root.children[0], root.children[2])

def get_coordinator_connective(root: Optional[DerivationNode]) -> Optional[str]:
    """
    If `root` is a top-level coordination node (see get_coordination_conjuncts), returns which closed logical connective ("AND"/"OR") its coordinator
    word represents; None otherwise. Lets core/pipeline.py combine a coordinated sentence's per-conjunct truth values correctly without needing to know
    anything about DerivationNode structure itself.
    """
    if root is None or len(root.children) != 3 or not root.children[1].is_leaf():
        return None
    return COORDINATORS.get(root.children[1].token)

# Negative polarity items and the closed set of words allowed to license them (Ladusaw/Fauconnier downward-entailment licensing): "I didn't see anyone" is fine, 
# "*I saw anyone" is not, because nothing negative structurally sits above "anyone" in the second sentence.
NPI_WORDS = {"any", "anyone", "anything", "ever", "at_all"}
NPI_LICENSORS = NEGATION_WORDS

# Idiom category-gating (syntax half only: core/figurative.py, not yet built, does the semantic rewrite). Each entry: trigger verb's base (lemma) form -> the exact object head noun 
# (with any determiner already stripped, matching how `arguments` is built below) that unlocks the idiomatic reading. This table is deliberately tiny right now; 
# it exists to prove the mechanism works, not to be a real idiom dictionary yet.
_IDIOM_OBJECT_TRIGGERS: Dict[str, str] = {
    "kick": "bucket",
}

# Supertagging: deciding which candidate categories a word could plausibly have, ranked best-first.
def supertag_content_word(word: str, lexicon_entry: Dict, inflection: Optional[str]) -> List[CCGCategory]:
    """
    Given a word's dictionary entry (from LexiconManager) and what inflection was detected on its surface form (e.g. "comparative" for "greater"), returns its candidate CCG categories, 
    most likely reading first. At DEMES's current vocabulary size each word's part of speech is already known from the lexicon, so this ranking is a direct, deterministic lookup rather than a learned
    model: exactly the "frequency/lexicon-driven ranking is enough to start" starting point the architecture plan calls for; a statistical supertagger is a later upgrade to this one function,
    not a rebuild of the chart around it.
    """
    category = lexicon_entry.get("category")
    valency = lexicon_entry.get("valency")

    if category == "noun":
        return [N]
    if category in ("proper_noun", "pronoun"):
        return [NP]
    if category == "verb":
        # A transitive/ditransitive verb used in its PARTICIPLE form ("kicked", "taken") can also be the demoted verb of a passive sentence ("was kicked
        # [by John]"), which takes no object at all: INTRANSITIVE_VERB (S\NP) is offered as a second, lower-ranked candidate whenever the inflection is
        # consistent with participle use. This is genuinely ambiguous for a regular verb (inflection "past" doesn't distinguish "kicked the ball" from
        # "was kicked": see core/lexicon.py's own docstring on this), so the chart decides which reading actually succeeds, the same underspecified-until-
        # composition pattern predicative-vs-attributive adjectives already use; "irregular_past_participle" is unambiguous and only ever offers this reading.
        if valency == "transitive":
            candidates = [TRANSITIVE_VERB]
            if inflection in ("past", "irregular_past_participle"):
                candidates.append(INTRANSITIVE_VERB)

            return candidates
        
        if valency == "ditransitive":
            candidates = [DITRANSITIVE_VERB]
            if inflection in ("past", "irregular_past_participle"):
                candidates.append(INTRANSITIVE_VERB)

            return candidates
        
        if valency == "clausal":
            return [CLAUSAL_VERB]

        return [INTRANSITIVE_VERB]
    
    if category == "adjective":
        if inflection == "comparative":
            return [COMPARATIVE_PREDICATIVE, PREDICATIVE_ADJECTIVE, COMPARATIVE_ATTRIBUTIVE, ATTRIBUTIVE_ADJECTIVE]
        return [PREDICATIVE_ADJECTIVE, ATTRIBUTIVE_ADJECTIVE]

    return [N]

def supertag_function_word(word: str) -> List[CCGCategory]:
    """
    The fixed category for each closed-class function word: never ambiguous by definition.
    """
    if word in _DETERMINER_LIKE:
        return [DETERMINER]
    if word in COPULA_PRESENT or word in COPULA_PAST:
        # Ordinary predicative attachment ((S\NP)/(S\NP), unchanged from before) is the default, most likely reading; TRANSITIVE_VERB is offered as a 
        # second candidate so an inverted copular question ("What is the suitcase?", "Who is John?") can parse too: the copula here behaves as an
        # ordinary two-place identity relation between the wh-word and the following NP, exactly the same SVO combinatorics "John kicked the ball"
        # already uses (see _extract_logical_form's own handling of this reading), not a new mechanism. The TRANSITIVE_VERB reading is inert whenever
        # nothing NP-shaped actually follows (an ordinary adjectival complement like "portable" never is), so this never competes with the ordinary
        # declarative reading.
        return [PREDICATE_MODIFIER, TRANSITIVE_VERB]
    if word in _PREDICATE_MODIFIER_WORDS:
        return [PREDICATE_MODIFIER]
    if word in COMPARISON_MARKERS:
        return [COMPARISON_PP]
    if word in FOCUS_PARTICLES:
        return [PREDICATE_MODIFIER]
    if word in NPI_WORDS:
        return [PREDICATE_MODIFIER] if word in ("ever", "at_all") else [NP]
    if word in PRONOUNS:
        return [NP]
    if word in SUBORDINATORS:
        # Both orders are offered as candidates ("Before X, Y" and "Y before X"): which one actually combines into a complete S is left for the chart to discover, the same way
        # an adjective's predicative-vs-attributive reading is.
        return [ADJUNCT_TAKING, TRAILING_ADJUNCT_TAKING]
    if word in COMPLEMENTIZER_WORDS:
        return [COMPLEMENTIZER]
    if word in COORDINATORS:
        # Tagged, but only so _supertag_all doesn't reject the sentence as containing an unrecognized word: COORD_ATOM is inert under every ordinary
        # combinator (see its own definition above), so this candidate category never actually gets used by the standard application/composition search;
        # the coordinator is consumed entirely by _run_chart's dedicated ternary coordination pass instead.
        return [COORD_ATOM]
    if word in PASSIVE_AGENT_MARKERS:
        return [BY_AGENT]
    if word in WH_WORDS:
        # Plain NP, the same category an ordinary pronoun or proper noun gets: this is what makes a SUBJECT wh-question ("Who walked?") parse with zero
        # new combinators at all: "who" simply occupies subject position exactly where "John" would. The inverted copular-question pattern above is
        # what additionally lets "What is the suitcase?" parse, using this same NP category for "what" on the other side of that combination.
        return [NP]

    return []

# Derivation-tree utilities: c-command, built directly on DerivationNode.dominates().
def find_parent(root: DerivationNode, target: DerivationNode) -> Optional[DerivationNode]:
    """
    The immediate parent of `target` within `root`'s tree, or None if `target` is the root itself.
    """
    for child in root.children:
        if child is target:
            return root
        found = find_parent(child, target)
        if found is not None:
            return found
        
    return None

def c_commands(root: DerivationNode, a: DerivationNode, b: DerivationNode) -> bool:
    """
    True if `a` c-commands `b` within `root`'s tree: neither dominates the other, and the first branching node above `a` (in this always-binary grammar, simply `a`'s immediate parent) does
    dominate `b`. This is the structural fact Principle C cataphora checks and NPI licensing are both built from: deciding what to *do* with that fact is each check's own job, not this one.
    """
    if a is b:
        return False
    if a.dominates(b):
        return False
    parent = find_parent(root, a)

    if parent is None:
        return False
    return parent.dominates(b)

def collect_leaves(node: DerivationNode) -> List[DerivationNode]:
    """
    Every word-bearing leaf under `node`, left to right.
    """
    if node.is_leaf():
        return [node]
    leaves: List[DerivationNode] = []

    for child in node.children:
        leaves.extend(collect_leaves(child))

    return leaves

def check_npi_licensing(root: DerivationNode) -> List[str]:
    """
    Returns the surface text of any negative-polarity-item leaf found in the tree without a c-commanding licensor above it: an empty list means the sentence is fine.
    """
    leaves = collect_leaves(root)
    violations = []
    for leaf in leaves:
        if not leaf.token or leaf.token.lower() not in NPI_WORDS:
            continue
        licensed = any(
            other.token and other.token.lower() in NPI_LICENSORS and c_commands(root, other, leaf)
            for other in leaves
        )

        if not licensed:
            violations.append(leaf.token)

    return violations

# The chart parser itself.
_ChartCell = List[Tuple[CCGCategory, DerivationNode, int]] # (category, tree, rank score: lower is better).

class ChartParser:
    """
    Builds sentence structure by trying every way of combining adjacent word groups (a standard CYK-style chart: fill in results for short spans first, 
    then build longer spans out of shorter ones already computed) using only the fixed combinator rules above, and keeps the resulting tree for later layers to inspect.
    """
    def __init__(self, lexicon_manager: Any):
        self.lexicon = lexicon_manager

    def tokenize(self, sentence: str) -> List[str]:
        """
        Splits a clean sentence into lowercase words, stripping punctuation from both the whole sentence's ends and from each individual word: the latter matters as of 
        adjunct clauses, which routinely have an internal comma boundary ("Before X, Y") that would otherwise glue onto the following word as if it were part of it.
        """
        clean_sentence = sentence.strip("?.!")
        raw_words = [word.strip(",;:").lower() for word in clean_sentence.split()]
        return [word for word in raw_words if word]

    def parse(self, sentence: str) -> Optional[Union[LogicalForm, List[LogicalForm]]]:
        """
        Parses a sentence end to end and returns just its LogicalForm: the entry point every existing caller and test uses. See parse_with_derivation() for the 
        variant that also returns the derivation tree itself, needed by later layers (core/discourse.py's cataphora resolution) that need real sentence structure, 
        not just the extracted result.
        """
        logical_form, _root = self.parse_with_derivation(sentence)
        return logical_form

    def parse_with_derivation(self, sentence: str) -> Tuple[Optional[Union[LogicalForm, List[LogicalForm]]], Optional[DerivationNode]]:
        """
        Same end-to-end process as parse(), but also returns the winning derivation tree rather than discarding it. Returns (None, None) if the sentence contains a word
        the lexicon can't find at all, if no combination of categories produces a complete sentence, or if an NPI licensing check fails.

        Returns a List[LogicalForm] instead of a single LogicalForm when the whole sentence is itself a coordinated pair ("John left and Mary left"):
        each conjunct is extracted independently (its own predicate, arguments, negation, tense: see _extract_logical_form), and core/pipeline.py is
        what actually evaluates and combines them via the coordinator's connective. Coordination NESTED inside a clause ("the suitcase or the trophy is
        heavy") is not specially detected here: the coordinated NP's two nouns simply both survive as ordinary flat arguments of that clause's single
        LogicalForm, the same honest flat-extraction limitation already documented on _extract_logical_form itself.
        """
        tokens = self.tokenize(sentence)
        if not tokens:
            return None, None

        cell_options = self._supertag_all(tokens)
        if cell_options is None:
            return None, None # An unrecognized word was encountered.

        root = self._run_chart(tokens, cell_options)
        if root is None:
            return None, None

        npi_violations = check_npi_licensing(root)
        if npi_violations:
            return None, None

        conjunct_trees = get_coordination_conjuncts(root)
        if conjunct_trees is not None:
            conjuncts = [self._extract_logical_form(tree) for tree in conjunct_trees]
            return conjuncts, root

        return self._extract_logical_form(root), root

    def _supertag_all(self, tokens: List[str]) -> Optional[List[List[CCGCategory]]]:
        """
        Looks up ranked candidate categories for every token, or None if any token is unknown.
        """
        all_options: List[List[CCGCategory]] = []
        for token in tokens:
            function_categories = supertag_function_word(token)
            if function_categories:
                all_options.append(function_categories)
                continue

            entry = self.lexicon.get_word_definition(token)
            if not entry:
                return None

            inflection = self.lexicon.detect_inflection(token)
            all_options.append(supertag_content_word(token, entry, inflection))

        return all_options

    def _run_chart(self, tokens: List[str], cell_options: List[List[CCGCategory]]) -> Optional[DerivationNode]:
        """
        The CYK chart fill. Returns the best full-sentence derivation tree, or None if none exists.
        """
        n = len(tokens)
        chart: Dict[Tuple[int, int], _ChartCell] = {}

        for i in range(n):
            chart[(i, i + 1)] = [
                (category, DerivationNode(label = repr(category), token = tokens[i], span = (i, i + 1)), rank)
                for rank, category in enumerate(cell_options[i])
            ]

        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                cell: _ChartCell = []
                for k in range(i + 1, j):
                    for left_category, left_node, left_rank in chart[(i, k)]:
                        for right_category, right_node, right_rank in chart[(k, j)]:
                            for _name, rule, penalty in _COMBINATOR_RULES:
                                result = rule(left_category, right_category)

                                if result is None:
                                    continue

                                node = DerivationNode(
                                    label = repr(result),
                                    children = (left_node, right_node),
                                    span = (i, j),
                                )
                                cell.append((result, node, left_rank + right_rank + penalty))

                # Coordination: a third, deliberately narrow combination mechanism alongside application/composition above, not a general CCG
                # coordination rule (which would need a category variable the algebra doesn't support: see COORDINATORS' own docstring). For a
                # coordinator token sitting at some position strictly between i and j, if the span to its left and the span to its right each already
                # produced the SAME concrete category, that category is also a valid result for the whole span: exactly enough to parse
                # "John left and Mary left" / "the suitcase or the trophy" without touching the category algebra itself.
                for coord_pos in range(i + 1, j - 1):
                    if tokens[coord_pos] not in COORDINATORS:
                        continue

                    left_conjunct_cell = chart[(i, coord_pos)]
                    right_conjunct_cell = chart[(coord_pos + 1, j)]
                    coordinator_leaf = DerivationNode(label = repr(COORD_ATOM), token = tokens[coord_pos], span = (coord_pos, coord_pos + 1))

                    for left_category, left_node, left_rank in left_conjunct_cell:
                        for right_category, right_node, right_rank in right_conjunct_cell:
                            if left_category != right_category:
                                continue

                            node = DerivationNode(
                                label = repr(left_category),
                                children = (left_node, coordinator_leaf, right_node),
                                span = (i, j),
                            )
                            cell.append((left_category, node, left_rank + right_rank + _COORDINATION_PENALTY))

                chart[(i, j)] = cell

        final_cell = chart.get((0, n), [])
        sentence_derivations = [(node, score) for category, node, score in final_cell if category == S]
        if not sentence_derivations:
            return None

        sentence_derivations.sort(key =lambda pair: pair[1])
        return sentence_derivations[0][0]

    def _extract_logical_form(self, root: DerivationNode) -> LogicalForm:
        """
        Reads a LogicalForm off a completed derivation tree. This is a deliberately simple extraction (find the predicate, find the arguments, note negation/tense/quantification)
        not full compositional semantics; real beta-reduction over the tree's structure is core/semantics.py's job. What this step gets right, thanks to having a real tree instead
        of a flat token list: an attributive adjective (tagged N/N in the winning derivation) is never mistaken for the sentence's predicate ("the heavy suitcase is portable" correctly
        identifies "portable" alone as the predicate, not "heavy"), quantifiers are found by walking the tree rather than checking only whether the sentence starts with one (what makes
        more than one quantifier per sentence representable at all), and (as of adjunct clauses) a fronted adjunct's own words never leak into the matrix clause's predicate or
        arguments, because extraction runs over the matrix clause's own subtree specifically, not the whole tree's flattened leaves.
        """
        leaves = self._matrix_clause_leaves(root)

        # Quantifier words are already excluded here (they're in _FUNCTION_WORDS, the same closed set determiners belong to): the noun each one restricts is not, and remains a
        # normal content leaf, exactly like an ordinary determiner's noun would.
        content_leaves = [leaf for leaf in leaves if leaf.token not in _FUNCTION_WORDS]

        # A copula used as an identity/equality relation ("What IS the suitcase?", "Who IS John?": the inverted-copular-question pattern from Phase 2.6, though nothing restricts it to 
        # questions specifically: "John is a teacher" would hit this exact same path) is itself the sentence's predicate: unlike an ordinary adjectival copula, where the ADJECTIVE leaf 
        # carries the predicate-shaped category and the copula is just a semantically-transparent modifier. Because the copula is a function word, it's already gone from content_leaves 
        # by this point and would never be found by the ordinary predicate search below, so this has to be checked for first, the same priority CLAUSAL_VERB gets.
        copula_relation_leaf = next(
            (leaf for leaf in leaves if leaf.token in COPULA_PRESENT or leaf.token in COPULA_PAST if leaf.label == repr(TRANSITIVE_VERB)),
            None,
        )
        if copula_relation_leaf is not None:
            return LogicalForm(
                predicate = "IS",
                arguments = [leaf.token for leaf in content_leaves],
                is_negated = any(leaf.token in NEGATION_WORDS for leaf in leaves),
                tense = self._detect_tense(leaves),
            )

        # A clausal verb (e.g. "thinks" in "John thinks that Mary is home") always IS the matrix predicate whenever one is present: its complement is a full embedded sentence, which
        # necessarily contains its own predicate-shaped leaf (here, "home") later in this same flat leaf list. An ordinary rightmost-scan would find that embedded predicate instead and
        # mistake it for the matrix one, so a clausal-verb leaf is checked for and preferred first.
        predicate_index = None
        for i, leaf in enumerate(content_leaves):
            if leaf.label == repr(CLAUSAL_VERB):
                predicate_index = i
                break

        if predicate_index is None:
            for i in range(len(content_leaves) - 1, -1, -1):
                leaf = content_leaves[i]

                if leaf.label in (repr(PREDICATIVE_ADJECTIVE), repr(COMPARATIVE_PREDICATIVE)) or self._is_verb_leaf(leaf):
                    predicate_index = i
                    break

        if predicate_index is None:
            predicate_index = len(content_leaves) - 1

        predicate_leaf = content_leaves[predicate_index]

        if predicate_leaf.label == repr(CLAUSAL_VERB):
            # A clausal verb's complement is a full embedded sentence, not a flat argument list: handled entirely separately (including its OWN negation/tense, read from only its own content, 
            # not inherited from, or leaking into, the matrix clause's).
            return self._extract_clausal_form(root, predicate_leaf, content_leaves, predicate_index, leaves)

        is_negated = any(leaf.token in NEGATION_WORDS for leaf in leaves)
        tense = self._detect_tense(leaves)
        quantifier_store = self._collect_stored_quantifiers(root, leaves)

        argument_leaves = content_leaves[:predicate_index] + content_leaves[predicate_index + 1:]
        arguments = [leaf.token for leaf in argument_leaves if leaf.token not in COMPARISON_MARKERS]
        plural_arguments = [leaf.token for leaf in argument_leaves if self._is_plural_noun_leaf(leaf)]

        form = LogicalForm(
            predicate = predicate_leaf.token.upper(),
            arguments = arguments,
            is_negated = is_negated,
            tense = tense,
            plural_arguments = plural_arguments,
            is_passive = self._is_passive_use(predicate_leaf),
        )

        if quantifier_store:
            form.quantifier_store = quantifier_store
            if len(quantifier_store) == 1:
                # The single-quantifier shape core/semantics.py's existing evaluator already knows how to check: see core/types.py's LogicalForm docstring for why both
                # fields exist rather than just quantifier_store.
                sole = quantifier_store[0]
                form.quantifier_meta = {"operator": sole.operator, "variable": sole.bound_variable, "restrictor": sole.restrictor}

        predicate_lemma = self.lexicon.lemmatize(predicate_leaf.token) or predicate_leaf.token
        idiom_object = _IDIOM_OBJECT_TRIGGERS.get(predicate_lemma)
        if idiom_object and idiom_object in arguments:
            form.predicate = f"IDIOM:{predicate_lemma}_{idiom_object}"

        return form

    def _extract_clausal_form(
        self,
        root: DerivationNode,
        verb_leaf: DerivationNode,
        content_leaves: List[DerivationNode],
        predicate_index: int,
        matrix_scoped_leaves: List[DerivationNode],
    ) -> LogicalForm:
        """
        Builds a LogicalForm for a mental-predicate verb ("John thinks that Mary is home") whose complement is a full embedded sentence, not a flat argument list. 
        The subject is whatever content leaf(s) precede the verb. The complement clause itself is found via the verb leaf's sibling in the tree (built by forward application, 
        (S\\NP)/S + S -> S\\NP: the same parent-and-sibling pattern _collect_stored_quantifiers uses for a quantifier's restrictor) and recursively extracted as its OWN nested 
        LogicalForm via a fresh call to _extract_logical_form, which is what correctly gives the complement its own negation and tense, read only from its own content.

        The matrix clause's own negation/tense are computed from matrix_scoped_leaves MINUS whichever leaves belong to the complement (by identity, not by assuming negation sits in
        any one fixed position): deliberately not just "whatever comes before the verb", because negation for a clausal verb can attach to the whole "verb + complement" constituent
        ("John does NOT think that Mary is home" negates the matrix, not stated positionally adjacent to "think" alone) rather than sitting between subject and verb.
        """
        subject_leaves = content_leaves[:predicate_index]
        arguments: List[Any] = [leaf.token for leaf in subject_leaves]

        parent = find_parent(root, verb_leaf)
        complement_node = None
        if parent is not None:
            complement_node = next((child for child in parent.children if child is not verb_leaf), None)

        complement_leaf_ids = set()
        if complement_node is not None:
            embedded_form = self._extract_logical_form(complement_node)
            complement_leaf_ids = {id(leaf) for leaf in collect_leaves(complement_node)}
            arguments.append(embedded_form)

        matrix_only_leaves = [leaf for leaf in matrix_scoped_leaves if id(leaf) not in complement_leaf_ids]
        matrix_is_negated = any(leaf.token in NEGATION_WORDS for leaf in matrix_only_leaves)
        matrix_tense = self._detect_tense(matrix_only_leaves)

        return LogicalForm(
            predicate = verb_leaf.token.upper(),
            arguments = arguments,
            is_negated = matrix_is_negated,
            tense = matrix_tense,
        )

    def _matrix_clause_leaves(self, root: DerivationNode) -> List[DerivationNode]:
        """
        Returns every leaf in the tree EXCEPT those belonging to an adjunct clause: fronted ("Before he entered the room, John left") or trailing ("John left before he entered the
        room"). This walks the whole tree looking for a completed adjunct-phrase node (labeled as either SENTENCE_MODIFIER, "(S/S)", for the fronted case, or TRAILING_VP_MODIFIER,
        "((S\\NP)\\(S\\NP))", for the trailing one: see those categories' own definitions above for why the trailing case attaches at the VP rather than the whole sentence) and excludes
        every leaf under it, rather than only checking specific fixed tree positions. That generality matters: it means adjunct exclusion keeps working correctly regardless of
        exactly where composition or a particular combinator sequence ends up attaching the adjunct, rather than only for one specific expected tree shape. The adjunct clause's own
        content is still part of the FULL tree parse_with_derivation returns (needed whole for structural checks like cataphora's c-command test): only extraction excludes it.

        A TRAILING_VP_MODIFIER-labeled node is NOT automatically a genuine adjunct clause any more, though: a passive sentence's "by John" agent phrase (BY_AGENT, see its own definition above) 
        reuses this exact same VP-attachment shape, since that shape is just what "attaches after a VP, gives back a VP" categorially requires: there's no way to give it a different category 
        and still have it combine correctly. The two are told apart by content, not label: a genuine adjunct clause's subtree contains one of SUBORDINATORS; a by-agent phrase never does 
        (it's just "by" + an NP). Only the former is excluded here: the agent phrase's own NP is meant to survive as an ordinary argument (core/pipeline.py's _open_modal_attitude_context-style callers 
        aside, this is what lets "The suitcase was kicked by John" put "john" into the sentence's arguments at all).
        """
        excluded_leaf_ids = set()

        def _mark_excluded(node: DerivationNode) -> None:
            if node.label == repr(SENTENCE_MODIFIER):
                for leaf in collect_leaves(node):
                    excluded_leaf_ids.add(id(leaf))
                return # The whole fronted adjunct phrase is excluded: no need to look inside it further.

            if node.label == repr(TRAILING_VP_MODIFIER):
                subtree_leaves = collect_leaves(node)
                if any(leaf.token in SUBORDINATORS for leaf in subtree_leaves):
                    for leaf in subtree_leaves:
                        excluded_leaf_ids.add(id(leaf))
                    return # A genuine trailing adjunct clause: excluded, same as the fronted case.
                
                # Not a genuine adjunct (e.g. a by-agent phrase sharing the same VP-attachment shape): fall through and keep walking normally.

            for child in node.children:
                _mark_excluded(child)

        _mark_excluded(root)
        return [leaf for leaf in collect_leaves(root) if id(leaf) not in excluded_leaf_ids]

    def _collect_stored_quantifiers(self, root: DerivationNode, leaves: List[DerivationNode]) -> List[StoredQuantifier]:
        """
        Finds every quantifier word in the tree and, for each, the noun it combined with to form an NP (its restrictor): by looking at the quantifier leaf's parent node (built via
        forward application, NP/N + N -> NP) and taking whichever sibling isn't the quantifier itself. context_id stays "global" for now; giving each stored quantifier a context tied to
        an actual opened Modal & Attitude context is what the 5a/6d intensional-scope coupling needs, and that depends on clausal complements existing at all (a later grammar step):
        not something this step can meaningfully do yet.
        """
        stored: List[StoredQuantifier] = []
        for leaf in leaves:
            if leaf.token not in QUANTIFIERS:
                continue
            parent = find_parent(root, leaf)

            if parent is None:
                continue
            restrictor_leaf = next((child for child in parent.children if child is not leaf and child.token), None)

            if restrictor_leaf is None:
                continue
            
            stored.append(StoredQuantifier(
                operator = QUANTIFIERS[leaf.token],
                restrictor = restrictor_leaf.token.upper(),
                bound_variable = "x",
                context_id = "global",
            ))

        return stored

    def _is_plural_noun_leaf(self, leaf: DerivationNode) -> bool:
        if not leaf.token:
            return False
        entry = self.lexicon.get_word_definition(leaf.token)
        if not entry or entry.get("category") != "noun":
            return False
        
        return self.lexicon.detect_inflection(leaf.token) == "plural_or_third_person"

    def _is_verb_leaf(self, leaf: DerivationNode) -> bool:
        return leaf.label in (repr(INTRANSITIVE_VERB), repr(TRANSITIVE_VERB), repr(DITRANSITIVE_VERB), repr(CLAUSAL_VERB))

    def _is_passive_use(self, predicate_leaf: DerivationNode) -> bool:
        """
        True if `predicate_leaf` is an inherently transitive/ditransitive verb that combined in the winning derivation using the demoted, object-less
        INTRANSITIVE_VERB (S\\NP) shape rather than its own ordinary TRANSITIVE_VERB/DITRANSITIVE_VERB category: exactly the passive-voice reading (see
        supertag_content_word's own docstring on why that shape is offered as a fallback candidate at all). This is a purely post-hoc check against the
        WORD's own lexicon valency, not a separately-named category: PASSIVE_VERB would be structurally identical to INTRANSITIVE_VERB (same repr, "(S\\NP)")
        and so couldn't be told apart from it by label anyway, the same reason PREDICATIVE_ADJECTIVE and INTRANSITIVE_VERB already share one label today.
        """
        entry = self.lexicon.get_word_definition(predicate_leaf.token)
        if not entry or entry.get("category") != "verb":
            return False
        
        return entry.get("valency") in ("transitive", "ditransitive") and predicate_leaf.label == repr(INTRANSITIVE_VERB)

    def _detect_tense(self, leaves: List[DerivationNode]) -> str:
        tokens_present = {leaf.token for leaf in leaves}
        if tokens_present & (COPULA_PAST | PAST_AUX):
            return "past"
        if tokens_present & FUTURE_MARKERS:
            return "future"
        
        return "present"
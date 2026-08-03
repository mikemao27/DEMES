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
quantified noun phrases, negation, tense marking, transitive/intransitive verbs, and attributive/predicative/comparative adjectives. It does not yet handle coordination
("and"/"or"), embedded clauses, passive voice, or wh-questions: those need more category assignments and, in a couple of cases, combinators (type-raising) 
that are implemented and tested here but not yet wired into the chart's automatic search, to keep the search space from exploding
before there's a real need to. Extending coverage later means adding more category data, not rewriting this engine.

This file also does not resolve pronouns or decide what a sentence's truth value is: producing the correct sentence STRUCTURE is this file's job; resolving 
what "it" refers to (core/discourse.py) and evaluating whether the sentence is true (core/semantics.py, core/world_model.py) are separate, later jobs 
that consume this file's output.
"""

from typing import List, Dict, Optional, Tuple, Any

from core.types import LogicalForm, DerivationNode, Gender, GrammaticalNumber

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

_COMBINATOR_RULES: Tuple[Tuple[str, Any], ...] = (
    (">", try_forward_application),
    ("<", try_backward_application),
    (">B", try_forward_composition),
    ("<B", try_backward_composition),
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

_DETERMINER_LIKE = set(QUANTIFIERS) | DETERMINERS
_PREDICATE_MODIFIER_WORDS = COPULA_PRESENT | COPULA_PAST | NEGATION_WORDS | FUTURE_MARKERS | PAST_AUX
_FUNCTION_WORDS = _DETERMINER_LIKE | _PREDICATE_MODIFIER_WORDS | COMPARISON_MARKERS | FOCUS_PARTICLES

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
        if valency == "transitive":
            return [TRANSITIVE_VERB]
        if valency == "ditransitive":
            return [DITRANSITIVE_VERB]
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
        Splits a clean sentence into lowercase words, stripping leading/trailing punctuation.
        """
        clean_sentence = sentence.strip("?.!")
        return [word.lower() for word in clean_sentence.split()]

    def parse(self, sentence: str) -> Optional[LogicalForm]:
        """
        Parses a sentence end to end and returns just its LogicalForm: the entry point every existing caller and test uses. See parse_with_derivation() for the 
        variant that also returns the derivation tree itself, needed by later layers (core/discourse.py's cataphora resolution) that need real sentence structure, 
        not just the extracted result.
        """
        logical_form, _root = self.parse_with_derivation(sentence)
        return logical_form

    def parse_with_derivation(self, sentence: str) -> Tuple[Optional[LogicalForm], Optional[DerivationNode]]:
        """
        Same end-to-end process as parse(), but also returns the winning derivation tree rather than discarding it. Returns (None, None) if the sentence contains a word 
        the lexicon can't find at all, if no combination of categories produces a complete sentence, or if an NPI licensing check fails.
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

        return self._extract_logical_form(tokens, root), root

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
                            for _name, rule in _COMBINATOR_RULES:
                                result = rule(left_category, right_category)

                                if result is None:
                                    continue

                                node = DerivationNode(
                                    label = repr(result),
                                    children = (left_node, right_node),
                                    span = (i, j),
                                )
                                cell.append((result, node, left_rank + right_rank))

                chart[(i, j)] = cell

        final_cell = chart.get((0, n), [])
        sentence_derivations = [(node, score) for category, node, score in final_cell if category == S]
        if not sentence_derivations:
            return None

        sentence_derivations.sort(key =lambda pair: pair[1])
        return sentence_derivations[0][0]

    def _extract_logical_form(self, tokens: List[str], root: DerivationNode) -> LogicalForm:
        """
        Reads a LogicalForm off a completed derivation tree. This is a deliberately simple extraction (find the predicate, find the arguments, note negation/tense/quantification)
        not full compositional semantics; real beta-reduction over the tree's structure is core/semantics.py's job, arriving next. What this step already gets right, thanks to
        having a real tree instead of a flat token list: an attributive adjective (tagged N/N in the winning derivation) is never mistaken for the sentence's predicate, because its label
        won't match the predicative category check below: "the heavy suitcase is portable" now correctly identifies "portable" alone as the predicate, not "heavy".
        """
        leaves = collect_leaves(root)
        is_negated = any(leaf.token in NEGATION_WORDS for leaf in leaves)
        tense = self._detect_tense(leaves)

        content_leaves = [leaf for leaf in leaves if leaf.token not in _FUNCTION_WORDS]

        first_token = tokens[0]
        if first_token in QUANTIFIERS:
            return self._build_quantified_form(tokens, first_token, is_negated, tense)

        predicate_index = None
        for i in range(len(content_leaves) - 1, -1, -1):
            leaf = content_leaves[i]

            if leaf.label in (repr(PREDICATIVE_ADJECTIVE), repr(COMPARATIVE_PREDICATIVE)) or self._is_verb_leaf(leaf):
                predicate_index = i
                break

        if predicate_index is None:
            predicate_index = len(content_leaves) - 1

        predicate_leaf = content_leaves[predicate_index]
        argument_leaves = content_leaves[:predicate_index] + content_leaves[predicate_index + 1:]
        arguments = [leaf.token for leaf in argument_leaves if leaf.token not in COMPARISON_MARKERS]

        form = LogicalForm(
            predicate = predicate_leaf.token.upper(),
            arguments = arguments,
            is_negated = is_negated,
            tense = tense,
        )

        predicate_lemma = self.lexicon.lemmatize(predicate_leaf.token) or predicate_leaf.token
        idiom_object = _IDIOM_OBJECT_TRIGGERS.get(predicate_lemma)
        if idiom_object and idiom_object in arguments:
            form.predicate = f"IDIOM:{predicate_lemma}_{idiom_object}"

        return form

    def _is_verb_leaf(self, leaf: DerivationNode) -> bool:
        return leaf.label in (repr(INTRANSITIVE_VERB), repr(TRANSITIVE_VERB), repr(DITRANSITIVE_VERB))

    def _detect_tense(self, leaves: List[DerivationNode]) -> str:
        tokens_present = {leaf.token for leaf in leaves}
        if tokens_present & (COPULA_PAST | PAST_AUX):
            return "past"
        if tokens_present & FUTURE_MARKERS:
            return "future"
        return "present"

    def _build_quantified_form(self, tokens: List[str], quantifier_word: str, is_negated: bool, tense: str) -> LogicalForm:
        """
        Quantified noun phrases ("every suitcase is portable") are handled positionally rather than purely tree-derived, matching the shape Cooper Storage (core/semantics.py, not yet
        built) will need to consume: head_noun is the restrictor, predicate_word is what's claimed about every member of it.
        """
        head_noun = tokens[1] if len(tokens) > 1 else "entity"
        predicate_word = tokens[-1] if len(tokens) > 2 else "state"

        form = LogicalForm(
            predicate = predicate_word.upper(),
            arguments = [head_noun],
            is_negated = is_negated,
            tense = tense,
        )
        form.quantifier_meta = {
            "operator": QUANTIFIERS[quantifier_word],
            "variable": "x",
            "restrictor": head_noun.upper(),
        }
        return form
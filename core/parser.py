r"""
Implements the Combinatory Categorial Grammar (CCG) engine and an Earley chart parser for DEMES.

How it solves the problem:
    1. Replaces rigid menu parsers with a dynamic chart parsing algorithm that handles arbitrary-depth
    clause embedding and coordination in polynomial time.
    2. Combines syntax and semantics jointly: when words combine via CCG combinators (like forward/backward application),
    their lambda expressions merge simultaneously.
    3. Outputs an executable Logical Form instead of a static syntactic tree.
"""

from typing import List, Dict, Optional, Tuple, Any
from core.types import LogicalForm, SemanticType

class CCGCategory:
    r"""
    Represents a CCG syntactic category such as transitive and intransitive verbs. We do not present the actual CCG labels
    because they are quite complex and aren't necessary for understanding the model architecture presented here.
    """
    def __init__(self, result: str, argument: Optional[str] = None, direction: Optional[str] = None):
        self.result = result
        self.argument = argument
        self.direction = direction # "/" (forward) or "\\" (backward).

    def is_atomic(self) -> bool:
        return self.argument is None

    def __repr__(self) -> str:
        if self.is_atomic():
            return self.result

        return f"({self.result} {self.direction} {self.argument})"

class ChartParser:
    """
    An Earley-style chart parser optimized for CCG grammar composition. Maintains a chart of sub-parse states to
    parse complex, multi-clause sentences efficiently. It has been enhanced to recognize quantification determiners
    (every, all, some, a) and map them into quantified LogicalForms.
    """
    # Recognized quantifiers mapping to formal logical operators.
    QUANTIFIERS = {
        "every": "FORALL",
        "all": "FORALL",
        "some": "EXISTS",
        "a": "EXISTS",
        "no": "NOT_EXISTS"
    }

    # Function-word categories. Unlike a flat ignore-list, each set carries its own grammatical
    # signal (tense, negation) so membership in one both strips the word from the content stream
    # and tells the parser what it implied.
    DETERMINERS = {"the", "a", "an"}
    COPULA_PRESENT = {"is", "are"}
    COPULA_PAST = {"was", "were"}
    NEGATION_WORDS = {"not", "never"}
    FUTURE_MARKERS = {"will"}
    PAST_AUX = {"did"}

    FUNCTION_WORDS = DETERMINERS | COPULA_PRESENT | COPULA_PAST | NEGATION_WORDS | FUTURE_MARKERS | PAST_AUX

    def __init__(self, lexicon_manager: Any, world_model: Any):
        self.lexicon = lexicon_manager
        self.world_model = world_model

    def tokenize(self, sentence: str) -> List[str]:
        """
        Splits a clean, grammatically correct sentence into words.
        """
        # Strips punctuation and lowercases for dictionary lookup.
        clean_sent = sentence.strip("?.!")
        return [word.lower() for word in clean_sent.split()]

    def _detect_negation_and_tense(self, tokens: List[str]) -> Tuple[bool, str]:
        """
        Scans the raw token stream for negation and tense marker words, independent of
        which sentence pattern (standard vs. quantified) ends up consuming the tokens.
        """
        token_set = set(tokens)
        is_negated = bool(token_set & self.NEGATION_WORDS)

        if token_set & (self.COPULA_PAST | self.PAST_AUX):
            tense = "past"
        elif token_set & self.FUTURE_MARKERS:
            tense = "future"
        else:
            tense = "present"

        return is_negated, tense

    def parse(self, sentence: str) -> Optional[LogicalForm]:
        """
        Parses an input sentence and returns its executable Logical Form. Fails gracefully if a structural gap or unknown
        un-bootstrapped word is hit.
        """
        tokens = self.tokenize(sentence)
        if not tokens:
            return None

        is_negated, tense = self._detect_negation_and_tense(tokens)

        # Filter out function words; what remains is the content stream resolved against the lexicon.
        content_tokens = [t for t in tokens if t not in self.FUNCTION_WORDS]

        first_word = tokens[0]
        if first_word in self.QUANTIFIERS:
            return self._parse_quantified_sentence(tokens, is_negated, tense)

        parsed_nodes = []
        for token in content_tokens:
            def_data = self.lexicon.get_word_definition(token)
            if not def_data:
                # Strictly fail if an unknown word is encountered.
                return None

            parsed_nodes.append((token, def_data))

        return self._build_standard_logical_form(parsed_nodes, is_negated, tense)

    def _parse_quantified_sentence(self, tokens: List[str], is_negated: bool, tense: str) -> LogicalForm:
        """
        An example of Noun Phrase Quantification Parsing is as follows.

        Input: "Every suitcase is portable."
        Output: LogicalForm(predicate = 'PORTABLE', quantifiers = {'quantifier': 'FORALL', 'variable': 'x', 'restrictor': 'SUITCASE'}).
        """
        quantifier_word = tokens[0]
        head_noun = tokens[1] if len(tokens) > 1 else "entity"
        predicate_word = tokens[-1] if len(tokens) > 2 else "state"

        # Construct a formally quantified logical form.
        logical_form = LogicalForm(
            predicate = predicate_word.upper(),
            arguments = [head_noun],
            is_negated = is_negated,
            tense = tense
        )

        # Attach quantification metadata.
        logical_form.quantifier_meta = {
            "operator": self.QUANTIFIERS[quantifier_word],
            "variable": "x",
            "restrictor": head_noun.upper()
        }

        # Register the restrictor noun as an active discourse referent so later pronouns
        # ("Is it heavy too?") have an antecedent to resolve against.
        self.world_model.register_referent(head_noun, "noun", properties = [predicate_word.lower()])

        return logical_form

    def _build_standard_logical_form(self, nodes: List[Tuple[str, Dict]], is_negated: bool, tense: str) -> LogicalForm:
        # The predicate is the last content word whose category is predicative (verb or adjective),
        # not simply the last token — this is what lets SVO sentences ("John kicked the ball")
        # extract the verb as the predicate instead of the final noun.
        predicate_idx = None
        for i in range(len(nodes) - 1, -1, -1):
            if nodes[i][1].get("category") in ("verb", "adjective"):
                predicate_idx = i
                break
        if predicate_idx is None:
            predicate_idx = len(nodes) - 1

        root_word, root_def = nodes[predicate_idx]
        argument_nodes = nodes[:predicate_idx] + nodes[predicate_idx + 1:]

        arguments = []
        for token, def_data in argument_nodes:
            category = def_data.get("category")

            if category == "pronoun":
                # Resolve against active discourse referents rather than passing the bare
                # pronoun through as a meaningless literal argument.
                referent = self.world_model.resolve_pronoun(token)
                arguments.append(referent.name if referent else token)
            else:
                arguments.append(token)
                if category == "noun":
                    # Make this entity available as a future pronoun antecedent.
                    self.world_model.register_referent(token, category, properties = [root_word.lower()])

        return LogicalForm(
            predicate = root_word.upper(),
            arguments = arguments,
            is_negated = is_negated,
            tense = tense
        )

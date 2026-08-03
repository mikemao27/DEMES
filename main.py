"""
The main terminal REPL (Read-Eval-Print Loop) for DEMES: where every other module actually gets wired together into a 
running conversation.

How everything fits together:
    1. User Input -> DEMESPipeline (core/pipeline.py): parses the sentence (core/parser.py) against the lexicon (core/lexicon.py), 
    resolves pronouns and registers new entities (core/discourse.py, core/world_model.py), disambiguates ambiguous words, classifies 
    the speech act, and evaluates truth (core/semantics.py): all in one process_utterance() call.

    2. Recovery: if parsing fails because of a word the lexicon doesn't recognize, this file tries two escalating, honest fixes before giving up; 
    bounded spelling correction (fully local), then (only if a local model is actually loaded) proposing a closed-primitive definition for a
    genuinely new word and re-parsing. Neither step pretends to work when no model is present.
    
    3. Evaluation Payload -> LocalStylist (interface/stylist.py): builds the actual response sentence symbolically from the LogicalForm first, then 
    (if a model is loaded) lets it polish that sentence for warmth. The model can make DEMES sound better; it cannot make DEMES wrong.
    
    4. Transparency View: the terminal displays the raw logical breakdown FIRST, followed by the styled response, so the engine's internal 
    understanding is never hidden behind its phrasing.

This file also owns the one local model instance DEMES ever loads: interface/stylist.py's LocalStylist loads it (or gracefully doesn't, 
if no weights are on disk), and that same instance is handed to interface/neural_bridge.py's NeuralBridge rather than loading the weights a second time.
"""

import sys
from typing import Any, Dict, List, Optional

from core.pipeline import DEMESPipeline
from core.parser import supertag_function_word
from interface.stylist import LocalStylist
from interface.neural_bridge import NeuralBridge, suggest_spelling_correction

# Terminal Color Setup: color escape codes are disabled automatically if stdout is piped to a file or non-TTY (non-terminal).
_IS_TTY = sys.stdout.isatty()

def _code(seq: str) -> str:
    return seq if _IS_TTY else ""

RESET = _code("\033[0m")
BOLD = _code("\033[1m")
DIM = _code("\033[2m")
CYAN = _code("\033[36m")
GREEN = _code("\033[32m")
YELLOW = _code("\033[33m")
MAGENTA = _code("\033[35m")
RED = _code("\033[31m")

INDENT = "  "
MAX_CONVERSATION_TURNS = 20
EXIT_COMMANDS = {"exit", "quit"}

def print_banner() -> None:
    """
    Displays the welcome banner and system architectural mode.
    """
    print(f"\n{BOLD}{MAGENTA}======================================================================{RESET}")
    print(f"{BOLD}{CYAN} DEMES {RESET}{DIM}— Deconstructive Meaning Encoding & Expressive Syntax{RESET}")
    print(f"\n{BOLD}{MAGENTA}======================================================================{RESET}")
    print(f"{DIM}Neuro-Symbolic NLU Engine | Closed-Vocabulary Symbolic Core + Optional Local Model{RESET}")
    print(f"{DIM}Type 'exit' or 'quit' to end the session.{RESET}\n")

def print_transparency_view(logical_form: Any, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Prints an explicit, color-coded inspection view of what DEMES structurally understood from the utterance BEFORE displaying 
    the styled output: including, when present, the secondary signals (word senses resolved, presuppositions accommodated, implicatures noted) 
    that don't change the headline truth value but are still part of what DEMES understood.
    """
    print(f"{DIM}[DEMES Core Inspection]{RESET}")
    print(f"{INDENT}{CYAN}Speech Act{RESET}: {DIM}{result.get('speech_act')}{RESET}")

    if logical_form:
        print(f"{INDENT}{CYAN}Predicate{RESET}: {BOLD}{logical_form.predicate}{RESET}")
        print(f"{INDENT}{CYAN}Arguments{RESET}: {DIM}{logical_form.arguments}{RESET}")

        truth = payload.get("truth_value", False)
        truth_str = f"{GREEN}Validated (True){RESET}" if truth else f"{RED}Conflict / Unmapped (False){RESET}"
        print(f"{INDENT}{CYAN}Truth Val{RESET}: {truth_str}")

        if payload.get("idiom_meaning"):
            print(f"{INDENT}{CYAN}Idiom{RESET}: {DIM}{payload['idiom_meaning']}{RESET}")
    else:
        reason = payload.get("reason", "Unknown parse error")
        print(f"{INDENT}{RED}Parse Failure{RESET} : {DIM}{reason}{RESET}")

    if result.get("word_senses"):
        print(f"{INDENT}{CYAN}Word Senses{RESET}: {DIM}{result['word_senses']}{RESET}")
    if result.get("accommodated_presuppositions"):
        print(f"{INDENT}{CYAN}Accommodated{RESET}: {DIM}{result['accommodated_presuppositions']}{RESET}")
    if result.get("implicatures"):
        print(f"{INDENT}{CYAN}Implicatures{RESET}: {DIM}{result['implicatures']}{RESET}")
    if result.get("focus"):
        print(f"{INDENT}{CYAN}Focus{RESET}: {DIM}{result['focus']}{RESET}")

def find_unrecognized_tokens(user_input: str, pipeline: DEMESPipeline) -> List[str]:
    """
    Identifies which tokens in a failed sentence are genuinely unrecognized: not a closed-class function word (determiner, negation, pronoun, ...) 
    and not found anywhere in the lexicon (permanent or provisional, inflection-resolved). A parse can also fail for purely structural reasons with 
    every word recognized; in that case this returns an empty list, and no recovery is attempted.
    """
    tokens = pipeline.parser.tokenize(user_input)
    unrecognized = []
    for token in tokens:
        if supertag_function_word(token):
            continue
        if pipeline.lexicon.get_word_definition(token) is not None:
            continue
        unrecognized.append(token)

    return unrecognized

def attempt_recovery(user_input: str, pipeline: DEMESPipeline, neural_bridge: NeuralBridge) -> Optional[str]:
    """
    When a sentence fails to parse, tries two escalating, honest recovery paths, cheapest first: bounded local spelling correction, then 
    (only if a model is actually loaded) proposing a closed-primitive definition for a genuinely novel word via NeuralBridge.induce_word (which
    itself refuses anything outside the closed vocabulary; see interface/neural_bridge.py). Returns a short user-facing note describing what was found, or None 
    if nothing was attempted or nothing helped.
    """
    unrecognized = find_unrecognized_tokens(user_input, pipeline)
    if not unrecognized:
        return None

    notes = []
    for token in unrecognized:
        suggestion = suggest_spelling_correction(token, pipeline.lexicon)
        if suggestion and suggestion != token:
            notes.append(f'"{token}" not recognized, did you mean "{suggestion}"?')
            continue

        induced = neural_bridge.induce_word(token, user_input, pipeline.lexicon)
        if induced:
            primitive_names = ", ".join(p["name"] for p in induced["primitives"])
            notes.append(f'"{token}" was not in the dictionary, provisionally learned it as a {induced["category"]} ({primitive_names}).')

    return "\n".join(notes) if notes else None

def check_indirect_speech_act(user_input: str, result: Dict[str, Any], neural_bridge: NeuralBridge) -> Optional[Dict[str, str]]:
    """
    Only attempted when a model is loaded and the utterance was classified as a plain assertive statement (a question or command already states its intent 
    directly: no hidden reading to look for), per the architecture's lazy-execution principle. Returned as an additional signal alongside the literal 
    parse, never in place of it.
    """
    if not neural_bridge.llm or result.get("speech_act") != "assertive":
        return None
    
    return neural_bridge.infer_indirect_speech_act(user_input)

def main() -> None:
    """
    Main execution loop for the DEMES terminal chatbot.
    """
    # 1. Initialize the symbolic core pipeline.
    pipeline = DEMESPipeline()

    # 2. Initialize the presentation layer (loads the one local model instance, if any weights exist).
    stylist = LocalStylist(pipeline.lexicon)

    # 3. Share that same model instance with the neural bridge - never load the weights twice.
    neural_bridge = NeuralBridge(llm=stylist.llm)

    print_banner()
    turn_count = 0

    while True:
        try:
            user_input = input(f"{BOLD}User > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Session closed.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print(f"{DIM}Exiting DEMES. Goodbye!{RESET}")
            break

        turn_count += 1
        print()

        # Step 1: Run the full NLU pipeline.
        result = pipeline.process_utterance(user_input)
        logical_form = result["logical_form"]
        payload = result["semantics"]

        # Step 2: If parsing failed, try honest recovery, then re-run the pipeline once.
        if logical_form is None:
            recovery_note = attempt_recovery(user_input, pipeline, neural_bridge)
            if recovery_note:
                print(f"{YELLOW}[DEMES Recovery]{RESET} {recovery_note}\n")
                result = pipeline.process_utterance(user_input)
                logical_form = result["logical_form"]
                payload = result["semantics"]

        # Step 3: Print transparency inspection of internal understanding.
        print_transparency_view(logical_form, payload, result)

        indirect = check_indirect_speech_act(user_input, result, neural_bridge)
        if indirect:
            print(f"{INDENT}{YELLOW}Possible indirect request{RESET}: {DIM}{indirect['action']} {indirect['target']}{RESET}")

        # Step 4: Pass structured payload (and the LogicalForm, for symbolic realization) to the presentation layer.
        rendered_response = stylist.render(payload, logical_form)
        print(f"\n{BOLD}{GREEN}DEMES > {RESET}{rendered_response}\n")

        # Step 5: Check 20-message turn constraint for discourse clarity.
        if turn_count >= MAX_CONVERSATION_TURNS:
            print(f"{YELLOW}[DEMES Notice] Reached {MAX_CONVERSATION_TURNS}-turn limit. Clearing active discourse referents.{RESET}\n")
            pipeline.world_model.clear_discourse()
            turn_count = 0

if __name__ == "__main__":
    main()
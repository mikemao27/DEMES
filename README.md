<div align="center"> 

## Deconstructive Encoding of Meaning and Expressive Syntax

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Neuro--Symbolic-blue?style=flat-square" alt="Architecture">
  <img src="https://img.shields.io/badge/Core-Primitive%20Logic-green?style=flat-square" alt="Core">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" alt="License">
</p>

*A local-first, truth-conditional natural language understanding engine powered by symbolic primitives and a lightweight neural presentation layer.*

</div>

**DEMES** is a project with the aim of developing syntax, semantics, and meaning encoded **natural language processing (NLP)** for artificial-intelligence models; with the aim of solving an age-old problem dominated by auto-regressive algorithms. This means that we aim to develop a system that has awareness of the syntax, semantics, and meaning of words and phrases in generative **natural language processing** rather than settling for auto-regressive probability based token-prediction algorithms. We note that the ideas behind DEMES may not be entirely new due to past ideas of NSM primitives, but they are a paradigm shift in today's culture of auto-regressive models that require immense amounts of compute and data. 

There are a few caveats when it comes to DEMES' functionality. To assist with textual parsing, we assume that all inputted text will be free of spelling or grammatical issues and that no made-up words like "brillig" will be present. This restricts the vocabulary to correctly spelled English words and keeps the lexicon clean (preventing runaway dictionary blot from literary nonsense). We do use auto-regressive LLMs, but not for reasoning. Free, open-source models are only used locally in the input and output stages to prevent rigid, cold model outputs and instead provide warm, stylistic responses to user queries. This does not effect the philosophical rigor of the model as meaning and language understanding is still effectively encoded through primitives.

*We aim for the DEMES model to serve as the foundation for the next phase of artificial intelligence models and their natural language processing capabilities. On their own, the DEMES model, in our opinion, poses no substantial ethical risks are are safe to be openly distributed to the world. We merely aim to pose a new paradigm for natural language processing (NLP) and natural language understanding (NLU)*

> [!IMPORTANT]
> The model included in this repository is restricted to natural language processing and should not be assumed to have any capabilities in domains such as mathematics, physics, software development, chemistry, etc. All technology is open-source with an Apache 2.0 License.

## Further Exploration
This project draws inspiration from a variety of sources in artificial intelligence research, neuroscience, and other domains. For those curious about the ideas behind our models, we provide resources and write-ups in our resources folder. We hope that interested parties check out the papers that built each of our models as well as our formalizations of the mathematics behind it all. For ease of identification, we provide the paper names and their publication dates in the file signature.

## Contacts
Feel free to reach out with questions or collaboration ideas. While we do not explore it explicitly here, we hope that the presence of such technology may encourage the development of linguistic models that may revive certain classical languages like Latin that may bee cast aside as "dead", but have immense literary and societal impact.

## Citation
If you find this project useful, please give it a star and cite it via [**GitHub**](https://github.com/mikemao27/DEMES). See `LICENSE` (Apache 2.0) for terms of use and attribution. We provide a sample **bibtex** citation blurb below for ease of usage. The software is free to use with attribution. The issue of natural language understanding is extremely complex, even with the current auto-regressive algorithms (which do not capture any sense of linguistic meaning) or primitive encoded language (which is brittle when it comes to spelling errors or ambiguous meanings like idioms). We hope that others may find this useful and build upon our software.

```bibtex
@software{DEMES,
  author = {Mao, Mike},
  title = {DEMES: Deconstructive Encoding of Meaning and Expressive Syntax},
  year = {2026},
  url = {https://github.com/mikemao27/DEMES},
  version = {1.0.0}
}
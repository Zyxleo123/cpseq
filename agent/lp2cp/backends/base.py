"""The backend seam. What a real solver stack would have to implement.

The design doc's division of labour gives the backend residue identities
(ProteinMPNN / BoltzGen / Rosetta packer), backbone coordinates (GenKIC /
BoltzGen) and scores (Rosetta / Boltz-2 / geometry checks), with one
constraint on the last of those: scores "must be LLM-independent or the loop is
self-grading".

This protocol is deliberately two methods wide. Everything above it -- the
agent, the prompt scaffold, the execution service, the scorer, the figures --
is written against these two signatures only, so replacing the simulator with
GenKIC + ProteinMPNN + Boltz-2 is a new subclass and nothing else. That is the
whole reason the end-to-end agent can be built and measured before the real
stack is wired in.

A real backend would implement:

    propose(compiled, current_seq)  ->  run GenKIC over compiled.spec's pivots,
                                        ProteinMPNN over its open mask, return
                                        whatever closed and packed
    score(compiled, cand, n_seeds)  ->  Boltz-2 at n_seeds, Rosetta interface
                                        scoring, strain decomposition, free
                                        peptide conformer sampling

and would be free to return fewer candidates than asked, to return candidates
that violate the mask (which the service catches), and to fail outright on a
refolding error -- all three of which the simulator does on purpose.
"""

from __future__ import annotations

from typing import Protocol

from ..types import Candidate, ScoreCard


class Backend(Protocol):
    """Emits molecules inside an LLM-chosen mask, and scores them."""

    name: str
    version: str

    def propose(self, compiled, current_seq: str) -> list[Candidate]:
        """Candidates for one compiled design problem. May be empty."""
        ...

    def score(self, compiled, cand: Candidate, n_seeds: int) -> ScoreCard:
        """Evaluate one candidate at a given evidence depth."""
        ...

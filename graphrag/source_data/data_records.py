"""Small records shared by the whole system: a side effect and a piece of evidence."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SideEffect:
    # Decagon side-effect code, for example "C0020538".
    code: str
    name: str


@dataclass(frozen=True)
class Evidence:
    """One numbered path in the graph that supports a statement about a drug pair.

    path is a list of edges {"source", "relation", "target", ...} where relation
    is "targets" (drug -> protein), "interacts_with" (protein - protein) or
    "has_side_effect" (drug pair -> side effect).
    """

    # "E1", "E2", ... in the order sent to the LLM.
    evidence_id: str
    tier: int
    # "known_pairwise_effect" (Tier 1), "shared_target_amplification" (Tier 2)
    # or "novel_pair_protein_bridge" (Tier 3).
    kind: str
    drug_pair: tuple[str, str]
    path: tuple[dict[str, str], ...]
    side_effect: str | None = None
    side_effect_code: str | None = None
    # Tier 3 only: side-effect names the LLM may use for this pair.
    candidate_side_effects: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "evidence_id": self.evidence_id,
            "tier": self.tier,
            "kind": self.kind,
            "drug_pair": list(self.drug_pair),
            "path": list(self.path),
        }
        if self.side_effect is not None:
            result["side_effect"] = self.side_effect
        if self.side_effect_code is not None:
            result["side_effect_code"] = self.side_effect_code
        if self.candidate_side_effects:
            result["candidate_side_effects"] = list(self.candidate_side_effects)
        return result

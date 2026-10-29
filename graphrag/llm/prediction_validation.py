"""Keep only LLM predictions whose cited evidence exists, has the right tier and supports the label.

Rejection reasons:
- unknown_evidence: an evidence ID that was not sent
- tier_mismatch: the evidence belongs to another tier
- invalid_path: an edge of the path is not in the raw data
- no_evidence: no evidence ID left
- label_mismatch: Tier 1/2 evidence is about another side effect
- unknown_label: the Tier 3 side effect is not a Decagon side effect
- label_not_candidate: the Tier 3 side effect is not a candidate of every cited bridge
"""

from dataclasses import dataclass, field

from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.knowledge_graph.evidence_path_validation import evidence_path_exists_in_graph
from graphrag.llm.prediction_data_models import ModelPrediction
from graphrag.source_data.data_records import Evidence


@dataclass
class PredictionCheck:
    supporting_evidence: list[Evidence] = field(default_factory=list)
    # Side-effect name as written in the data.
    side_effect: str = ""
    # Empty when the prediction is accepted.
    rejection_reason: str = ""


def check_prediction_at_tier(
    prediction: ModelPrediction,
    tier: int,
    evidence_ids: list[str],
    evidence_by_id: dict[str, Evidence],
    graph: DecagonGraph,
) -> PredictionCheck:
    unique_evidence_ids: list[str] = []
    for evidence_id in evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            return PredictionCheck(rejection_reason="unknown_evidence")
        if evidence.tier != tier:
            return PredictionCheck(rejection_reason="tier_mismatch")
        if not evidence_path_exists_in_graph(graph, evidence):
            return PredictionCheck(rejection_reason="invalid_path")
        if evidence_id not in unique_evidence_ids:
            unique_evidence_ids.append(evidence_id)
    if not unique_evidence_ids:
        return PredictionCheck(rejection_reason="no_evidence")

    cited_evidence = [evidence_by_id[evidence_id] for evidence_id in unique_evidence_ids]
    predicted_label = prediction.side_effect.casefold().strip()
    if tier in (1, 2):
        for evidence in cited_evidence:
            if (evidence.side_effect or "").casefold().strip() != predicted_label:
                return PredictionCheck(rejection_reason="label_mismatch")
        return PredictionCheck(cited_evidence, cited_evidence[0].side_effect or "")

    canonical_name = graph.canonical_side_effect_names.get(predicted_label)
    if canonical_name is None:
        return PredictionCheck(rejection_reason="unknown_label")
    # Every cited bridge must list the label among its candidates; a bridge
    # without candidates supports no label at all.
    for evidence in cited_evidence:
        candidate_labels = {name.casefold() for name in evidence.candidate_side_effects}
        if predicted_label not in candidate_labels:
            return PredictionCheck(rejection_reason="label_not_candidate")
    return PredictionCheck(cited_evidence, canonical_name)


def validate_predictions(
    predictions: list[ModelPrediction],
    evidence_items: list[Evidence],
    graph: DecagonGraph,
    rejections: list[dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    """Keep only claims whose cited evidence re-checks against the raw data.

    A failed Tier 2 claim is downgraded to Tier 1 when the evidence it cited
    includes documented Tier 1 items for the same label; nothing new is added.
    Rejected claims are appended to rejections when it is given.
    """
    evidence_by_id = {evidence.evidence_id: evidence for evidence in evidence_items}
    validated: list[dict[str, object]] = []
    seen_keys: set[tuple[int, str, tuple[str, ...]]] = set()
    for prediction in predictions:
        tier = prediction.tier
        check = check_prediction_at_tier(
            prediction, tier, prediction.evidence_ids, evidence_by_id, graph
        )
        downgraded_from = None
        if check.rejection_reason and tier == 2:
            tier1_evidence_ids = [
                evidence_id
                for evidence_id in prediction.evidence_ids
                if evidence_id in evidence_by_id and evidence_by_id[evidence_id].tier == 1
            ]
            if tier1_evidence_ids:
                tier1_check = check_prediction_at_tier(
                    prediction, 1, tier1_evidence_ids, evidence_by_id, graph
                )
                # The original rejection reason is kept when Tier 1 fails too.
                check.supporting_evidence = tier1_check.supporting_evidence
                check.side_effect = tier1_check.side_effect
                if not tier1_check.rejection_reason:
                    downgraded_from = 2
                    tier = 1
                    check.rejection_reason = ""
        if check.rejection_reason:
            if rejections is not None:
                rejections.append(
                    {
                        "side_effect": prediction.side_effect,
                        "tier": str(prediction.tier),
                        "reason": check.rejection_reason,
                    }
                )
            continue

        cited_ids = tuple(sorted(evidence.evidence_id for evidence in check.supporting_evidence))
        duplicate_key = (tier, check.side_effect.casefold(), cited_ids)
        if duplicate_key in seen_keys:
            continue
        seen_keys.add(duplicate_key)
        result: dict[str, object] = {
            "side_effect": check.side_effect,
            "tier": tier,
            "hypothesis": tier == 3,
            "explanation": prediction.explanation.strip(),
            "evidence": [evidence.to_dict() for evidence in check.supporting_evidence],
        }
        if downgraded_from is not None:
            result["downgraded_from"] = downgraded_from
        validated.append(result)
    return validated

"""End-to-end Graph-RAG pipeline, called directly by the Streamlit app.

    system = PolypharmacySystem.load()
    result = system.predict(["Panadol", "ibuprofen", "warfarin"])

One predict() call = resolve names -> retrieve and rank the subgraph -> one
LLM request with all evidence as JSON -> validate -> readable result. If the
LLM is not configured or fails, the result uses template explanations.
"""

from dataclasses import dataclass

import httpx

from graphrag.drug_names.drug_name_resolver import DrugResolver
from graphrag.drug_names.name_index import NameIndex
from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.knowledge_graph.regimen_evidence_retrieval import retrieve_regimen_evidence
from graphrag.llm.prediction_data_models import LLMConfigurationError, LLMResponseError
from graphrag.llm.prediction_request import generate_predictions
from graphrag.llm.prediction_validation import validate_predictions
from graphrag.pipeline.regimen_error import RegimenError
from graphrag.result_presentation.coverage_summary import summarize_data_coverage
from graphrag.result_presentation.drug_information_cards import build_drug_cards
from graphrag.result_presentation.interaction_cards import build_interaction_cards
from graphrag.result_presentation.llm_context import build_llm_context
from graphrag.result_presentation.mechanism_summary import (
    protein_bridge_examples,
    readable_shared_targets,
    summarize_each_drug_pair,
)
from graphrag.result_presentation.vietnamese_labels import DISCLAIMER_VIETNAMESE
from graphrag.settings.application_settings import Settings
from graphrag.side_effect_ranking.effects_to_display import EffectDisplayLimits


MINIMUM_DRUGS_IN_REGIMEN = 3


@dataclass
class ResolvedRegimen:
    drug_ids: list[str]
    # Drug ID -> {"input": what the user typed, "method": how it was matched}.
    input_sources: dict[str, dict[str, object]]


@dataclass
class LLMOutcome:
    validated_predictions: list[dict[str, object]]
    rejected_predictions: list[dict[str, str]]
    # "ok", or "fallback" when the template explanations are used.
    status: str
    error_message: str | None


@dataclass
class PolypharmacySystem:
    settings: Settings
    graph: DecagonGraph
    names: NameIndex
    resolver: DrugResolver

    @classmethod
    def load(cls, settings: Settings | None = None) -> "PolypharmacySystem":
        settings = settings or Settings.from_environment()
        graph = DecagonGraph.load_from_settings(settings)
        names = NameIndex.load(
            drug_names_file=settings.drug_names_file,
            brand_aliases_file=settings.brand_aliases_file,
            gene_names_file=settings.gene_names_file,
            side_effect_translations_file=settings.side_effect_translations_file,
            serious_keywords_file=settings.serious_keywords_file,
            serious_effects_file=settings.serious_effects_file,
        )
        return cls(settings, graph, names, DrugResolver(names, graph.drug_ids))

    def status(self) -> dict[str, object]:
        drugs_with_names = 0
        for drug in self.graph.drug_ids:
            if self.names.drug_name(drug) != drug:
                drugs_with_names += 1
        return {
            "drugs": len(self.graph.drug_ids),
            "drugs_with_targets": len(self.graph.target_proteins_by_drug),
            "drugs_with_names": drugs_with_names,
            "side_effect_translations": len(self.names.side_effect_translations),
            "llm_configured": bool(self.settings.llm_api_key),
            "llm_model": self.settings.llm_model,
        }

    def resolve_regimen(self, inputs: list[str]) -> ResolvedRegimen:
        cleaned_inputs = [text.strip() for text in inputs if text and text.strip()]
        resolutions = [self.resolver.resolve(text) for text in cleaned_inputs]
        unresolved = [resolution for resolution in resolutions if resolution.stitch_id is None]
        if unresolved:
            raise RegimenError(
                "Không xác định được một số thuốc; hãy chọn từ gợi ý",
                [resolution.to_dict() for resolution in unresolved],
            )
        drug_ids: list[str] = []
        input_sources: dict[str, dict[str, object]] = {}
        for resolution in resolutions:
            if resolution.stitch_id not in input_sources:
                drug_ids.append(resolution.stitch_id)
                input_sources[resolution.stitch_id] = {
                    "input": resolution.query,
                    "method": resolution.method,
                }
        if len(drug_ids) < MINIMUM_DRUGS_IN_REGIMEN:
            raise RegimenError(
                "Cần ít nhất 3 thuốc khác nhau (một số tên trỏ tới cùng hoạt chất)"
            )
        return ResolvedRegimen(drug_ids, input_sources)

    def predict(self, inputs: list[str]) -> dict[str, object]:
        regimen = self.resolve_regimen(inputs)
        drugs = regimen.drug_ids
        limits = EffectDisplayLimits(
            max_serious_effects=self.settings.max_serious_effects,
            max_other_effects=self.settings.max_other_effects,
        )
        retrieved = retrieve_regimen_evidence(
            self.graph,
            drugs,
            is_serious=self.names.is_serious,
            weights=self.settings.ranking_weights,
            is_critical=self.names.is_critical,
            display_limits=limits,
        )
        graph_context = retrieved.graph_context
        evidence_items = retrieved.evidence_items
        llm_context = build_llm_context(drugs, graph_context, evidence_items, self.names, limits)
        llm_outcome = self._ask_llm(drugs, llm_context, evidence_items)

        tier_counts = {}
        for tier in (1, 2, 3):
            tier_counts[f"tier_{tier}"] = sum(evidence.tier == tier for evidence in evidence_items)
        return {
            "drugs": build_drug_cards(self.graph, self.names, drugs, regimen.input_sources),
            "interactions": build_interaction_cards(
                drugs,
                graph_context,
                evidence_items,
                llm_outcome.validated_predictions,
                self.names,
                limits,
            ),
            "mechanism_bridges": protein_bridge_examples(drugs, evidence_items, self.names),
            "coverage": summarize_data_coverage(
                drugs, self.graph, graph_context, evidence_items, self.names
            ),
            "pair_summary": summarize_each_drug_pair(drugs, graph_context, self.names),
            "shared_targets": readable_shared_targets(graph_context, self.names),
            "predictions": llm_outcome.validated_predictions,
            "llm_status": llm_outcome.status,
            "llm_error": llm_outcome.error_message,
            "evidence_truncated": bool(graph_context["evidence_truncated"]),
            "disclaimer": DISCLAIMER_VIETNAMESE,
            "retrieval_summary": {
                "evidence_count": len(evidence_items),
                "tier_counts": tier_counts,
                "max_serious_effects": limits.max_serious_effects,
                "max_other_effects": limits.max_other_effects,
                "max_evidence_items": self.settings.max_evidence_items,
                "hub_degree_threshold": self.graph.hub_degree_threshold,
                "rejected_predictions": llm_outcome.rejected_predictions,
            },
        }

    def _ask_llm(
        self, drugs: list[str], llm_context: dict[str, object], evidence_items: list
    ) -> LLMOutcome:
        """Call the LLM and validate its answer; any failure means "fallback" with a message."""
        validated: list[dict[str, object]] = []
        rejected: list[dict[str, str]] = []
        try:
            predictions = generate_predictions(self.settings, drugs, llm_context)
            validated = validate_predictions(predictions, evidence_items, self.graph, rejected)
        except (LLMConfigurationError, LLMResponseError) as error:
            return LLMOutcome(validated, rejected, "fallback", str(error))
        except httpx.HTTPStatusError as error:
            message = f"LLM provider returned HTTP {error.response.status_code}"
            try:
                detail = error.response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                detail = None
            if detail:
                message += f": {detail}"
            return LLMOutcome(validated, rejected, "fallback", message)
        except httpx.RequestError as error:
            message = f"Could not reach the LLM provider: {error.__class__.__name__}"
            return LLMOutcome(validated, rejected, "fallback", message)
        return LLMOutcome(validated, rejected, "ok", None)

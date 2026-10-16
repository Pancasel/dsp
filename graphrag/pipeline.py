"""End-to-end Graph-RAG pipeline, called directly by the Streamlit app.

    system = PolypharmacySystem.load()
    result = system.predict(["Panadol", "ibuprofen", "warfarin"])

One predict() call = resolve names -> retrieve and rank the subgraph -> one
LLM request with all evidence as JSON -> validate -> readable result. If the
LLM is not configured or fails, the result uses template explanations.
"""

from dataclasses import dataclass

import httpx

from graphrag.config import Settings
from graphrag.dataset import DecagonGraph
from graphrag.llm import (
    LLMConfigurationError,
    LLMResponseError,
    generate_predictions,
    validate_predictions,
)
from graphrag.names import NameIndex
from graphrag.presenter import (
    DISCLAIMER_VI,
    build_interactions,
    build_llm_context,
    coverage,
    drug_cards,
    mechanism_bridges,
)
from graphrag.resolver import DrugResolver, ResolveResult
from graphrag.vision import detected_names, prepare_image, read_image


class RegimenError(ValueError):
    """The regimen cannot be analysed; `details` says which inputs failed."""

    def __init__(self, message: str, details: list[dict[str, object]] | None = None) -> None:
        super().__init__(message)
        self.details = details or []


@dataclass
class PolypharmacySystem:
    settings: Settings
    graph: DecagonGraph
    names: NameIndex
    resolver: DrugResolver

    @classmethod
    def load(cls, settings: Settings | None = None) -> "PolypharmacySystem":
        settings = settings or Settings.from_env()
        mono = settings.mono_file if settings.mono_file and settings.mono_file.exists() else None
        graph = DecagonGraph.load(
            settings.targets_file,
            settings.ppi_file,
            settings.combo_file,
            settings.cache_db,
            settings.hub_degree_threshold,
            tier3_one_hop_per_pair=settings.tier3_one_hop_per_pair,
            tier3_two_hop_per_pair=settings.tier3_two_hop_per_pair,
            max_evidence=settings.max_evidence,
            mono_file=mono,
            tier3_candidates_per_pair=settings.tier3_candidates_per_pair,
        )
        names = NameIndex.load(
            settings.drug_names_file,
            settings.brand_aliases_file,
            settings.gene_names_file,
            settings.side_effects_vi_file,
            settings.serious_keywords_file,
        )
        return cls(settings, graph, names, DrugResolver(names, graph.drug_ids))

    def status(self) -> dict[str, object]:
        return {
            "drugs": len(self.graph.drug_ids),
            "drugs_with_targets": len(self.graph.targets),
            "drugs_with_names": sum(
                1 for drug in self.graph.drug_ids if self.names.drug_name(drug) != drug
            ),
            "side_effect_translations": len(self.names.side_effects_vi),
            "llm_configured": bool(self.settings.llm_api_key),
            "llm_model": self.settings.llm_model,
            "vision_model": self.settings.vision_model,
        }

    def list_drugs(self) -> list[dict[str, object]]:
        items = []
        for drug in self.graph.drug_ids:
            display = self.names.drug_display(drug)
            items.append(
                {
                    "stitch_id": drug,
                    "name": display["name"],
                    "aliases_vi": display["aliases_vi"],
                    "has_targets": bool(self.graph.targets.get(drug)),
                }
            )
        return sorted(items, key=lambda item: str(item["name"]).casefold())

    def search(self, query: str) -> ResolveResult:
        return self.resolver.resolve(query)

    def identify_image(self, data: bytes, content_type: str | None) -> dict[str, object]:
        """Image -> names read by the vision model -> resolved drugs.

        Raises ImageError for bad files and LLMConfigurationError /
        LLMResponseError / httpx errors when the vision call fails.
        """
        payload = read_image(self.settings, prepare_image(data, content_type))
        detected = [
            {"detected_text": name, **self.resolver.resolve(name).to_dict()}
            for name in detected_names(payload)
        ]
        return {
            "raw_text": payload["raw_text"],
            "products": payload["drugs"],
            "detected": detected,
            "message": None if detected else "Không đọc được tên thuốc trong ảnh",
        }

    def resolve_regimen(self, inputs: list[str]) -> tuple[list[str], dict[str, dict[str, object]]]:
        cleaned = [item.strip() for item in inputs if item and item.strip()]
        resolved = [self.resolver.resolve(item) for item in cleaned]
        unresolved = [item for item in resolved if item.stitch_id is None]
        if unresolved:
            raise RegimenError(
                "Không xác định được một số thuốc; hãy chọn từ gợi ý",
                [item.to_dict() for item in unresolved],
            )
        drugs: list[str] = []
        sources: dict[str, dict[str, object]] = {}
        for item in resolved:
            if item.stitch_id not in sources:
                drugs.append(item.stitch_id)
                sources[item.stitch_id] = {"input": item.query, "method": item.method}
        if len(drugs) < 3:
            raise RegimenError(
                "Cần ít nhất 3 thuốc khác nhau (một số tên trỏ tới cùng hoạt chất)"
            )
        return drugs, sources

    def predict(self, inputs: list[str], top_k: int | None = None) -> dict[str, object]:
        drugs, sources = self.resolve_regimen(inputs)
        top_k = top_k or self.settings.top_k_effects
        graph_context, evidence = self.graph.retrieve(
            drugs, self.names.is_serious, self.settings.ranking
        )
        llm_context = build_llm_context(drugs, graph_context, evidence, self.names, top_k)

        llm_status, llm_error = "ok", None
        validated: list[dict[str, object]] = []
        try:
            raw = generate_predictions(self.settings, drugs, llm_context)
            validated = validate_predictions(raw, evidence, self.graph)
        except (LLMConfigurationError, LLMResponseError) as exc:
            llm_status, llm_error = "fallback", str(exc)
        except httpx.HTTPStatusError as exc:
            llm_status = "fallback"
            llm_error = f"LLM provider returned HTTP {exc.response.status_code}"
        except httpx.RequestError as exc:
            llm_status = "fallback"
            llm_error = f"Could not reach the LLM provider: {exc.__class__.__name__}"

        return {
            "drugs": drug_cards(self.graph, self.names, drugs, sources),
            "interactions": build_interactions(
                drugs, graph_context, evidence, validated, self.names, top_k
            ),
            "mechanism_bridges": mechanism_bridges(drugs, evidence, self.names),
            "coverage": coverage(drugs, self.graph, graph_context, evidence, self.names),
            "predictions": validated,
            "llm_status": llm_status,
            "llm_error": llm_error,
            "evidence_truncated": bool(graph_context["evidence_truncated"]),
            "disclaimer": DISCLAIMER_VI,
            "retrieval_summary": {
                "evidence_count": len(evidence),
                "tier_counts": {
                    f"tier_{tier}": sum(item.tier == tier for item in evidence)
                    for tier in (1, 2, 3)
                },
                "top_k": top_k,
                "max_evidence": self.settings.max_evidence,
                "hub_degree_threshold": self.settings.hub_degree_threshold,
            },
        }

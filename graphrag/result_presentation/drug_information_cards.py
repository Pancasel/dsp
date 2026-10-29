"""One card per drug: names, brands, target proteins and side effects when taken alone."""

from graphrag.drug_names.name_index import NameIndex
from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.source_data.data_records import SideEffect


MAX_SINGLE_DRUG_EFFECTS_SHOWN = 10
MAX_TARGET_GENES_SHOWN = 12


def top_single_drug_side_effects(
    graph: DecagonGraph, names: NameIndex, drug: str
) -> list[dict[str, object]]:
    """The most characteristic single-drug effects: serious first, then the rarest among drugs.

    Decagon's single-drug list has no frequencies, so effects listed for few
    other drugs are taken as the most characteristic of this one.
    """
    effects = graph.single_drug_side_effects(drug)
    drug_count_by_code = graph.single_drug_effect_drug_counts()

    def serious_then_rarest_then_name(effect: SideEffect) -> tuple[bool, int, str]:
        return (
            not names.is_serious(effect.name),
            drug_count_by_code.get(effect.code, 0),
            effect.name.casefold(),
        )

    ordered_effects = sorted(effects, key=serious_then_rarest_then_name)
    return [
        {
            "name_en": names.side_effect_english(effect.name),
            "name_vi": names.side_effect_vietnamese(effect.name),
            "serious": names.is_serious(effect.name),
        }
        for effect in ordered_effects[:MAX_SINGLE_DRUG_EFFECTS_SHOWN]
    ]


def build_drug_cards(
    graph: DecagonGraph,
    names: NameIndex,
    drugs: list[str],
    input_sources: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """input_sources: drug -> {"input": what the user typed, "method": how it was matched}."""
    cards = []
    for drug in drugs:
        card = names.drug_identity(drug)
        targets = sorted(graph.target_proteins(drug))
        source = input_sources.get(drug, {})
        card.update(
            {
                "input": source.get("input", drug),
                "match_method": source.get("method", "stitch_id"),
                "has_targets": bool(targets),
                "target_count": len(targets),
                "target_genes": [
                    names.gene_symbol(gene) for gene in targets[:MAX_TARGET_GENES_SHOWN]
                ],
                "individual_side_effect_count": len(graph.single_drug_side_effects(drug)),
                "individual_side_effects": top_single_drug_side_effects(graph, names, drug),
            }
        )
        cards.append(card)
    return cards

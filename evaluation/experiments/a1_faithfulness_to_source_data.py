"""A1: does everything the system returns for random regimens really exist in the raw Decagon CSV files?

The CSV files are read here independently of the SQLite cache. Checked:
Tier 1 effects (precision and recall), Tier 2 flags (protein shared by >= 3
drugs), and every edge of every evidence path.
"""

import collections
import itertools
import math
import random
import time

from graphrag.drug_names.name_index import NameIndex
from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.knowledge_graph.regimen_evidence_retrieval import retrieve_regimen_evidence
from graphrag.settings.application_settings import Settings
from graphrag.source_data.csv_file_reader import read_csv_rows_with_required_columns
from evaluation.shared.result_csv_files import write_result_csv


REGIMEN_SIZES = (3, 4, 5, 6)


def read_raw_source_files(settings: Settings, pairs_needed: set[tuple[str, str]]):
    """Effects of the needed pairs, drug targets, protein interactions and protein degrees."""
    effect_codes_by_pair: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    rows = read_csv_rows_with_required_columns(
        settings.drug_pair_side_effects_file, ("STITCH 1", "STITCH 2", "Polypharmacy Side Effect")
    )
    for row in rows:
        pair = tuple(sorted((row["STITCH 1"], row["STITCH 2"])))
        if pair in pairs_needed:
            effect_codes_by_pair[pair].add(row["Polypharmacy Side Effect"])
    targets_by_drug: dict[str, set[str]] = collections.defaultdict(set)
    for row in read_csv_rows_with_required_columns(settings.drug_targets_file, ("STITCH", "Gene")):
        targets_by_drug[row["STITCH"]].add(row["Gene"])
    interacting_protein_pairs: set[tuple[str, str]] = set()
    degree_by_protein: collections.Counter = collections.Counter()
    rows = read_csv_rows_with_required_columns(settings.protein_interactions_file, ("Gene 1", "Gene 2"))
    for row in rows:
        protein_a, protein_b = row["Gene 1"], row["Gene 2"]
        if protein_a != protein_b and (protein_a, protein_b) not in interacting_protein_pairs:
            interacting_protein_pairs.add((protein_a, protein_b))
            interacting_protein_pairs.add((protein_b, protein_a))
            degree_by_protein[protein_a] += 1
            degree_by_protein[protein_b] += 1
    return effect_codes_by_pair, targets_by_drug, interacting_protein_pairs, degree_by_protein


def run_faithfulness_experiment(
    settings: Settings, graph: DecagonGraph, names: NameIndex, seed: int, regimen_count: int
) -> dict:
    print(f"A1 faithfulness vs raw CSV ({regimen_count} regimens)")
    random_generator = random.Random(seed)
    drugs_with_targets = sorted(graph.target_proteins_by_drug)
    all_drugs = sorted(graph.drug_ids)
    regimens = []
    for index in range(regimen_count):
        # Half from drugs with targets, so that Tier 2/3 evidence actually appears.
        drug_pool = drugs_with_targets if index % 2 == 0 else all_drugs
        regimens.append(random_generator.sample(drug_pool, random_generator.choice(REGIMEN_SIZES)))

    retrieved_by_regimen = []
    start = time.perf_counter()
    for regimen in regimens:
        retrieved = retrieve_regimen_evidence(graph, regimen, names.is_serious, settings.ranking_weights)
        retrieved_by_regimen.append((regimen, retrieved.graph_context, retrieved.evidence_items))
    retrieve_seconds = time.perf_counter() - start

    pairs_needed = {
        pair for regimen, _, _ in retrieved_by_regimen for pair in itertools.combinations(sorted(regimen), 2)
    }
    raw_effects, raw_targets, raw_interactions, raw_degree = read_raw_source_files(settings, pairs_needed)

    tally = collections.Counter()
    for regimen, graph_context, evidence_items in retrieved_by_regimen:
        pairs = list(itertools.combinations(sorted(regimen), 2))
        returned = {
            (tuple(pair), effect["code"])
            for effect in graph_context["ranked_effects"]
            for pair in effect["pairs"]
        }
        truth = {(pair, code) for pair in pairs for code in raw_effects.get(pair, ())}
        tally["t1_returned"] += len(returned)
        tally["t1_supported"] += len(returned & truth)
        tally["t1_truth"] += len(truth)
        tally["t1_recalled"] += len(returned & truth)

        # Tier 2: pair with documented effects + a protein shared by >= 3 regimen drugs.
        drug_count_by_protein = collections.Counter(
            gene for drug in regimen for gene in raw_targets.get(drug, ())
        )
        shared_proteins = {gene for gene, count in drug_count_by_protein.items() if count >= 3}
        pairs_that_should_be_amplified = {
            pair
            for pair in pairs
            if raw_effects.get(pair)
            and shared_proteins & raw_targets.get(pair[0], set()) & raw_targets.get(pair[1], set())
        }
        flagged_pairs = {
            tuple(pair) for effect in graph_context["ranked_effects"] for pair in effect["amplified_pairs"]
        }
        tally["t2_flagged"] += len(flagged_pairs)
        tally["t2_flagged_correct"] += len(flagged_pairs & pairs_that_should_be_amplified)
        tally["t2_expected"] += len(pairs_that_should_be_amplified)
        tally["t2_expected_found"] += len(flagged_pairs & pairs_that_should_be_amplified)

        for evidence in evidence_items:
            tally[f"evidence_tier{evidence.tier}"] += 1
            path_is_correct = _path_edges_exist(evidence.path, raw_effects, raw_targets, raw_interactions)
            if evidence.tier == 3:
                # Tier 3 is only for undocumented pairs, through non-hub middle proteins.
                path_is_correct &= not raw_effects.get(evidence.drug_pair)
                regimen_targets = set().union(*(raw_targets.get(drug, set()) for drug in regimen))
                for edge in evidence.path[1:-1]:
                    for protein in (edge["source"], edge["target"]):
                        if protein not in regimen_targets:
                            path_is_correct &= raw_degree[protein] <= settings.hub_degree_threshold
            tally[f"evidence_tier{evidence.tier}_ok"] += path_is_correct

    def ratio(numerator_key: str, denominator_key: str) -> float:
        if not tally[denominator_key]:
            return float("nan")
        return tally[numerator_key] / tally[denominator_key]

    rows = [
        {"check": "Tier 1 – precision: tác dụng phụ trả về có trong combo CSV",
         "numerator": tally["t1_supported"], "denominator": tally["t1_returned"],
         "value": ratio("t1_supported", "t1_returned")},
        {"check": "Tier 1 – recall: tác dụng phụ trong CSV của các cặp được trả về",
         "numerator": tally["t1_recalled"], "denominator": tally["t1_truth"],
         "value": ratio("t1_recalled", "t1_truth")},
        {"check": "Tier 2 – precision: cặp gắn Tier 2 thỏa quy tắc protein chung",
         "numerator": tally["t2_flagged_correct"], "denominator": tally["t2_flagged"],
         "value": ratio("t2_flagged_correct", "t2_flagged")},
        {"check": "Tier 2 – recall: cặp thỏa quy tắc được gắn Tier 2",
         "numerator": tally["t2_expected_found"], "denominator": tally["t2_expected"],
         "value": ratio("t2_expected_found", "t2_expected")},
    ]
    for tier in (1, 2, 3):
        rows.append(
            {
                "check": f"Tier {tier} – đường dẫn bằng chứng đúng từng cạnh",
                "numerator": tally[f"evidence_tier{tier}_ok"],
                "denominator": tally[f"evidence_tier{tier}"],
                "value": ratio(f"evidence_tier{tier}_ok", f"evidence_tier{tier}"),
            }
        )
    for row in rows:
        row["value"] = round(row["value"], 4) if not math.isnan(row["value"]) else "n/a"
        print(f"  {row['check']}: {row['numerator']}/{row['denominator']} = {row['value']}")
    write_result_csv("a1_faithfulness.csv", rows)
    return {"regimens": len(regimens), "retrieve_seconds_mean": retrieve_seconds / len(regimens)}


def _path_edges_exist(path, raw_effects, raw_targets, raw_interactions) -> bool:
    path_is_correct = True
    for edge in path:
        relation = edge["relation"]
        if relation == "targets":
            path_is_correct &= edge["target"] in raw_targets.get(edge["source"], set())
        elif relation == "interacts_with":
            path_is_correct &= (edge["source"], edge["target"]) in raw_interactions
        elif relation == "has_side_effect":
            pair = tuple(sorted((edge["source"], edge["target"])))
            path_is_correct &= edge["side_effect_code"] in raw_effects.get(pair, set())
        else:
            path_is_correct = False
    return path_is_correct

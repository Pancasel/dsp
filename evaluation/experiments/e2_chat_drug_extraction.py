"""E2: are the right drugs found in whole chat messages (the path the chat interface really uses)?

Test cases: evaluation/test_cases/chat_extraction_cases.csv
"""

from graphrag.drug_names.drug_name_resolver import DrugResolver
from evaluation.shared.result_csv_files import read_test_cases, write_result_csv


def run_chat_drug_extraction_experiment(resolver: DrugResolver) -> None:
    print("E2 drug extraction from chat messages")
    rows = []
    true_positives = false_positives = false_negatives = questions_asked = 0
    for case in read_test_cases("chat_extraction_cases.csv"):
        expected_drugs = {
            resolver.resolve(name).stitch_id for name in case["expected"].split("|") if name
        }
        found = resolver.extract(case["message"])
        found_drugs = {resolution.stitch_id for resolution in found if resolution.stitch_id}
        clarifying_questions = [
            resolution.query
            for resolution in found
            if resolution.stitch_id is None and resolution.method == "ambiguous"
        ]
        true_positives += len(found_drugs & expected_drugs)
        false_positives += len(found_drugs - expected_drugs)
        false_negatives += len(expected_drugs - found_drugs)
        questions_asked += len(clarifying_questions)
        rows.append(
            {
                "message": case["message"],
                "category": case["category"],
                "expected": "|".join(sorted(resolver.names.drug_name(drug) for drug in expected_drugs)),
                "got": "|".join(sorted(resolver.names.drug_name(drug) for drug in found_drugs)),
                "asked": "|".join(clarifying_questions),
                "exact": int(found_drugs == expected_drugs),
            }
        )
        if found_drugs != expected_drugs:
            print(
                f"  MISMATCH: {case['message']!r}: expected {rows[-1]['expected']} "
                f"got {rows[-1]['got']} asked {clarifying_questions}"
            )
    write_result_csv("e2_extraction.csv", rows)

    found_total = true_positives + false_positives
    expected_total = true_positives + false_negatives
    summary = {
        "messages": len(rows),
        "exact_match": round(sum(row["exact"] for row in rows) / len(rows), 4),
        "drug_precision": round(true_positives / found_total, 4) if found_total else 0.0,
        "drug_recall": round(true_positives / expected_total, 4) if expected_total else 0.0,
        "wrong_drugs": false_positives,
        "missed_drugs": false_negatives,
        "clarifying_questions": questions_asked,
    }
    write_result_csv("e2_summary.csv", [summary])
    print(f"  {summary}")

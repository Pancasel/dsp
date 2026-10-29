"""E1: is each typed drug name (generic, Vietnamese brand, with strength, typo, unknown) resolved correctly?

Test cases: evaluation/test_cases/name_resolution_cases.csv
Outcome per case: "correct", "asked" (the user is asked to choose) or "wrong".
"""

import collections

from graphrag.drug_names.drug_name_resolver import DrugResolver
from evaluation.shared.result_csv_files import read_test_cases, write_result_csv


def run_name_resolution_experiment(resolver: DrugResolver) -> None:
    print("E1 name resolution")
    rows = []
    for case in read_test_cases("name_resolution_cases.csv"):
        resolution = resolver.resolve(case["query"])
        resolved_name = resolution.name or ""
        expected_name = case["expected_name"]
        if expected_name:
            if resolved_name == expected_name:
                outcome = "correct"
            elif resolution.stitch_id is None:
                outcome = "asked"
            else:
                outcome = "wrong"
        else:
            # The drug is not in the data: the only correct answer is "not found".
            outcome = "correct" if resolution.stitch_id is None else "wrong"
        rows.append(
            {
                "query": case["query"],
                "category": case["category"],
                "expected": expected_name,
                "got": resolved_name,
                "method": resolution.method,
                "score": resolution.score,
                "outcome": outcome,
            }
        )
    write_result_csv("e1_resolver.csv", rows)

    outcomes_by_category = collections.defaultdict(list)
    for row in rows:
        outcomes_by_category[row["category"]].append(row["outcome"])
    for category, outcomes in outcomes_by_category.items():
        print(
            f"  {category:14} n={len(outcomes):3} correct={outcomes.count('correct') / len(outcomes):.0%} "
            f"asked={outcomes.count('asked')} wrong={outcomes.count('wrong')}"
        )
    all_outcomes = [row["outcome"] for row in rows]
    print(
        f"  overall accuracy {all_outcomes.count('correct') / len(all_outcomes):.1%}, "
        f"wrong {all_outcomes.count('wrong')}"
    )

"""Map the Entrez Gene IDs used by Decagon to HGNC gene symbols (run once, needs internet).

Usage (from the project root):
    python -m resource_builders.build_gene_names [--out resources/gene_names.csv]
"""

import argparse
import csv
import gzip
import io
from pathlib import Path

import httpx

from graphrag.settings.application_settings import Settings
from graphrag.settings.environment_variable_readers import PROJECT_ROOT
from graphrag.source_data.csv_file_reader import read_csv_rows_with_required_columns


# NCBI table of all human genes (ID, symbol, description, ...).
NCBI_HUMAN_GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/"
    "Homo_sapiens.gene_info.gz"
)


def genes_used_by_decagon(settings: Settings) -> set[str]:
    gene_ids = {row["Gene"] for row in read_csv_rows_with_required_columns(settings.drug_targets_file, ("Gene",))}
    for row in read_csv_rows_with_required_columns(settings.protein_interactions_file, ("Gene 1", "Gene 2")):
        gene_ids.add(row["Gene 1"])
        gene_ids.add(row["Gene 2"])
    return {gene_id for gene_id in gene_ids if gene_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="resources/gene_names.csv")
    arguments = parser.parse_args()
    settings = Settings.from_environment()
    wanted_gene_ids = genes_used_by_decagon(settings)
    print(f"{len(wanted_gene_ids)} gene IDs in Decagon")

    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        response = client.get(NCBI_HUMAN_GENE_INFO_URL)
        response.raise_for_status()
    text = gzip.decompress(response.content).decode("utf-8")

    rows = []
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    header = next(reader)
    gene_id_column = header.index("GeneID")
    symbol_column = header.index("Symbol")
    description_column = header.index("description")
    for record in reader:
        if record[gene_id_column] in wanted_gene_ids:
            rows.append((record[gene_id_column], record[symbol_column], record[description_column]))

    output_file = Path(arguments.out)
    if not output_file.is_absolute():
        output_file = PROJECT_ROOT / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def numeric_gene_id(row: tuple[str, str, str]) -> int:
        return int(row[0])

    with output_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["gene_id", "symbol", "description"])
        writer.writerows(sorted(rows, key=numeric_gene_id))
    print(f"wrote {output_file} ({len(rows)}/{len(wanted_gene_ids)} genes named)")


if __name__ == "__main__":
    main()

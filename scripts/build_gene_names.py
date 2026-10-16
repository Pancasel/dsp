"""Map Entrez Gene IDs used by Decagon to HGNC symbols (run once, needs internet).

Usage:
    python scripts/build_gene_names.py [--out resources/gene_names.csv]
"""

import argparse
import csv
import gzip
import io
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphrag.config import PROJECT_ROOT, Settings  # noqa: E402
from graphrag.dataset import _rows  # noqa: E402


GENE_INFO = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/"
    "Homo_sapiens.gene_info.gz"
)


def decagon_genes(settings: Settings) -> set[str]:
    genes = {row["Gene"] for row in _rows(settings.targets_file, ("Gene",))}
    for row in _rows(settings.ppi_file, ("Gene 1", "Gene 2")):
        genes.add(row["Gene 1"])
        genes.add(row["Gene 2"])
    return {gene for gene in genes if gene}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="resources/gene_names.csv")
    args = parser.parse_args()
    settings = Settings.from_env()
    wanted = decagon_genes(settings)
    print(f"{len(wanted)} gene IDs in Decagon")

    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        response = client.get(GENE_INFO)
        response.raise_for_status()
    text = gzip.decompress(response.content).decode("utf-8")

    rows = []
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    header = next(reader)
    gene_id_col = header.index("GeneID")
    symbol_col = header.index("Symbol")
    description_col = header.index("description")
    for record in reader:
        if record[gene_id_col] in wanted:
            rows.append((record[gene_id_col], record[symbol_col], record[description_col]))

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["gene_id", "symbol", "description"])
        writer.writerows(sorted(rows, key=lambda row: int(row[0])))
    print(f"wrote {out} ({len(rows)}/{len(wanted)} genes named)")


if __name__ == "__main__":
    main()

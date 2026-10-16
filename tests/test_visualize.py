import json

import pytest

pytest.importorskip("pyvis")

from graphrag.visualize import build_graph_html  # noqa: E402


def _result() -> dict:
    return {
        "drugs": [
            {"name": name, "aliases_vi": [], "target_count": 1}
            for name in ("Alphazol", "Betamin", "Gammacet")
        ],
        "pair_summary": [
            {"pair": ["Alphazol", "Betamin"], "documented_effects": 2, "serious_effects": 1},
            {"pair": ["Alphazol", "Gammacet"], "documented_effects": 0, "serious_effects": 0},
            {"pair": ["Betamin", "Gammacet"], "documented_effects": 0, "serious_effects": 0},
        ],
        "shared_targets": [{"protein": "GENE1", "drugs": ["Alphazol", "Betamin", "Gammacet"]}],
        "mechanism_bridges": [
            {
                "readable_path": [
                    {"source": "Alphazol", "relation": "targets", "target": "GENE1"},
                    {"source": "GENE1", "relation": "interacts_with", "target": "GENE3"},
                    {"source": "Gammacet", "relation": "targets", "target": "GENE3"},
                ]
            }
        ],
        "interactions": [
            {"serious": True, "side_effect_en": "gastrointestinal haemorrhage",
             "side_effect_vi": "xuất huyết tiêu hóa", "tier": 1, "pairs": [["Alphazol", "Betamin"]]},
            {"serious": False, "side_effect_en": "nausea", "side_effect_vi": "buồn nôn",
             "tier": 1, "pairs": [["Alphazol", "Betamin"]]},
        ],
    }


def _has(html: str, label: str) -> bool:
    # pyvis writes node data as ASCII JSON ("xu\u1ea5t ...").
    return label in html or json.dumps(label)[1:-1] in html


def test_graph_contains_layers_and_multiline_tooltips() -> None:
    html = build_graph_html(_result())
    for label in ("Alphazol", "GENE1", "GENE3", "xuất huyết tiêu hóa", "chưa có dữ liệu"):
        assert _has(html, label)
    assert not _has(html, "buồn nôn")  # only serious effects are drawn
    assert ".vis-tooltip{white-space:pre-line}" in html


def test_graph_layers_can_be_hidden() -> None:
    html = build_graph_html(_result(), show_proteins=False, show_effects=False)
    assert "GENE1" not in html
    assert not _has(html, "xuất huyết tiêu hóa")

import csv
import io
from pathlib import Path

import pytest
from PIL import Image

from graphrag.config import Settings
from graphrag.llm import LLMConfigurationError, ModelPrediction
from graphrag.pipeline import PolypharmacySystem, RegimenError
from graphrag.vision import ImageError


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


@pytest.fixture
def system(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PolypharmacySystem:
    files = {name: tmp_path / f"{name}.csv" for name in (
        "targets", "ppi", "combo", "mono", "names", "aliases", "genes", "vi", "serious"
    )}
    _write_csv(
        files["targets"],
        ["STITCH", "Gene"],
        [["D1", "G1"], ["D2", "G2"], ["D3", "G3"], ["D4", "G4"], ["D5", "G5"], ["D6", "G6"]],
    )
    _write_csv(files["ppi"], ["Gene 1", "Gene 2"], [["G1", "G3"]])
    _write_csv(
        files["combo"],
        ["STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"],
        [
            ["D1", "D2", "C1", "known event"],
            ["D1", "D2", "C9", "gastrointestinal haemorrhage"],
            ["D3", "D4", "C1", "known event"],
        ],
    )
    _write_csv(
        files["mono"],
        ["STITCH", "Individual Side Effect", "Side Effect Name"],
        [["D1", "C1", "known event"], ["D1", "M1", "headache"]],
    )
    _write_csv(
        files["names"],
        ["stitch_id", "pubchem_cid", "name", "synonyms"],
        [["D1", "1", "Alphazol", "alfa"], ["D2", "2", "Betamin", ""], ["D3", "3", "Gammacet", ""]],
    )
    _write_csv(files["aliases"], ["alias", "generic_name", "stitch_id", "note"], [["Alphadol", "alphazol", "D1", ""]])
    _write_csv(files["genes"], ["gene_id", "symbol", "description"], [["G1", "GENE1", ""]])
    _write_csv(
        files["vi"],
        ["name_en", "display_en", "name_vi"],
        [["gastrointestinal haemorrhage", "gastrointestinal haemorrhage", "xuất huyết tiêu hóa"]],
    )
    _write_csv(files["serious"], ["keyword", "category"], [["haemorrhage", "bleeding"]])
    env = {
        "TARGETS_FILE": files["targets"],
        "PPI_FILE": files["ppi"],
        "COMBO_FILE": files["combo"],
        "MONO_FILE": files["mono"],
        "DRUG_NAMES_FILE": files["names"],
        "BRAND_ALIASES_FILE": files["aliases"],
        "GENE_NAMES_FILE": files["genes"],
        "SIDE_EFFECTS_VI_FILE": files["vi"],
        "SERIOUS_KEYWORDS_FILE": files["serious"],
        "CACHE_DB": tmp_path / "cache.sqlite3",
    }
    for name, path in env.items():
        monkeypatch.setenv(name, str(path))
    # Empty key: no request ever leaves the machine during tests.
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("VISION_API_KEY", "")
    return PolypharmacySystem.load(Settings.from_env())


def test_status_and_search(system: PolypharmacySystem) -> None:
    status = system.status()
    assert status["drugs"] == 6
    assert status["drugs_with_names"] == 3
    assert system.search("Alphadol 500mg").stitch_id == "D1"
    assert system.search("zzzz").stitch_id is None
    assert system.list_drugs()[0]["name"] == "Alphazol"


def test_regimen_validation(system: PolypharmacySystem) -> None:
    with pytest.raises(RegimenError):
        system.predict(["D1", "D2"])
    with pytest.raises(RegimenError) as error:
        system.predict(["D1", "D2", "UNKNOWN"])
    assert error.value.details[0]["query"] == "UNKNOWN"
    # Brand and generic of the same drug count once.
    with pytest.raises(RegimenError):
        system.predict(["Alphadol", "Alphazol", "D2"])


def test_predict_without_llm_uses_template_fallback(system: PolypharmacySystem) -> None:
    body = system.predict(["Alphadol", "Betamin", "D3", "D4"])
    assert body["llm_status"] == "fallback"
    assert [card["name"] for card in body["drugs"]][:3] == ["Alphazol", "Betamin", "Gammacet"]
    assert body["drugs"][0]["match_method"] == "brand_alias"
    top = body["interactions"][0]
    # Serious, rare, not-in-mono effect is ranked above the generic one.
    assert top["side_effect_en"] == "gastrointestinal haemorrhage"
    assert top["side_effect_vi"] == "xuất huyết tiêu hóa"
    assert top["serious"] is True
    assert top["explanation_source"] == "template"
    assert top["evidence"][0]["path_text"]
    assert body["coverage"]["pairs_total"] == 6
    assert body["coverage"]["pairs_with_data"] == 2
    assert "GENE1" in body["mechanism_bridges"][0]["path_text"]


def test_predict_with_llm_attaches_verified_explanations(
    system: PolypharmacySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_predictions(settings, drugs, context):
        evidence = {item["side_effect"]: item["evidence_id"] for item in context["evidence"] if "side_effect" in item}
        return [
            ModelPrediction(side_effect="known event", tier=1, evidence_ids=[evidence["known event"]], explanation="Giải thích từ LLM."),
            ModelPrediction(side_effect="invented", tier=1, evidence_ids=["E999"], explanation="Bịa."),
        ]

    monkeypatch.setattr("graphrag.pipeline.generate_predictions", fake_predictions)
    body = system.predict(["D1", "D2", "D3"])
    assert body["llm_status"] == "ok"
    assert len(body["predictions"]) == 1
    known = next(item for item in body["interactions"] if item["side_effect_en"] == "known event")
    assert known["explanation_source"] == "llm"
    assert known["explanation_vi"] == "Giải thích từ LLM."


def test_identify_image(system: PolypharmacySystem, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ImageError):
        system.identify_image(b"hello", "text/plain")
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buffer, format="PNG")
    with pytest.raises(LLMConfigurationError):
        system.identify_image(buffer.getvalue(), "image/png")

    def fake_read_image(settings, data_url):
        assert data_url.startswith("data:image/jpeg;base64,")
        return {
            "drugs": [{"brand": "Alphadol", "active_ingredients": ["Alphazol"], "strength": "500mg"}],
            "raw_text": "ALPHADOL 500mg",
        }

    monkeypatch.setattr("graphrag.pipeline.read_image", fake_read_image)
    body = system.identify_image(buffer.getvalue(), "image/png")
    assert [item["stitch_id"] for item in body["detected"]] == ["D1", "D1"]
    assert body["detected"][1]["method"] == "brand_alias"

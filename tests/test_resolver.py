import pytest

from graphrag.names import DrugName, NameIndex, normalize
from graphrag.resolver import DrugResolver, split_query


@pytest.fixture
def resolver() -> DrugResolver:
    drugs = {
        "CID000001983": DrugName("CID000001983", "Acetaminophen", ["Paracetamol", "Tylenol"], ["Panadol", "Hapacol"]),
        "CID000003672": DrugName("CID000003672", "Ibuprofen", ["Brufen"], []),
        "CID000006691": DrugName("CID000006691", "Warfarin", [], ["Coumadin"]),
        "CID000004594": DrugName("CID000004594", "Omeprazole", [], ["Losec"]),
        "CID000003249": DrugName("CID000003249", "Ergocalciferol", ["Vitamin D2"], []),
        "CID000003961": DrugName("CID000003961", "Losartan", [], []),
    }
    names = NameIndex(drugs, {}, {}, [])
    return DrugResolver(names, set(drugs))


def test_normalize_strips_accents_strength_and_forms() -> None:
    assert normalize("Panadol 500mg viên nén") == "panadol"
    assert normalize("Hapacol 250") == "hapacol"
    assert normalize("Losartan kali 50 mg") == "losartan"
    assert normalize("Thuốc Đau") == "thuoc dau"


@pytest.mark.parametrize(
    ("query", "expected", "method"),
    [
        ("CID000006691", "CID000006691", "stitch_id"),
        ("ibuprofen", "CID000003672", "exact"),
        ("PARACETAMOL", "CID000001983", "exact"),
        ("Panadol Extra 500mg", "CID000001983", "brand_alias"),
        ("Hapacol 250", "CID000001983", "brand_alias"),
        ("ibuprofn", "CID000003672", "fuzzy"),
        ("wafarin", "CID000006691", "fuzzy"),
        ("losartan kali", "CID000003961", "exact"),
    ],
)
def test_resolves_names(resolver: DrugResolver, query: str, expected: str, method: str) -> None:
    result = resolver.resolve(query)
    assert result.stitch_id == expected
    assert result.method == method


@pytest.mark.parametrize("query", ["esomeprazole", "Vitamin C"])
def test_near_miss_is_not_auto_accepted(resolver: DrugResolver, query: str) -> None:
    result = resolver.resolve(query)
    assert result.stitch_id is None
    assert result.method in {"ambiguous", "not_found"}


def test_unknown_drug_is_not_found(resolver: DrugResolver) -> None:
    assert resolver.resolve("xyzabc").method == "not_found"


def test_split_query_handles_combinations() -> None:
    assert split_query("Paracetamol + Codein, ibuprofen và warfarin") == [
        "Paracetamol",
        "Codein",
        "ibuprofen",
        "warfarin",
    ]


def test_resolve_many(resolver: DrugResolver) -> None:
    results = resolver.resolve_many("Panadol, Brufen; Coumadin")
    assert [item.stitch_id for item in results] == [
        "CID000001983",
        "CID000003672",
        "CID000006691",
    ]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("tôi đang uống panadol với ibuprofen và warfarin", ["CID000001983", "CID000003672", "CID000006691"]),
        ("Mình dùng Hapacol 500mg cùng Losec thì có sao không?", ["CID000001983", "CID000004594"]),
        ("I'm taking Coumadin and Brufen", ["CID000006691", "CID000003672"]),
        ("thêm wafarin", ["CID000006691"]),
        ("CID000003961 và paracetamol", ["CID000003961", "CID000001983"]),
        ("xin chào", []),
    ],
)
def test_extract_drugs_from_chat_messages(resolver: DrugResolver, message: str, expected: list[str]) -> None:
    assert [item.stitch_id for item in resolver.extract(message)] == expected


def test_extract_keeps_ambiguous_names_for_confirmation(resolver: DrugResolver) -> None:
    results = resolver.extract("uống esomeprazole với ibuprofen")
    assert results[0].stitch_id is None and results[0].method in {"ambiguous", "not_found"}
    assert results[1].stitch_id == "CID000003672"

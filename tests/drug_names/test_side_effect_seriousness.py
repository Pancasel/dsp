from graphrag.drug_names.name_index import NameIndex


def test_serious_keywords_match_whole_words() -> None:
    names = NameIndex(
        drugs_by_id={},
        gene_symbols_by_id={},
        side_effect_translations={},
        critical_keywords=["coma", "haemorrhage"],
    )
    assert names.is_serious("coma")
    assert names.is_serious("gastrointestinal haemorrhage")
    assert not names.is_serious("glaucoma")


def test_serious_labels_win_and_keywords_handle_plurals_and_exclusions() -> None:
    names = NameIndex(
        drugs_by_id={},
        gene_symbols_by_id={},
        side_effect_translations={},
        critical_keywords=["thrombosis", "coma", "agranulocytosis"],
        serious_labels={"heart attack": True, "coma": False},
        excluded_keywords=["scale"],
    )
    assert names.is_serious("Heart attack")  # EMA label, no keyword needed
    assert not names.is_serious("coma")  # the label overrides the keyword
    assert names.is_serious("deep vein thromboses")  # plural of a keyword
    assert not names.is_serious("glasgow coma scale")  # excluded lab scale

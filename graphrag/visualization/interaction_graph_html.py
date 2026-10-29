"""Interactive relation graph of an analysis result, as HTML (pyvis / vis-network).

Layers, each optional:
- drugs, with an edge per pair labelled by its documented side-effect count;
- proteins targeted by >= 3 drugs (Tier 2 signal) and protein bridges (Tier 3);
- the serious side effects shown in the result, linked to the drugs involved.

The HTML embeds vis-network inline, so it renders offline. pyvis is imported
inside the function so that a missing package disables only the graph, not the whole app.
"""

COLORS = {
    "drug": "#2563eb",
    "shared_protein": "#ea580c",
    "bridge_protein": "#7c3aed",
    "side_effect": "#dc2626",
    "drug_pair": "#64748b",
}
# Edge width grows with the number of side effects, up to this count.
EDGE_WIDTH_MAX_EFFECT_COUNT = 300


def build_graph_html(
    result: dict[str, object],
    show_proteins: bool = True,
    show_effects: bool = True,
    height: str = "620px",
) -> str:
    from pyvis.network import Network

    network = Network(
        height=height, width="100%", bgcolor="#ffffff", font_color="#111827", cdn_resources="in_line"
    )
    network.barnes_hut(gravity=-6000, spring_length=140)
    drug_names = [card["name"] for card in result["drugs"]]
    _add_drugs_and_pairs(network, result)
    if show_proteins:
        _add_shared_proteins(network, result)
        _add_bridge_proteins(network, result, drug_names)
    if show_effects:
        _add_serious_side_effects(network, result)

    html = network.generate_html(notebook=False)
    # vis-network styles tooltips with white-space: nowrap, which would put
    # the multi-line titles above on a single line.
    return html.replace("</head>", "<style>.vis-tooltip{white-space:pre-line}</style></head>", 1)


def _add_drugs_and_pairs(network, result: dict[str, object]) -> None:
    for card in result["drugs"]:
        brand_names = ", ".join(card["aliases_vi"][:4]) or "-"
        tooltip = f"{card['name']}\nBiệt dược: {brand_names}\nProtein đích: {card['target_count']}"
        network.add_node(
            f"drug:{card['name']}",
            label=card["name"],
            title=tooltip,
            color=COLORS["drug"],
            shape="dot",
            size=28,
            font={"size": 18, "color": "#111827"},
        )

    for pair_summary in result["pair_summary"]:
        drug_a, drug_b = pair_summary["pair"]
        effect_count = pair_summary["documented_effects"]
        if effect_count:
            # "TDP" = tác dụng phụ (side effects).
            label = f"{effect_count} TDP ({pair_summary['serious_effects']} nặng)"
            network.add_edge(
                f"drug:{drug_a}",
                f"drug:{drug_b}",
                label=label,
                title=label,
                color=COLORS["drug_pair"],
                width=1 + min(effect_count, EDGE_WIDTH_MAX_EFFECT_COUNT) / 60,
            )
        else:
            network.add_edge(
                f"drug:{drug_a}", f"drug:{drug_b}", label="chưa có dữ liệu", dashes=True, color="#cbd5e1"
            )


def _add_shared_proteins(network, result: dict[str, object]) -> None:
    for shared in result["shared_targets"]:
        node_id = f"protein:{shared['protein']}"
        network.add_node(
            node_id,
            label=shared["protein"],
            title=f"Protein đích chung của {', '.join(shared['drugs'])}",
            color=COLORS["shared_protein"],
            shape="diamond",
            size=16,
        )
        for drug in shared["drugs"]:
            network.add_edge(f"drug:{drug}", node_id, color=COLORS["shared_protein"], title="tác động lên")


def _add_bridge_proteins(network, result: dict[str, object], drug_names: list[str]) -> None:
    def node_id(label: str) -> str:
        if label in drug_names:
            return f"drug:{label}"
        return f"protein:{label}"

    for bridge in result["mechanism_bridges"]:
        for edge in bridge["readable_path"]:
            for end_label in (edge["source"], edge["target"]):
                if end_label not in drug_names and f"protein:{end_label}" not in network.get_nodes():
                    network.add_node(
                        f"protein:{end_label}",
                        label=end_label,
                        title="Protein trên cầu nối (Tier 3)",
                        color=COLORS["bridge_protein"],
                        shape="diamond",
                        size=12,
                    )
            if edge["relation"] == "targets":
                edge_title = "tác động lên"
            else:
                edge_title = "tương tác protein"
            network.add_edge(
                node_id(edge["source"]),
                node_id(edge["target"]),
                color=COLORS["bridge_protein"],
                dashes=edge["relation"] == "interacts_with",
                title=edge_title,
            )


def _add_serious_side_effects(network, result: dict[str, object]) -> None:
    for card in result["interactions"]:
        if not card["serious"]:
            continue
        node_id = f"effect:{card['side_effect_en']}"
        network.add_node(
            node_id,
            label=card["side_effect_vi"],
            title=f"{card['side_effect_vi']} ({card['side_effect_en']})\nTier {card['tier']}",
            color=COLORS["side_effect"],
            shape="box",
            font={"color": "#ffffff"},
        )
        drugs_involved = sorted({drug_name for pair in card["pairs"] for drug_name in pair})
        for drug_name in drugs_involved:
            network.add_edge(f"drug:{drug_name}", node_id, color="#fca5a5", width=1)

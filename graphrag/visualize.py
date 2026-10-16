"""Interactive relation graph of an analysis result (pyvis / vis-network).

Layers, each optional:
- drugs, with an edge per pair labelled by its documented side-effect count;
- proteins targeted by >= 3 drugs (Tier 2 signal) and protein bridges (Tier 3);
- the serious side effects shown in the result, linked to the drugs involved.

The HTML embeds vis-network inline, so it renders offline. pyvis is imported
lazily so that a missing package disables only the graph, not the whole app.
"""

COLORS = {
    "drug": "#2563eb",
    "shared": "#ea580c",
    "bridge": "#7c3aed",
    "effect": "#dc2626",
    "pair": "#64748b",
}


def build_graph_html(
    result: dict[str, object],
    show_proteins: bool = True,
    show_effects: bool = True,
    height: str = "620px",
) -> str:
    from pyvis.network import Network

    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#111827", cdn_resources="in_line")
    net.barnes_hut(gravity=-6000, spring_length=140)
    drug_names = [card["name"] for card in result["drugs"]]
    for card in result["drugs"]:
        tip = f"{card['name']}\nBiệt dược: {', '.join(card['aliases_vi'][:4]) or '-'}\nProtein đích: {card['target_count']}"
        net.add_node(f"drug:{card['name']}", label=card["name"], title=tip, color=COLORS["drug"],
                     shape="dot", size=28, font={"size": 18, "color": "#111827"})

    for entry in result["pair_summary"]:
        a, b = entry["pair"]
        total = entry["documented_effects"]
        if total:
            label = f"{total} TDP ({entry['serious_effects']} nặng)"
            net.add_edge(f"drug:{a}", f"drug:{b}", label=label, title=label, color=COLORS["pair"],
                         width=1 + min(total, 300) / 60)
        else:
            net.add_edge(f"drug:{a}", f"drug:{b}", label="chưa có dữ liệu", dashes=True, color="#cbd5e1")

    if show_proteins:
        for item in result["shared_targets"]:
            node = f"protein:{item['protein']}"
            net.add_node(node, label=item["protein"], title=f"Protein đích chung của {', '.join(item['drugs'])}",
                         color=COLORS["shared"], shape="diamond", size=16)
            for drug in item["drugs"]:
                net.add_edge(f"drug:{drug}", node, color=COLORS["shared"], title="tác động lên")
        for bridge in result["mechanism_bridges"]:
            for edge in bridge["readable_path"]:
                for end in (edge["source"], edge["target"]):
                    if end not in drug_names and f"protein:{end}" not in net.get_nodes():
                        net.add_node(f"protein:{end}", label=end, title="Protein trên cầu nối (Tier 3)",
                                     color=COLORS["bridge"], shape="diamond", size=12)
                source = f"drug:{edge['source']}" if edge["source"] in drug_names else f"protein:{edge['source']}"
                target = f"drug:{edge['target']}" if edge["target"] in drug_names else f"protein:{edge['target']}"
                net.add_edge(source, target, color=COLORS["bridge"], dashes=edge["relation"] == "interacts_with",
                             title="tác động lên" if edge["relation"] == "targets" else "tương tác protein")

    if show_effects:
        for item in result["interactions"]:
            if not item["serious"]:
                continue
            node = f"effect:{item['side_effect_en']}"
            net.add_node(node, label=item["side_effect_vi"], title=f"{item['side_effect_vi']} ({item['side_effect_en']})\nTier {item['tier']}",
                         color=COLORS["effect"], shape="box", font={"color": "#ffffff"})
            for drug in sorted({name for pair in item["pairs"] for name in pair}):
                net.add_edge(f"drug:{drug}", node, color="#fca5a5", width=1)

    html = net.generate_html(notebook=False)
    # vis-network styles tooltips with white-space: nowrap, which would put
    # the multi-line titles above on a single line.
    return html.replace("</head>", "<style>.vis-tooltip{white-space:pre-line}</style></head>", 1)

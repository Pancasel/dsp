const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, BorderStyle, AlignmentType, Tab, PositionalTab, PositionalTabAlignment,
  PositionalTabLeader, ShadingType, Footer, PageNumber, LevelFormat,
} = require("docx");

const PAGE_WIDTH_DXA = 12240; // US Letter
const PAGE_HEIGHT_DXA = 15840;
const MARGIN = 1440; // 1 inch
const CONTENT_WIDTH = PAGE_WIDTH_DXA - 2 * MARGIN;

const FONT = "Times New Roman";

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    children: [new TextRun({ text, font: FONT, size: 22, ...opts })],
  });
}

// A paragraph built from mixed runs (plain / italic / code / superscript).
function rich(parts, opts = {}) {
  const runs = parts.map((part) => {
    if (typeof part === "string") {
      return new TextRun({ text: part, font: FONT, size: 22 });
    }
    const { text, italic, code, sup, bold } = part;
    return new TextRun({
      text,
      font: code ? "Consolas" : FONT,
      size: code ? 20 : 22,
      italics: !!italic,
      bold: !!bold,
      superScript: !!sup,
    });
  });
  return new Paragraph({ spacing: { after: 160, line: 276 }, ...opts, children: runs });
}

function heading(text, level, numbering) {
  return new Paragraph({
    heading: level,
    spacing: { before: 280, after: 160 },
    children: [
      new TextRun({ text: numbering ? `${numbering}  ${text}` : text, font: FONT, bold: true }),
    ],
  });
}

function caption(label, text) {
  return new Paragraph({
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text: `${label}: `, font: FONT, size: 20, bold: true }),
      new TextRun({ text, font: FONT, size: 20, bold: true })],
  });
}

function equationParagraph(text, number) {
  // IEEE style: centred equation, equation number right-aligned on the same line.
  return new Paragraph({
    tabStops: [
      { type: "center", position: CONTENT_WIDTH / 2 },
      { type: "right", position: CONTENT_WIDTH },
    ],
    spacing: { before: 160, after: 160 },
    children: [
      new TextRun({ text: "\t" + text + "\t", font: "Cambria Math", italics: true, size: 22 }),
      new TextRun({ text: `(${number})`, font: FONT, size: 22 }),
    ],
  });
}

function footnoteLine(marker, text) {
  return new Paragraph({
    spacing: { after: 80 },
    children: [
      new TextRun({ text: marker, font: FONT, size: 18, superScript: true }),
      new TextRun({ text: " " + text, font: FONT, size: 18 }),
    ],
  });
}

function cell(text, opts = {}) {
  const { width, bold = false, header = false, code = false } = opts;
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header ? { type: ShadingType.CLEAR, fill: "D9D9D9" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [
      new Paragraph({
        children: [
          new TextRun({
            text,
            font: code ? "Consolas" : FONT,
            size: code ? 19 : 20,
            bold: bold || header,
          }),
        ],
      }),
    ],
  });
}

function makeTable(columnWidths, rows, headerBold = true) {
  const total = columnWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "000000" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000" },
      left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "999999" },
      insideVertical: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    },
    rows: rows.map((r, i) =>
      new TableRow({
        children: r.map((text, j) =>
          cell(text, { width: columnWidths[j], header: headerBold && i === 0, code: j === 0 && i > 0 })
        ),
      })
    ),
  });
}

const children = [];

children.push(
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { after: 160 },
    children: [new TextRun({ text: "Chapter 4", font: FONT, bold: true })],
  }),
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { after: 240 },
    children: [new TextRun({ text: "Experiments and Results", font: FONT, bold: true })],
  }),
  heading("Implementation", HeadingLevel.HEADING_2, "4.1"),
  p(
    "This section describes how the Graph-RAG pipeline of Chapter 3 is realised as a working system: " +
    "the concrete data artefacts consumed at run time (4.1.1), the software architecture (4.1.2), the " +
    "retrieval and ranking parameters (4.1.3), how the large language model (LLM) is invoked and " +
    "prompted (4.1.4), and how its output is validated before being shown to a user (4.1.5)."
  ),

  heading("Data", HeadingLevel.HEADING_3, "4.1.1"),
  rich([
    "The knowledge graph is built directly from the Decagon dataset, distributed by the Stanford " +
    "SNAP BioSNAP collection", { text: "1", sup: true },
    ", as four plain CSV files; the fourth, ", { text: "bio-decagon-mono.csv", code: true },
    ", is optional and is used only as a ranking feature (Section 4.1.3), never for retrieval. " +
    "Table 4.1 reports the exact size of each file as distributed – obtained by loading all " +
    "four files with no filtering – and matches the totals already summarised in Section " +
    "1.3.2 up to the small differences expected from counting raw rows rather than deduplicated " +
    "entities.",
  ]),
  caption("Table 4.1", "Decagon dataset, as distributed (unfiltered row and entity counts)"),
  makeTable(
    [3600, 1500, 4840],
    [
      ["File", "Rows", "Key entities"],
      ["bio-decagon-targets.csv", "18,690", "284 drugs, 3,648 proteins"],
      ["bio-decagon-ppi.csv", "715,612", "19,081 proteins"],
      ["bio-decagon-combo.csv", "4,649,441", "645 drugs, 63,473 drug pairs, 1,317 effect codes"],
      ["bio-decagon-mono.csv (optional)", "174,977", "639 drugs, 10,184 effect codes"],
    ]
  ),
  p(""),
  rich([
    "Two structural properties of the graph directly motivated implementation choices in Section " +
    "4.1.3. First, both the drug–target and the protein–protein subgraphs are heavily " +
    "right-skewed (Table 4.2): half of all drugs bind at most 11 proteins, yet one outlier binds " +
    "2,840, and half of all proteins have at most 33 interaction partners, yet the busiest protein " +
    "has 2,519. Treating every protein identically during bridge search would let a handful of " +
    "such hub proteins connect almost any two drugs through a biologically uninformative path. " +
    "Fixing ", { text: "hub_degree_threshold", code: true },
    " at 500 (Table 4.5) removes only the extreme tail – 220 of 19,081 proteins (1.15%), " +
    "close to the 99th percentile of the observed degree distribution – while leaving the " +
    "rest of the PPI network intact.",
  ]),
  caption("Table 4.2", "Degree-distribution statistics of the drug–target and protein–protein graphs"),
  makeTable(
    [3300, 1100, 1100, 1100, 1100, 1100, 1140],
    [
      ["Degree", "Min", "Median", "Mean", "P90", "P99", "Max"],
      ["Targets per drug", "1", "11", "65.8", "170", "206", "2,840"],
      ["Drugs per target protein", "1", "1", "5.1", "5", "78", "94"],
      ["PPI neighbours per protein", "1", "33", "75.0", "201", "525", "2,519"],
    ]
  ),
  p(""),
  p(
    "Second, the 645 drugs that carry documented pairwise history in bio-decagon-combo.csv are " +
    "not the same population as the 284 drugs listed in bio-decagon-targets.csv: only 284 of the " +
    "645 (44%) have a known target set. For the remaining 361 drugs, Tier-1 evidence (a documented " +
    "pairwise effect, Table 3.2) is still available whenever the pair itself was tested, but Tier-2 " +
    "amplification and Tier-3 bridging are reachable only for drugs with target annotations, since " +
    "both require a protein-level path. This asymmetry is a property of the released data, not of " +
    "the retrieval algorithm, and bounds how much mechanistic – as opposed to purely " +
    "historical – evidence the pipeline can offer for target-unannotated drugs."
  ),
  p(
    "Third, the volume of documented pairwise history is itself heavy-tailed at two levels " +
    "(Table 4.3). A drug pair that has any documented effect has a median of 53 and up to 540 of " +
    "them, which is why effects are ranked (Eq. 1) rather than listed in full; and a given effect " +
    "code is documented for a median of 1,596 pairs but as few as 1, which is why the " +
    "inverse-frequency term of Eq. 2 matters in practice. Of the 1,317 distinct effect codes " +
    "present in the raw file, 963 occur in at least 500 pairs – matching, within one effect, " +
    "the 964 side effects that Zitnik et al. [1] report evaluating after applying the same " +
    "frequency cut-off – while the remaining ~350 codes are comparatively rare and receive a " +
    "correspondingly larger I(e) rather than being dropped."
  ),
  caption("Table 4.3", "Side-effect frequency statistics in bio-decagon-combo.csv"),
  makeTable(
    [3300, 1100, 1100, 1100, 1100, 1100, 1140],
    [
      ["", "Min", "Median", "Mean", "P90", "P99", "Max"],
      ["Effects per documented pair", "1", "53", "73.3", "165", "305", "540"],
      ["Pairs documenting a given effect", "1", "1,596", "3,530.3", "9,530", "21,396", "28,568"],
    ]
  ),
  p(""),
  rich([
    "These characteristics also shape how the raw files are turned into a queryable graph at run " +
    "time. Drug–target and protein–protein interaction edges are loaded once into " +
    "in-memory adjacency maps (", { text: "dict[str, set[str]]", code: true },
    "), giving O(1) neighbour look-ups during retrieval. The combination file, at 4,649,441 rows, " +
    "is not re-parsed on every run: it is materialised once into a local SQLite cache (",
    { text: "side_effects(drug_a, drug_b, effect_code, effect_name)", code: true },
    ", stored ", { text: "WITHOUT ROWID", code: true },
    " with a composite primary key, plus an index on ", { text: "drug_b", code: true },
    " for symmetric pair look-ups), together with two derived tables used directly by the ranking " +
    "formula of Section 4.1.3: ", { text: "effect_counts", code: true },
    " (n_e, the number of pairs documenting each effect) and ", { text: "stats", code: true },
    " (N, the total number of distinct documented pairs). A SHA-256 signature of each source " +
    "file's size and modification time, together with a cache-schema version, is stored in a ",
    { text: "metadata", code: true },
    " table; the cache is rebuilt automatically only when a source file or the schema changes, " +
    "which keeps repeated start-ups fast and avoids re-reading millions of rows on every process " +
    "launch.",
  ]),
  rich([
    "A free-text drug name typed by a user – including Vietnamese brand names, typos, or " +
    "several drugs written in one sentence – is resolved to a Decagon STITCH identifier by a " +
    "dedicated resolver. Resolution is attempted, in order, against (i) a STITCH-ID pattern, (ii) an " +
    "exact match on normalised generic names, synonyms, or Vietnamese brand aliases, and (iii) fuzzy " +
    "string matching (RapidFuzz", { text: "2", sup: true }, ", indel ratio), accepted automatically " +
    "above a score of 90 and otherwise offered only as a suggestion above 75. A curated stop-word " +
    "list removes conversational filler (dosage, timing, and symptom words, in both Vietnamese and " +
    "English) before matching, so that a message such as “em đang uống panadol với " +
    "ibuprofen” resolves to the two intended drugs rather than being treated as one unmatched " +
    "phrase. Name resolution and prompt serialisation draw on five supporting resource tables, built " +
    "offline by the ", { text: "scripts/", code: true },
    " utilities: canonical generic names per STITCH ID, Vietnamese brand aliases, gene symbols, " +
    "Vietnamese translations of effect names, and two curated lists marking EMA Important Medical " +
    "Events and life-threatening effects.",
  ]),

  heading("Software Architecture", HeadingLevel.HEADING_3, "4.1.2"),
  rich([
    "The system is implemented in Python. Table 4.4 summarises the responsibility of each module " +
    "under ", { text: "graphrag/", code: true }, "; ", { text: "app.py", code: true },
    " is a Streamlit", { text: "3", sup: true },
    " front-end that calls the single orchestration entry point, ",
    { text: "PolypharmacySystem.predict()", code: true },
    ". Configuration – file paths, the LLM endpoint and model, and the retrieval and ranking " +
    "parameters of Table 4.5 – is read from environment variables through a ",
    { text: ".env", code: true },
    " file, so the retrieval bounds and the model used for generation can be changed without " +
    "editing code. External dependencies are kept deliberately small: ", { text: "httpx", code: true },
    " for the synchronous HTTP call to the LLM provider, ", { text: "pydantic", code: true },
    " for schema validation of the LLM's output, ", { text: "rapidfuzz", code: true },
    " for name resolution, and ", { text: "pyvis", code: true },
    " for rendering the interaction graph in the user interface.",
  ]),
  caption("Table 4.4", "Module responsibilities"),
  makeTable(
    [2600, 7140],
    [
      ["Module", "Responsibility"],
      ["config.py", "Environment-driven settings, ranking weights"],
      ["dataset.py", "Graph loading, SQLite cache, subgraph retrieval"],
      ["names.py / resolver.py", "Name normalisation and drug-name resolution"],
      ["ranking.py", "Regimen-level effect scoring (Sec. 4.1.3)"],
      ["llm.py", "Prompt construction, API call, schema validation"],
      ["presenter.py", "Assembling evidence into UI-ready structures"],
      ["pipeline.py", "PolypharmacySystem: end-to-end orchestration"],
      ["visualize.py", "Interaction-graph rendering (PyVis)"],
    ]
  ),
  p(""),

  heading("Subgraph Retrieval, Pruning, and Ranking", HeadingLevel.HEADING_3, "4.1.3"),
  rich([
    "Retrieval follows the six steps of Table 3.1, subject to the bounds in Table 4.5. A protein is " +
    "pruned as a generic hub, and excluded from any bridge, once its number of PPI neighbours " +
    "exceeds ", { text: "hub_degree_threshold", code: true },
    "; a direct target of a queried drug is always retained regardless of degree. Tier-3 bridge " +
    "search is bounded per drug pair (", { text: "tier3_one_hop_per_pair", code: true }, ", ",
    { text: "tier3_two_hop_per_pair", code: true },
    ") and stops as soon as both caps are full and one further candidate has confirmed that more " +
    "bridges exist, so the search is deterministic and never silently under-reports truncation. The " +
    "total evidence serialised to the LLM is capped at ", { text: "max_evidence", code: true },
    " and allocated across drug pairs by round-robin selection, so that a single high-degree pair " +
    "cannot consume the whole budget. Documented evidence is built only for the effects that will be " +
    "shown, and Tier 2 and Tier 3 each have a reserved quarter of the budget, so Tier 1 cannot starve " +
    "them; unused slots go to the other tiers.",
  ]),
  caption("Table 4.5", "Retrieval and pruning bounds (default values)"),
  makeTable(
    [2800, 1200, 5740],
    [
      ["Parameter", "Default", "Role"],
      ["hub_degree_threshold", "500", "PPI degree above which a protein is pruned"],
      ["max_evidence", "400", "Cap on evidence items serialised to the LLM"],
      ["tier3_one_hop_per_pair", "8", "Max. 1-hop bridges sampled per undocumented pair"],
      ["tier3_two_hop_per_pair", "4", "Max. 2-hop bridges sampled per undocumented pair"],
      ["tier3_candidates_per_pair", "20", "Max. candidate labels offered for a Tier-3 pair"],
      ["max_serious/other_effects", "10 / 5", "Effects surfaced per analysis after ranking"],
    ]
  ),
  p(""),
  p(
    "Because a documented drug pair can list dozens to hundreds of side effects (a median of 53, " +
    "up to 540, in Decagon), the effects of each pair are never shown unranked. Every documented " +
    "effect e in the regimen receives a transparent, additive score:"
  ),
  equationParagraph("score(e) = w_r·R(e) + w_i·I(e) + w_m·M(e) + w_s·S(e) + w_c·[S(e)∧C(e)] + w_a·A(e)", "1"),
  rich([
    "where R(e) is the number of regimen drug pairs that document e (recurrence); M(e), S(e), and " +
    "C(e) are indicators for “absent from either drug's individual side-effect list”, " +
    "“an EMA Important Medical Event”, and “additionally life-threatening”; A(e) " +
    "indicates that a pair documenting e also shares a protein target with a third regimen drug " +
    "(the Tier-2 trigger of Table 3.2); and I(e) is an inverse pair-frequency term:",
  ]),
  equationParagraph("I(e) = log( N / max(n_e, 1) )", "2"),
  p(
    "with N the total number of distinct drug pairs documented in Decagon and n_e the number of " +
    "pairs documenting e, so that effects specific to a few pairs are weighted above ubiquitous " +
    "ones. The six weights default to w_r = 1.0, w_i = 0.5, w_m = 1.0, w_s = 2.0, w_c = 2.0, " +
    "w_a = 1.0, and are exposed through Settings.ranking so that they can be varied in an " +
    "ablation. Ranked effects determine both the order in which documented effects are offered as " +
    "evidence and, through A(e), which pairs are eligible for a Tier-2 prediction."
  ),

  heading("Large Language Model Usage", HeadingLevel.HEADING_3, "4.1.4"),
  rich([
    "Table 4.6 lists the request configuration. The LLM is accessed through OpenRouter's",
    { text: "4", sup: true },
    " OpenAI-compatible ", { text: "/chat/completions", code: true },
    " endpoint, which lets the model be swapped (via the ", { text: "LLM_MODEL", code: true },
    " environment variable) without changing application code; the default model used in this " +
    "project is Google's ", { text: "gemini-2.5-flash", code: true }, { text: "5", sup: true },
    ". The model is deliberately used as a constrained extraction-and-explanation layer over " +
    "evidence the pipeline has already retrieved, ranked, and bounded (Section 4.1.3), rather than " +
    "as an open-ended reasoner over the raw graph: the request asks for ",
    { text: "json_object", code: true },
    " output and disables the model's own extended reasoning (",
    { text: "reasoning: {\"effort\": \"none\", \"exclude\": true}", code: true },
    "), which reduces latency and cost without affecting prediction quality, since the biological " +
    "reasoning – which proteins are shared, which bridges exist, which pairs already have " +
    "documented history – is already encoded in the serialised evidence rather than left for " +
    "the model to derive on its own.",
  ]),
  caption("Table 4.6", "LLM request configuration"),
  makeTable(
    [3200, 6540],
    [
      ["Parameter", "Value"],
      ["Endpoint", "OpenRouter, /chat/completions (OpenAI-compatible)"],
      ["Default model", "google/gemini-2.5-flash (overridable via LLM_MODEL)"],
      ["response_format", "json_object"],
      ["reasoning", '{"effort": "none", "exclude": true}'],
      ["max_tokens", "4000"],
      ["Timeout", "120 s (15 s connect)"],
      ["Retries", "up to 3, exponential back-off, on connection failure only"],
    ]
  ),
  p(""),
  rich([
    "The system prompt specifies the full output contract rather than leaving it to free text: a " +
    "JSON object with a ", { text: "predictions", code: true },
    " array whose items carry ", { text: "side_effect", code: true }, ", ",
    { text: "tier", code: true }, " (1–3), one or more ", { text: "evidence_ids", code: true },
    ", and a Vietnamese, at-most-three-sentence ", { text: "explanation", code: true },
    " naming the drugs and, where relevant, the shared or bridging protein targets. The prompt " +
    "restates the tier semantics of Table 3.2 as hard constraints on evidence use: a Tier-1 " +
    "prediction must reproduce a documented pairwise effect; a Tier-2 prediction must reproduce a " +
    "documented pairwise effect additionally supported by a target shared by at least three " +
    "regimen drugs; a Tier-3 prediction is only a hypothesis, permitted solely when a retrieved " +
    "bridge links the pair and that pair has no documented effect, and its ",
    { text: "side_effect", code: true },
    " label must be copied verbatim from that bridge's ",
    { text: "candidate_side_effects", code: true },
    ". Output is capped at 15 predictions, prioritising Tier 1, then Tier 2, then at most two " +
    "Tier-3 hypotheses, and the model is instructed to return an empty ",
    { text: "predictions", code: true },
    " array rather than invent evidence when support is insufficient.",
  ]),

  heading("Grounding, Validation, and Fallback Behaviour", HeadingLevel.HEADING_3, "4.1.5"),
  rich([
    "The raw response is parsed with a strict pydantic", { text: "6", sup: true },
    " schema (", { text: "ModelPrediction", code: true }, "/", { text: "PredictionList", code: true },
    "); if that fails because the model wrapped the JSON in a fenced code block or stopped " +
    "mid-stream, a tolerant recovery path scans the text with a streaming JSON decoder and keeps " +
    "every complete, schema-valid object found before the cut-off, rather than discarding the " +
    "whole response.",
  ]),
  p(
    "Parsing success does not make a prediction trustworthy: every item is re-validated against " +
    "the actually retrieved subgraph, independently of the model's own claims. A prediction is " +
    "kept only if every cited evidence_id exists, carries the declared tier, and (Tier 1/2) names " +
    "the same side-effect label on every cited item, or (Tier 3) lists that label among the cited " +
    "bridge's own candidates; the underlying graph edges of each cited path are additionally " +
    "checked against the loaded target, PPI, and side-effect tables, and duplicate (tier, label, " +
    "evidence-set) predictions are removed. A prediction failing any check is silently dropped " +
    "rather than shown to the user. This two-stage check – schema validation, then graph " +
    "grounding – is what allows Tier-2 and Tier-3 output to be treated as traceable rather " +
    "than as an unverified model claim."
  ),
  p(
    "Because the LLM call can fail for several independent reasons (no API key configured, a " +
    "malformed or non-conforming JSON body, an HTTP error from the provider, or a network " +
    "failure), the pipeline treats all of them as a single degraded mode: llm_status is set to " +
    "\"fallback\" with a diagnostic message, and the request still returns the retrieved and " +
    "ranked subgraph, interaction graph, and coverage summary without raising an exception, so a " +
    "user still sees the underlying evidence even when no generated explanation is available."
  ),

  new Paragraph({
    border: { top: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 4 } },
    spacing: { before: 400 },
    children: [new TextRun({ text: "", font: FONT })],
  }),
  footnoteLine("1", "Decagon dataset (Stanford SNAP BioSNAP), https://snap.stanford.edu/decagon/"),
  footnoteLine("2", "RapidFuzz, https://github.com/rapidfuzz/RapidFuzz"),
  footnoteLine("3", "Streamlit, https://streamlit.io"),
  footnoteLine("4", "OpenRouter, https://openrouter.ai"),
  footnoteLine("5", "Gemini API model index, https://ai.google.dev/gemini-api/docs/models"),
  footnoteLine("6", "Pydantic, https://docs.pydantic.dev"),
);

const doc = new Document({
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_WIDTH_DXA, height: PAGE_HEIGHT_DXA },
          margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
        },
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              tabStops: [{ type: "right", position: CONTENT_WIDTH }],
              children: [
                new TextRun({ text: "Polypharmacy Side-Effect Prediction", font: FONT, size: 18 }),
                new TextRun({ text: "\t", font: FONT }),
                new TextRun({ text: "Group 8", font: FONT, size: 18 }),
              ],
            }),
          ],
        }),
      },
      children,
    },
  ],
  styles: {
    default: {
      document: { run: { font: FONT, size: 22 } },
      heading1: { run: { font: FONT, size: 32, bold: true }, paragraph: { spacing: { before: 240, after: 160 } } },
      heading2: { run: { font: FONT, size: 28, bold: true }, paragraph: { spacing: { before: 240, after: 140 } } },
      heading3: { run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 200, after: 120 } } },
    },
  },
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(__dirname + "/chapter4_implementation.docx", buf);
  console.log("written");
});

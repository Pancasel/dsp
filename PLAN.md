# PLAN: Graph-RAG cảnh báo tác dụng phụ khi phối hợp ≥ 3 thuốc

## 0. Bài toán

Người dùng **chat** để nêu các thuốc đang dùng (≥ 3), bằng tên hoạt chất hoặc biệt dược Việt Nam, viết thành câu tự nhiên. Hệ thống trả về:

1. **Từng thuốc là gì**: tên generic, biệt dược, protein đích, tác dụng phụ khi dùng riêng.
2. **Tác dụng phụ khi phối hợp**, chia 3 tầng theo report:
   - **Tier 1, Đã ghi nhận**: có trong dữ liệu cho một cặp thuốc của phác đồ.
   - **Tier 2, Khuếch đại**: đã ghi nhận, và cặp đó có protein đích chung với thuốc thứ ba.
   - **Tier 3, Giả thuyết**: cặp thuốc chưa có dữ liệu nhưng nối với nhau qua cầu protein (≤ 2 bước).

Mỗi kết quả kèm đường dẫn bằng chứng trong đồ thị và lời giải thích tiếng Việt.

Tiêu chí ưu tiên: **kết quả phải đúng và truy vết được về dữ liệu gốc**. Tiếng Việt chỉ là phần hỗ trợ.

Ràng buộc kỹ thuật:
- Chạy local.
- LLM gọi qua OpenRouter, **1 request cho mỗi lần phân tích**.
- UI chỉ dùng Streamlit, gồm khung chat và đồ thị quan hệ có thể bật/tắt; không có server API riêng.
- Không có phần chụp ảnh (đã bỏ theo yêu cầu).

---

## 1. Dữ liệu (đã đo trên CSV thật)

| File | Số liệu thực đo | Hệ quả |
|---|---|---|
| `bio-decagon-combo.csv` | 4,649,441 dòng; 645 thuốc; 63,473 cặp có dữ liệu; 1,317 loại tác dụng phụ | Chỉ 30.6% trong 207,690 cặp khả dĩ có dữ liệu. Trung vị 53 tác dụng phụ/cặp (tối đa 540) → bắt buộc phải xếp hạng |
| `bio-decagon-mono.csv` | 639 thuốc; 9,702 loại tác dụng phụ đơn lẻ | Dùng cho thẻ thông tin từng thuốc và để lọc hiệu ứng thật sự do phối hợp |
| `bio-decagon-targets.csv` | **chỉ 284 thuốc** có protein đích | 361/645 thuốc không có Tier 2/3, chỉ ra được Tier 1 |
| `bio-decagon-ppi.csv` | 715,612 cạnh, khoảng 19k protein | Đủ cho cầu protein 1–2 bước |

**Kết luận:** dữ liệu đủ cho phạm vi 645 thuốc. Data gốc không có tên thuốc nên đã bổ sung (lưu trong `resources/`, runtime chạy offline):

| Bổ sung | Cách làm | Kết quả |
|---|---|---|
| Tên thuốc | STITCH `CID0…` → PubChem CID; dự phòng bằng tên SIDER; sửa tay có ghi bằng chứng | 645/645 thuốc có tên generic đọc được (18 CID đã bị PubChem gộp được tra qua "Preferred Compound"; 36 tên tạp chất/biệt dược/racemic được chuẩn hoá về INN, ví dụ Dichloromethanediamine → Cisplatin, Clexane → Enoxaparin; bằng chứng trong `drug_name_overrides.csv`) |
| Biệt dược Việt Nam | Bảng viết tay, có script kiểm tra | 123 biệt dược, 0 lỗi |
| Tên gene | NCBI `Homo_sapiens.gene_info` | 18,941/19,089 gene |
| Tiếng Việt | Agent dịch trực tiếp, không gọi API | 1,317 tên phối hợp + 1,320 tên đơn lẻ có hiện trên UI |

**Giới hạn phải ghi rõ:**
- Dữ liệu không có liều, tần suất hay mức độ nặng.
- Dữ liệu tổng hợp từ báo cáo FAERS/TWOSIDES nên có nhiễu; danh sách mono còn lẫn cả thủ thuật và xét nghiệm.
- Thuốc ngoài 645 thuốc này được báo "không có trong dữ liệu", hệ thống không đoán.

**Số liệu cần sửa trong report (mục 1.3.2):**
- targets chỉ phủ **284** thuốc, không phải ~645;
- file combo có **1,317** loại tác dụng phụ (964 là tập đã lọc trong bài báo Decagon);
- bổ sung: chỉ 30.6% cặp thuốc có dữ liệu.

---

## 2. Kiến trúc

```
streamlit run app.py
  │
  ├─ PolypharmacySystem.load()            (cache bằng st.cache_resource)
  │    DecagonGraph (SQLite cache + dict targets/PPI) + NameIndex + DrugResolver
  │
  ├─ resolver.extract(tin nhắn chat)      → tách câu, bỏ từ đệm, khớp chính xác/biệt dược
  │                                          → tên dài nhất có trong câu → fuzzy (chưa chắc thì hỏi lại)
  └─ predict([≥3 thuốc])
       retrieve: targets + protein chung + cầu PPI ≤ 2 bước + lịch sử cặp
       → bỏ hub (bậc PPI > HUB_DEGREE_THRESHOLD)
       → xếp hạng tác dụng phụ (ranking.py)
       → đóng toàn bộ evidence vào MỘT JSON → 1 request LLM
       → validator loại mọi dự đoán không khớp evidence
       → chọn tối đa 10 tác dụng phụ nghiêm trọng + 5 tác dụng phụ khác
       → gắn tên thật, tiếng Việt, đường dẫn bằng chứng
  └─ visualize.build_graph_html(result)   → đồ thị pyvis (thuốc, cặp, protein, tác dụng phụ)
       (không có LLM / LLM lỗi → giải thích mẫu, kết quả vẫn trả về đầy đủ)
```

```
polypharmacy-graph-rag/
├── app.py                  # UI chat Streamlit + khung đồ thị
├── graphrag/
│   ├── config.py           # Settings từ .env
│   ├── dataset.py          # DecagonGraph: cache SQLite, retrieve, bridge, ứng viên Tier 3
│   ├── ranking.py          # điểm xếp hạng tác dụng phụ
│   ├── names.py            # tên thuốc/gene, tiếng Việt, từ khóa nghiêm trọng
│   ├── resolver.py         # chuỗi người dùng nhập → STITCH ID
│   ├── llm.py              # prompt + gọi LLM + validator
│   ├── visualize.py        # đồ thị quan hệ (pyvis)
│   ├── presenter.py        # context gửi LLM, giải thích mẫu, kết quả hiển thị
│   └── pipeline.py         # PolypharmacySystem
├── resources/              # CSV tên/biệt dược/gene/tiếng Việt/từ khóa (commit)
├── scripts/                # sinh resources (chạy 1 lần)
├── eval/                   # thí nghiệm E1–E6, đánh giá độ chính xác A1–A4
└── tests/
```

---

## 3. Thiết kế chính

**Resolver**
- Chuẩn hóa chuỗi nhập: bỏ dấu, bỏ hàm lượng ("500mg", "250"), bỏ dạng bào chế và tên muối ("kali", "natri", "HCl").
- Fuzzy chỉ tự nhận khi điểm ≥ 90, độ giống ký tự ≥ 92 và ứng viên thứ nhất bỏ xa ứng viên thứ hai. Nếu không, hệ thống hỏi lại người dùng.
- Ví dụ: "esomeprazole" không bao giờ bị nhận thành Omeprazole, "Vitamin C" không bị nhận thành Ergocalciferol.
- Trong câu chat, một từ đơn chỉ được nhận là thuốc nếu là tên chính, biệt dược đã soát, hoặc đồng nghĩa ≥ 5 ký tự. Viết tắt PubChem ngắn ("CoM") và từ thông dụng ("vitamin", "delta") không được tự khớp: trước đây "ăn cơm" bị nhận thành Mesna.

**Hiển thị:** không cho người dùng chọn số lượng. Hệ thống hiện tối đa `MAX_SERIOUS_EFFECTS` = 10 tác dụng phụ nghiêm trọng có điểm cao nhất. Tối đa `MAX_OTHER_EFFECTS` = 5 tác dụng phụ khác nằm trong một mục thu gọn.

**Xếp hạng**

```
score = 1.0·recurrence   (số cặp trong phác đồ cùng ghi nhận)
      + 0.5·idf          (log(tổng số cặp / số cặp có tác dụng phụ này))
      + 1.0·not_in_mono  (không có trong mono của thuốc nào trong cặp)
      + 2.0·serious      (Important Medical Event của EMA, resources/serious_effects.csv)
      + 2.0·critical     (nghiêm trọng VÀ thuộc nhóm đe dọa tính mạng trong serious_keywords.csv)
      + 1.0·amplified    (Tier 2)
```
Các trọng số đặt trong `.env`. Tier 2 chỉ áp cho đúng các cặp khuếch đại (`amplified_pairs`).

**Thế nào là "nghiêm trọng"**
- Chuẩn: danh sách *Important Medical Event* (IME) của EMA (MedDRA 29.1). `scripts/build_serious_effects.py` ánh xạ từng tên Decagon sang thuật ngữ MedDRA: khớp chính xác sau chuẩn hóa (Anh–Mỹ, số nhiều), khớp khi đảo thứ tự từ, hoặc qua `resources/effect_meddra_map.csv` cho tên thông dụng ("heart attack" → Myocardial infarction). Có trong IME mới là nghiêm trọng.
- Loại các thuật ngữ IME mà FAERS thường ghi như chỉ định hay bệnh nền: ung thư, dị tật bẩm sinh, thai sản, nhiễm trùng cộng đồng (viêm gan B/C, giang mai…), bệnh mạn tính (viêm khớp dạng thấp, đa xơ cứng…).
- Tên không ánh xạ được thì dùng từ khóa (`serious_keywords.csv`), có nhận số nhiều và loại trừ xét nghiệm, thủ thuật ("glasgow coma scale", "hepatitis B antibody positive").
- Trước đây chỉ dùng từ khóa nên bỏ sót "gastrointestinal bleed", "heart attack", "Embolism pulmonary", "deep vein thromboses", "septic shock" và gắn nhầm "Hepatitis C antibody positive", "heat stroke" (đánh giá A6).

**Tier 3**
- Nhãn chỉ được chọn trong danh sách ứng viên của chính cặp đó. Ứng viên là các tác dụng phụ mỗi thuốc đã có với những thuốc khác; ưu tiên nhãn chung của cả hai, rồi theo tổng số lần.
- Validator loại mọi nhãn Tier 3 nằm ngoài danh sách ứng viên.

**LLM**
- Mỗi lần phân tích gửi đúng 1 request, kèm toàn bộ evidence dưới dạng JSON (tên thật kèm ID, top tác dụng phụ, đường dẫn).
- LLM trả JSON. Dự đoán nào trích evidence ID không tồn tại, sai tier hoặc sai nhãn đều bị loại.

---

## 4. Thí nghiệm (chương 4 của report)

| Mã | Nội dung | Trạng thái / kết quả |
|---|---|---|
| E1 | Resolver trên 71 input (generic, biệt dược, kèm hàm lượng, sai chính tả, không có trong data) | Đúng 98.6%, nhận sai 0. Ca duy nhất không tự nhận ("diazepan") thì hệ thống hỏi lại |
| E3 | Tỷ lệ dự đoán của LLM qua được validator | Chưa chạy, cần API: `eval/e3_llm_groundedness.py --yes-call-api` |
| E4 | Ứng viên Tier 3 trên 2000 cặp bị ẩn | precision@10: đồ thị 0.507, phổ biến toàn cục 0.418, ngẫu nhiên 0.065. Hit@10: 0.966 / 0.946 / 0.420 |
| E5 | Ngưỡng hub 100 → không giới hạn | Số cầu Tier 3 trung bình tăng 3.6 → 4.4, số hub bị loại giảm 41.5 → 0. Ngưỡng 500 giữ 98% số cầu |
| E6 | Thời gian truy xuất | Trung bình: k=3 là 0.02–0.04 s, k=5 là 0.05–0.12 s, k=8 là 0.12–0.26 s (dao động theo tải máy) |
| A1 | Đối chiếu đầu ra của 200 phác đồ với CSV gốc (`eval/run_accuracy.py`) | Tier 1 precision = recall = 100% (52,880); Tier 2 72/72; mọi đường dẫn Tier 1/2/3 đúng từng cạnh (40,529) |
| A2 | Cầu protein có dự đoán được cặp có tương tác? | AUROC 0.543 (gần ngẫu nhiên 0.495); baseline độ phổ biến 0.943. Cầu có ở 95.7% số cặp → Tier 3 chỉ là giả thuyết cơ chế |
| A3 | Dự đoán từng tác dụng phụ, giao thức Decagon (963 TDP ≥ 500 cặp) | AUROC 0.951, AUPRC 0.940, AP@50 0.966; baseline độ phổ biến 0.908/0.889/0.883; Decagon công bố 0.872/0.832/0.803 (giao thức khác, chỉ tham chiếu) |
| A4 | Validator với dự đoán bị làm sai có chủ đích | Loại 950/950 dự đoán sai, giữ 300/300 dự đoán hợp lệ |
| E2 | Trích thuốc từ 30 câu chat (`eval/extract_cases.csv`) | Đúng 30/30 câu, precision = recall thuốc = 100%, 2 câu hỏi lại |
| A5 | 26 tương tác kinh điển (`eval/clinical_benchmark.csv`) | Hit@10 53.8%, Hit@3 23.1%, MRR 0.22; ngang phương án chỉ dùng từ khóa (53.8% / 30.8%); chỉ IME không cộng nhóm đe dọa tính mạng: 42.3% |
| A6 | Phân loại "nghiêm trọng" so với IME (2,998 tên) | Từ khóa cũ precision 0.89 / recall 0.10; từ khóa mới 0.92 / 0.15 → hệ thống dùng nhãn IME |

Chi tiết, bảng và hình cho report: `eval/results/ACCURACY.md`.

---

## 5. Giới hạn đã đo được

- **Xếp hạng (A5):** khoảng một nửa số tương tác kinh điển lên top 10. Decagon không có tần suất hay mức độ, nên không phân biệt được tác dụng phụ đặc trưng của tương tác với hàng trăm tác dụng phụ khác của cùng cặp. Tụt huyết áp (sildenafil + nitrat) không nằm trong IME nên không hiện ở nhóm nghiêm trọng.
- **Tier 3 (A2):** cầu protein có ở ~96% số cặp và gần như không dự đoán được tương tác; chỉ dùng làm giả thuyết cơ chế.
- **A3 so với Decagon:** cặp âm ngẫu nhiên dễ phân biệt (baseline độ phổ biến đã đạt 0.91), nên không kết luận là tốt hơn mô hình GCN.

## 6. Việc còn lại

- [ ] Chạy thử với LLM thật và soát chất lượng giải thích.
- [ ] E3 (khoảng 10 request).
- [ ] Người có chuyên môn soát bản dịch và bảng ánh xạ `effect_meddra_map.csv`.
- [ ] Sửa số liệu trong report theo mục 1.
- [ ] Nếu có số lượng báo cáo FAERS: thay IDF bằng disproportionality (PRR/ROR) để xếp hạng.

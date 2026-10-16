# PLAN: Graph-RAG cảnh báo tác dụng phụ khi phối hợp ≥ 3 thuốc

## 0. Bài toán

Người dùng nhập **≥ 3 thuốc** bằng một trong ba cách: gõ tên hoạt chất, gõ biệt dược Việt Nam, hoặc chụp ảnh hộp thuốc. Hệ thống trả về:

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
- Đọc ảnh bằng một vision model (có thể dùng nhà cung cấp riêng).
- UI chỉ dùng Streamlit, không có server API riêng.

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
| Tên thuốc | STITCH `CID0…` → PubChem CID; dự phòng bằng tên SIDER; sửa tay có ghi bằng chứng | 634/645 thuốc có tên |
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
  ├─ search(text) / identify_image(ảnh)   → resolver: STITCH ID → khớp chính xác/biệt dược → fuzzy
  │                                          (ảnh: 1 request tới vision model, chỉ để đọc chữ)
  └─ predict([≥3 thuốc])
       retrieve: targets + protein chung + cầu PPI ≤ 2 bước + lịch sử cặp
       → bỏ hub (bậc PPI > HUB_DEGREE_THRESHOLD)
       → xếp hạng tác dụng phụ (ranking.py)
       → đóng toàn bộ evidence vào MỘT JSON → 1 request LLM
       → validator loại mọi dự đoán không khớp evidence
       → gắn tên thật, tiếng Việt, đường dẫn bằng chứng
       (không có LLM / LLM lỗi → giải thích mẫu, kết quả vẫn trả về đầy đủ)
```

```
polypharmacy-graph-rag/
├── app.py                  # UI Streamlit
├── graphrag/
│   ├── config.py           # Settings từ .env
│   ├── dataset.py          # DecagonGraph: cache SQLite, retrieve, bridge, ứng viên Tier 3
│   ├── ranking.py          # điểm xếp hạng tác dụng phụ
│   ├── names.py            # tên thuốc/gene, tiếng Việt, từ khóa nghiêm trọng
│   ├── resolver.py         # chuỗi người dùng nhập → STITCH ID
│   ├── llm.py              # prompt + gọi LLM + validator
│   ├── vision.py           # ảnh → tên thuốc
│   ├── presenter.py        # context gửi LLM, giải thích mẫu, kết quả hiển thị
│   └── pipeline.py         # PolypharmacySystem
├── resources/              # CSV tên/biệt dược/gene/tiếng Việt/từ khóa (commit)
├── scripts/                # sinh resources (chạy 1 lần)
├── eval/                   # thí nghiệm E1–E6
└── tests/
```

---

## 3. Thiết kế chính

**Resolver**
- Chuẩn hóa chuỗi nhập: bỏ dấu, bỏ hàm lượng ("500mg", "250"), bỏ dạng bào chế và tên muối ("kali", "natri", "HCl").
- Fuzzy chỉ tự nhận khi điểm ≥ 90, độ giống ký tự ≥ 92 và ứng viên thứ nhất bỏ xa ứng viên thứ hai. Nếu không, hệ thống hỏi lại người dùng.
- Ví dụ: "esomeprazole" không bao giờ bị nhận thành Omeprazole, "Vitamin C" không bị nhận thành Ergocalciferol.

**Xếp hạng**

```
score = 1.0·recurrence   (số cặp trong phác đồ cùng ghi nhận)
      + 0.5·idf          (log(tổng số cặp / số cặp có tác dụng phụ này))
      + 1.0·not_in_mono  (không có trong mono của thuốc nào trong cặp)
      + 2.0·serious      (khớp nguyên từ trong serious_keywords.csv)
      + 1.0·amplified    (Tier 2)
```
Các trọng số đặt trong `.env`. Tier 2 chỉ áp cho đúng các cặp khuếch đại (`amplified_pairs`).

**Tier 3**
- Nhãn chỉ được chọn trong danh sách ứng viên của chính cặp đó. Ứng viên là các tác dụng phụ mỗi thuốc đã có với những thuốc khác; ưu tiên nhãn chung của cả hai, rồi theo tổng số lần.
- Validator loại mọi nhãn Tier 3 nằm ngoài danh sách ứng viên.

**LLM**
- Mỗi lần phân tích gửi đúng 1 request, kèm toàn bộ evidence dưới dạng JSON (tên thật kèm ID, top tác dụng phụ, đường dẫn).
- LLM trả JSON. Dự đoán nào trích evidence ID không tồn tại, sai tier hoặc sai nhãn đều bị loại.

**Vision**
- Dùng các biến `VISION_MODEL`, `VISION_BASE_URL`, `VISION_API_KEY`. Nếu để trống thì dùng lại cấu hình `LLM_*`.
- Ảnh được thu nhỏ còn cạnh dài ≤ 1600px trước khi gửi.
- Mọi tên đọc được đều đi qua resolver, và người dùng xác nhận trước khi thêm vào phác đồ.

---

## 4. Thí nghiệm (chương 4 của report)

| Mã | Nội dung | Trạng thái / kết quả |
|---|---|---|
| E1 | Resolver trên 70 input (generic, biệt dược, kèm hàm lượng, sai chính tả, không có trong data) | Đúng 98.6%, nhận sai 0. Ca duy nhất không tự nhận ("diazepan") thì hệ thống hỏi lại |
| E2 | Đọc ảnh khoảng 30 hộp/vỉ thuốc | Chưa chạy, cần API vision |
| E3 | Tỷ lệ dự đoán của LLM qua được validator | Chưa chạy, cần API: `eval/e3_llm_groundedness.py --yes-call-api` |
| E4 | Ứng viên Tier 3 trên 2000 cặp bị ẩn | precision@10: đồ thị 0.507, phổ biến toàn cục 0.418, ngẫu nhiên 0.065. Hit@10: 0.966 / 0.946 / 0.420 |
| E5 | Ngưỡng hub 100 → không giới hạn | Số cầu Tier 3 trung bình tăng 3.6 → 4.4, số hub bị loại giảm 41.5 → 0. Ngưỡng 500 giữ 98% số cầu |
| E6 | Thời gian truy xuất | Trung bình: k=3 là 0.02 s, k=5 là 0.05 s, k=8 là 0.12 s (p95 với k=8: 0.29 s) |

**Kiểm tra tính đúng trên dữ liệu thật** (đối chiếu trực tiếp với CSV gốc):
- Mọi cặp và tác dụng phụ trả về đều có trong `bio-decagon-combo.csv`: 0 lỗi trên 4 phác đồ.
- Tier 2 chỉ gắn cho cặp có protein đích chung với ≥ 3 thuốc: 0 lỗi.
- Đường dẫn Tier 3 được kiểm từng cạnh với targets/PPI, ví dụ Bisoprolol → ADRB1 ↔ GCG ↔ CHRM3 ← Loratadine.

---

## 5. Việc còn lại

- [ ] Chạy thử với LLM thật và soát chất lượng giải thích.
- [ ] E3 (khoảng 10 request).
- [ ] E2: cấu hình vision model rồi thử bằng ảnh thật. Nếu kết quả không ổn thì bỏ phần ảnh (chỉ cần ẩn khung chụp trong UI).
- [ ] Người có chuyên môn soát bản dịch, ưu tiên các tác dụng phụ có dấu ⚠.
- [ ] Sửa số liệu trong report theo mục 1.

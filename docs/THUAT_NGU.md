# Thuật ngữ và tên viết tắt

Trong code, tên hàm và tên biến được viết đầy đủ. Một số tên vẫn giữ dạng viết tắt hoặc thuật ngữ chuyên ngành, vì đó là tên chuẩn của dữ liệu, của phương pháp, hoặc là tên cột trong file. File này giải thích các tên đó.

## 1. Dữ liệu nguồn

| Tên | Nghĩa |
|---|---|
| **Decagon** | Bộ dữ liệu tương tác thuốc của Zitnik và cộng sự (2018), phát hành qua BioSNAP (Stanford). Dự án dùng 4 file của bộ này. |
| `bio-decagon-combo` | Cặp thuốc và tác dụng phụ đã được báo cáo khi dùng chung. Trong code: `drug_pair_side_effects`. |
| `bio-decagon-mono` | Tác dụng phụ khi dùng riêng một thuốc. Trong code: `single_drug_side_effects`. |
| `bio-decagon-targets` | Thuốc và protein đích (protein mà thuốc tác động). Trong code: `target_proteins_by_drug`. |
| `bio-decagon-ppi` | Tương tác protein–protein (PPI = protein–protein interaction). Trong code: `protein_interactions`. |
| **STITCH ID** | Mã hóa chất trong cơ sở dữ liệu STITCH, ví dụ `CID000006691` (Warfarin). Đây là khóa duy nhất của một thuốc trong toàn hệ thống (`stitch_id`). |
| **CID** | PubChem Compound ID: phần số trong STITCH ID. |
| **Entrez Gene ID** | Mã gen của NCBI, ví dụ `5743`. Decagon dùng mã này cho protein. |
| **Gene symbol / HGNC** | Ký hiệu gen dễ đọc, ví dụ `PTGS2`, theo chuẩn đặt tên HGNC. |
| **FAERS** | Hệ thống báo cáo biến cố bất lợi của FDA (Mỹ), nguồn gốc của dữ liệu cặp thuốc. Vì là dữ liệu tự báo cáo nên có thiên lệch báo cáo (reporting bias). |
| **PubChem, SIDER, NCBI** | Các nguồn tên thuốc và tên gen dùng trong `resource_builders/`. |

## 2. Khái niệm của hệ thống

| Tên | Nghĩa |
|---|---|
| **Graph-RAG** | Retrieval-Augmented Generation trên đồ thị: lấy bằng chứng từ đồ thị trước, rồi đưa cho LLM viết giải thích. Tên package `graphrag/` lấy từ đây. |
| **LLM** | Large Language Model, mô hình ngôn ngữ lớn. Giữ nguyên tên `llm` trong code (thư mục `graphrag/llm/`, biến `llm_api_key`…). |
| **Regimen (phác đồ)** | Danh sách các thuốc người dùng đang dùng, tối thiểu 3 thuốc. |
| **Tier 1 – Đã ghi nhận** | Tác dụng phụ đã được báo cáo cho cặp thuốc. |
| **Tier 2 – Khuếch đại** | Tác dụng phụ Tier 1 của một cặp thuốc mà cặp đó có protein đích chung với một thuốc thứ ba. |
| **Tier 3 – Giả thuyết** | Cặp thuốc chưa có dữ liệu nhưng nối được với nhau qua protein (cầu protein). Chỉ là giả thuyết. |
| **Evidence (bằng chứng)** | Một đường đi trong đồ thị ủng hộ một nhận định, được đánh số `E1`, `E2`, … khi gửi cho LLM. |
| **Protein bridge (cầu protein)** | Đường từ đích của thuốc A tới đích của thuốc B: một bước (`one hop`, hai đích tương tác trực tiếp) hoặc hai bước (`two hop`, qua một protein trung gian `middle_protein`). |
| **Hub protein** | Protein có quá nhiều liên kết (nhiều hơn `HUB_DEGREE_THRESHOLD`). Đường đi qua hub gần như không mang thông tin nên bị loại (`pruned_hubs`). |
| **Degree (bậc)** | Số protein tương tác với một protein. Dùng trong `hub_degree_threshold`, `hub_degree_percentile`. |
| **Convergence protein (protein hội tụ)** | Protein nằm cách đích của ít nhất 3 thuốc một bước tương tác. Bật bằng `COUNT_CONVERGENCE_PROTEINS_AS_AMPLIFICATION=1`. |
| **Effect rarity (độ hiếm) / IDF** | `log(tổng số cặp ÷ số cặp có tác dụng phụ đó)`, tương tự IDF (inverse document frequency) trong tìm kiếm văn bản. Tác dụng phụ càng hiếm thì điểm càng cao. Nhãn "bỏ IDF" trong kết quả A5 nghĩa là trọng số này bằng 0. |
| **Serious (nghiêm trọng)** | Tác dụng phụ thuộc danh sách IME của EMA (xem mục 3). |
| **Critical (đe dọa tính mạng)** | Tác dụng phụ nghiêm trọng có từ khóa trong `resources/serious_keywords.csv` (chảy máu, loạn nhịp, suy tạng…). Chỉ dùng để xếp hạng. |
| **Template explanation** | Giải thích viết sẵn bằng tiếng Việt, dùng khi không gọi được LLM (`llm_status = "fallback"`). |
| **Cache** | File SQLite `.cache/decagon.sqlite3`, bản sao của dữ liệu tác dụng phụ để tra cứu nhanh. Tự tạo lại khi file nguồn hoặc `CACHE_SCHEMA_VERSION` thay đổi. |

## 3. Thuật ngữ y khoa

| Tên | Nghĩa |
|---|---|
| **MedDRA** | Từ điển thuật ngữ y khoa chuẩn dùng trong cảnh giác dược. |
| **PT (Preferred Term)** | Thuật ngữ ưu tiên trong MedDRA. Cột `meddra_pt` trong `resources/serious_effects.csv`. |
| **SOC (System Organ Class)** | Nhóm cơ quan trong MedDRA, ví dụ "Cardiac disorders". Cột `soc`. Trong code: `organ_class`. |
| **EMA** | Cơ quan Dược phẩm châu Âu. |
| **IME (Important Medical Event)** | Danh sách thuật ngữ MedDRA mà EMA coi là biến cố y khoa quan trọng. Dùng làm chuẩn cho "nghiêm trọng". |
| **NOS** | "Not otherwise specified", hậu tố trong tên tác dụng phụ Decagon. |
| **INN** | Tên chung quốc tế của hoạt chất (generic name). |

## 4. Tên cột và khóa dữ liệu giữ dạng ngắn

Các tên này nằm trong file CSV hoặc JSON kết quả nên được giữ nguyên để không phải sửa dữ liệu và báo cáo.

| Tên | Nghĩa |
|---|---|
| hậu tố `_en` | Tiếng Anh, ví dụ `name_en`, `side_effect_en`, `display_en` (tên tiếng Anh đã sửa lỗi chính tả để hiển thị). |
| hậu tố `_vi` | Tiếng Việt, ví dụ `name_vi`, `side_effect_vi`, `aliases_vi` (biệt dược bán tại Việt Nam), `explanation_vi`. |
| `vocabulary` = `combo` / `mono` | Trong `serious_effects.csv`: tên lấy từ dữ liệu cặp thuốc / dữ liệu thuốc dùng riêng. |
| `individual_side_effects` | Trong thẻ thuốc: tác dụng phụ khi dùng riêng thuốc đó. |
| `pruned_hubs` | Các protein hub đã bị loại khỏi đường bằng chứng. |
| `graph_norm` (cột A3) | Phương pháp "graph normalized": tần suất chuẩn hóa. |
| TDP | "Tác dụng phụ", dùng trong nhãn giao diện và báo cáo. |

## 5. Đánh giá (thư mục `evaluation/`)

| Tên | Nghĩa |
|---|---|
| **E1–E6** | Thí nghiệm về hệ thống: nhận tên thuốc (E1), trích thuốc từ câu chat (E2), LLM bám bằng chứng (E3), nhãn Tier 3 (E4), ngưỡng hub (E5), tốc độ (E6). |
| **A1–A8** | Thí nghiệm về độ chính xác: trung thực với dữ liệu (A1), dự đoán cặp (A2), dự đoán từng tác dụng phụ (A3), validator (A4), benchmark lâm sàng (A5), phân loại nghiêm trọng (A6), thêm thuốc nhiễu (A7), phác đồ 4–8 thuốc (A8). |
| **C1–C3** | Kiểm tra độ phủ toàn bộ thuốc: cache (C1), tên thuốc (C2), chạy hết pipeline (C3). |
| **AUROC** | Diện tích dưới đường ROC: xác suất một mẫu dương được chấm điểm cao hơn một mẫu âm. 0.5 là ngẫu nhiên, 1 là hoàn hảo. |
| **AUPRC / Average precision** | Diện tích dưới đường precision–recall. |
| **AP@K** | Average precision chỉ tính trên K kết quả đầu (Decagon dùng K = 50). |
| **FPR / TPR** | Tỷ lệ dương tính giả / dương tính thật, hai trục của đường ROC. |
| **Hit@K** | Tỷ lệ ca có kết quả đúng nằm trong K vị trí đầu. |
| **Precision@K, Recall@K** | Precision và recall chỉ tính trên K kết quả đầu. |
| **MRR** | Mean Reciprocal Rank: trung bình của 1/hạng của kết quả đúng (0 nếu không tìm thấy). |
| **Lift** | Tỷ lệ dương khi có đặc trưng chia cho tỷ lệ dương chung. Lift = 1 nghĩa là đặc trưng không giúp gì. |
| **Baseline** | Phương pháp đơn giản để so sánh (độ phổ biến, ngẫu nhiên). |
| **Holdout** | Ẩn một phần dữ liệu đã biết để kiểm tra dự đoán. |
| **Seed** | Hạt giống số ngẫu nhiên; cùng seed thì cùng kết quả. |

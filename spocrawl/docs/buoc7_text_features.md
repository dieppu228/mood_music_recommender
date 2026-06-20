# Bước 7 — Text features: Vấn đề & Hướng xử lý

Tài liệu mô tả các vấn đề của bước tạo `combined_text` / `tokens` trong `process_and_visualize.ipynb`, dựa trên phân tích dataset thực tế và tham chiếu research — không khẳng định vượt quá bằng chứng.

---

## Bối cảnh

Spotify Dev Mode **không cung cấp** lyrics, audio features (valence, energy…), genres, popularity. Bước 7 thay thế bằng metadata text:

```
combined_text = track_name + artist + tags + search_query + source_name
tokens        = tách từ + bỏ stopwords (sklearn)
token_count   = số token
```

Đây là **workaround hợp lý** trong constraint đồ án, không phải setup chuẩn trong MER research (thường dùng audio hoặc lyrics).

---

## Vấn đề đã xác nhận trên data (8,678 track)

### 1. Bug `NaN` → token `"nan"`

| | |
|---|---|
| **Hiện tượng** | Profile track thiếu `search_query`/`tags` → chuỗi `"nan"` xuất hiện trong `combined_text` và `tokens` |
| **Quy mô** | **1,309 dòng (12.8%)** — gần hết là `data_origin = profile` |
| **Nguyên nhân** | `bool(float('nan')) == True` trong Python; `str(NaN)` → `"nan"` vẫn đi qua filter |
| **Ví dụ** | `santa monica bubble tea and cigarettes nan nan liked songs` |
| **Mức tin cậy** | **Chắc chắn** — bug xử lý, nên sửa |

### 2. Nhiễu từ `source_name` generic

| | |
|---|---|
| **Hiện tượng** | Token `liked` xuất hiện trên phần lớn track profile |
| **Quy mô** | **~48.9%** track profile |
| **Nguyên nhân** | `source_name = "Liked Songs"` được ghép vào text nhưng không mang thông tin mood |
| **Mức tin cậy** | **Chắc chắn** — nhiễu có thể đo được |

### 3. Lặp từ từ query/tags

| | |
|---|---|
| **Hiện tượng** | `search_query` và `tags` thường trùng nội dung → cùng cụm từ lặp 2–4 lần trong `combined_text` |
| **Ví dụ** | `happy vibes simon more happy happy vibes happy vibes happy vibes` |
| **Hệ quả** | `token_count` cao nhưng **thông tin mới ít**; TF-IDF có thể bias về từ query |
| **Mức tin cậy** | **Quan sát được** trên data |

### 4. Profile vs search không đồng nhất

| Nguồn | `token_count` TB | Có `search_query` | Độ tin cậy mood label |
|-------|------------------|-------------------|------------------------|
| search | ~14.5 | Có | Cao (gắn query crawl) |
| profile | ~7.9 | Không | Thấp hơn (~1,134 bài gán mặc định `calm`) |

Cùng một cột `combined_text` nhưng **hai phân phối text khác nhau** → model có thể học pattern nguồn thay vì mood.

---

## Vấn đề cần diễn đạt cẩn thận (không overclaim)

### 5. `search_query` gắn chặt với `mood_label`

**Số liệu trên track search:**

- Query có nghĩa mood rõ (một mood) → khớp `mood_label`: **~94.9%**
- Chỉ title + artist có gợi ý mood → khớp label: **~78.8%** (khi có hint)

**Giải thích:**

- Với track **search**, mood được gán **từ query lúc crawl** → correlation cao là **đúng thiết kế**, không phải bug ngẫu nhiên.
- Theo [framework label leakage](https://pmc.ncbi.nlm.nih.gov/articles/PMC10746313/) (association-based): nếu lúc **predict** không có `search_query` mà feature vẫn chứa query → metric train có thể **lạc quan**.
- **Chưa chứng minh** model sẽ fail hoàn toàn: title + artist vẫn có tín hiệu mood (~79% trong subset có hint).

**Phụ thuộc use case team model:**

| Kịch bản inference | `search_query` trong feature có hợp lệ? |
|--------------------|----------------------------------------|
| Phân loại mood từ **metadata bài hát** (tên, artist) | **Không** — query là artifact crawl |
| Chatbot: user nhập câu → map mood → gợi ý bài | User text có ở runtime, nhưng **không phải** `search_query` trên từng track |

→ Nên **tách cột** để team thử nghiệm, không khẳng định trước mức ảnh hưởng.

### 6. So với research

| Nguồn | Điểm liên quan |
|-------|----------------|
| [MTG-Jamendo](https://mtg.github.io/mtg-jamendo-dataset/) | Mood tag + **audio** — chuẩn auto-tagging |
| [CSE6242 Spotify](https://github.com/jroblar/cse6242-project) | Label playlist + **audio features** |
| Multimodal mood (lyrics TF-IDF/BERT) | Text-only ~70–77% khi có **lyrics**, không phải chỉ title |

**Kết luận research:** Title + artist là **tín hiệu yếu hơn lyrics/audio**; chấp nhận được cho đồ án nếu **ghi rõ hạn chế** trong báo cáo.

---

## Hướng xử lý gợi ý

### Mức 1 — Nên làm (bằng chứng rõ, rủi ro thấp)

#### a) Hàm `safe_str` — fix bug NaN

```python
def safe_str(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s
```

#### b) Bỏ nguồn generic khỏi text

Không đưa `source_name` vào feature nếu giá trị là `"Liked Songs"`, `"liked"`, hoặc tương đương.

#### c) Dedup token

Sau khi ghép text, loại token trùng (giữ thứ tự) trước khi lưu `tokens`.

#### d) Blocklist token nhiễu

```python
GENERIC_TOKENS = {"liked", "songs", "song", "playlist", "nan", "unknown"}
```

Loại khỏi `tokens` ngoài stopwords sklearn.

---

### Mức 2 — Khuyến nghị cho team model (tách cột)

Thay một `combined_text` duy nhất bằng:

| Cột | Nội dung | Mục đích |
|-----|----------|----------|
| `text_content` | `track_name_clean` + `artists_clean` + `album` (nếu có) | **Feature chính** — an toàn khi predict từ metadata bài |
| `text_crawl_context` | `search_query` + `tags_normalized` (chỉ search, dedup) | EDA / baseline thử nghiệm — **không khuyến nghị làm feature duy nhất cho eval cuối** nếu inference không có query |
| `tokens` | Tokenize từ `text_content` | Input cho TF-IDF / bag-of-words |
| `tokens_context` | (tuỳ chọn) Tokenize từ `text_crawl_context` | Ablation: so sánh có/không query |

#### `label_confidence` — hỗ trợ train có chọn lọc

| Giá trị | Điều kiện |
|---------|-----------|
| `high` | `data_origin == search` |
| `medium` | `data_origin == profile` và `mood_inferred == True` |
| `low` | `data_origin == profile` và mood gán mặc định `calm` |

Team có thể train chỉ trên `high` + `medium` để giảm noise từ label yếu.

---

### Mức 3 — Không bắt buộc trước baseline

| Ý tưởng | Lý do chưa ưu tiên |
|---------|---------------------|
| Chuẩn hóa tên bài (`feat.`, `Remaster`) | Chưa đo impact trên data này |
| Embedding / BERT | Phức tạp; chưa cần trước TF-IDF baseline |
| API lyrics bên thứ ba | Ngoài scope Spotify crawl |

---

## Ba phương án quyết định

| Phương án | Nội dung | Khi chọn |
|-----------|----------|----------|
| **A** | Giữ `combined_text` + chỉ fix bug (mức 1) | Ship nhanh; chấp nhận metric có thể cao trên search |
| **B** | Mức 1 + tách `text_content` / `text_crawl_context` + `label_confidence` | **Khuyến nghị** — team so sánh 2 feature set |
| **C** | Chỉ `text_content` cho model | Predict mood từ metadata bài; chatbot xử lý user text riêng |

---

## Việc team model nên làm sau khi có file processed

1. Chạy baseline trên `tokens` (từ `text_content`).
2. (Tuỳ chọn) Chạy thêm trên `tokens` cũ / `tokens_context` — so sánh metric.
3. Báo cáo nêu rõ: không có lyrics/audio; label search gắn query; subset `label_confidence`.
4. Cân nhắc class weight / oversample cho mood thiếu (`stressed`, `romantic`, `energetic`).

---

## Tóm tắt một dòng

**Bước 7 hiện tại đủ cho baseline đồ án**, nhưng có **bug `nan`**, **nhiễu `liked`**, và **rủi ro association** giữa `search_query` và label — nên **fix bug + tách `text_content` / `text_crawl_context`** để team chọn feature phù hợp inference, không overclaim độ chính xác production.

---

## Cập nhật — Notebook đã vá (Bước 7 mới)

`process_and_visualize.ipynb` đã implement **Phương án B/C**:

| Cột | Nội dung | Ghi chú |
|-----|----------|---------|
| `text_content` | `track_name_clean` + `artists_clean` + `album` (+ `lyrics` nếu có) | **Feature chính** — không chứa query/tags |
| `text_crawl_context` | `search_query` + `tags_normalized` (chỉ `data_origin==search`) | EDA / ablation — **không** train `target=mood` rồi báo accuracy |
| `tokens` | Tokenize từ `text_content` | TF-IDF input |
| `label_source` | `search_query` / `inferred` / `default_calm` | Lọc subset train |
| `combined_text` | Alias `text_content` (giữ tương thích) | |

**Cần chạy lại notebook (Bước 7–8)** để ghi các cột mới vào `spotify_hybrid_processed.csv`. Sau re-run, `combined_text` = `text_content` (alias) — **CSV leaky cũ không còn**, số before chỉ lưu archived.

---

## Kết luận nguồn lyrics (đã thử, có bằng chứng)

### musiXmatch / MSD (~2011)

Probe `scripts/probe_musixmatch.py` join `title+artist` với `mxm_779k_matches.txt`:

| Nhóm | Match |
|------|-------|
| Toàn bộ 10,197 bài | **258 (2.5%)** |
| Tây (~87%) | **253 (2.9%)** |
| **2020+** (~78% dataset) | **56 (0.7%)** |
| pre-2010 | 134/437 (30.7%) |

→ Corpus quá cũ so với playlist hiện đại. **Không phải xương sống** — chỉ ~250 bài BoW miễn phí nếu cần.

### Genius API

| Tầng | Kết quả pilot |
|------|----------------|
| Token + `api.genius.com/search` | **200 OK**, hit đúng bài (Pumped Up Kicks, James Brown, Billie Eilish) |
| Scrape trang `genius.com/...-lyrics` | **`scrape_403`** (Cloudflare) — 100% trên probe 3 bài và pilot 200 bài |

→ API chính thức **không trả full lyrics**. Scrape tự động **không khả thi** ở quy mô đồ án. **Không dùng `lyrics_sample_summary.json` match_pct=0%** làm coverage — đó là số liệu probe hỏng (search OK, scrape fail).

**Quyết định:** Dừng pipeline lyrics. Classifier mood = **metadata-only** (`text_content`).

---

## Leakage: trước / sau tách feature (`idk.py`)

Chạy: `python idk.py data/processed/spotify_hybrid_processed.csv`

**Quan trọng:** Sau re-run Bước 7–8, CSV **không còn** bản leaky. `idk.py` in **ARCHIVED** (probe đầu trên CSV cũ) + **PROBE HIỆN TẠI**. **Đừng** so `[B] cũ` vs `[B'] mới trên cùng file post-patch — `Δ=0` chỉ vì `combined_text` đã = `text_content`.

### TRƯỚC vá (archived — CSV cũ đã ghi đè, không tái tạo)

| Probe | Giá trị | Ý nghĩa |
|-------|---------|---------|
| [A] recovery combined_text | **0.811** | ~81% label khôi phục bằng keyword rule |
| [B] model tokens (leaky) | **0.993** | Vòng lặp gần hoàn hảo |
| [E] query-only | **1.000** | Query alone dự đoán mood |
| [E] tokens GROUPED query | **0.906** | Memorize query |

### SAU vá (probe trên CSV hiện tại — `text_content`)

| Probe | Giá trị | Ý nghĩa |
|-------|---------|---------|
| **[A] recovery text_content** | **0.267** | Keyword gán nhãn **đã rời** feature (< baseline 0.301) — **xác nhận cắt leakage** |
| [B] model tokens | **0.707** | Metadata-only, không vòng lặp query |
| [C] ablation keyword | **0.649** | Bỏ keyword chỉ rớt nhẹ → signal chủ yếu metadata thật |
| [D] chỉ track+artist | **0.671** | ≈ [B] — album ít ảnh hưởng |
| [G] artist-only | **0.531** | Cận dưới phòng thủ |
| [E] query-only | **1.000** | Label vẫn từ query — **CẤM** đưa vào feature |
| [E'] crawl_context | **1.000** | Cùng lý do — chỉ demo leakage |

**Câu chuyện báo cáo:** recovery 0.811 → **0.267**; accuracy leaky **0.993** → **0.707**; query/context vẫn 1.0 nếu dùng nhầm cột.

**label_source:** 8891 `search_query` / 1134 `default_calm` / 172 `inferred` — train nên **bỏ 1134 default_calm**.

**Không overclaim:** 0.707 > [G] 0.531 — có thể còn **rò nhẹ qua track_name** (từng trong logic gán nhãn profile). Báo cáo: *"metadata F1 ≈ 0.71; artist-only ≈ 0.53 là cận dưới."*

---

## Handoff cho model team

### File & cột

**Input:** `data/processed/spotify_hybrid_processed.csv` (sau re-run Bước 7–8)

| Cột | Train? | Inference? |
|-----|--------|------------|
| `text_content` / `tokens` | **Có** — feature chính | **Có** — metadata bài |
| `text_crawl_context` | Chỉ ablation / demo leakage | **Không** — user runtime không có crawl query trên từng bài |
| `mood_label` | Label (weak supervision) | Target để gán nhãn bài (nếu cần) |
| `label_source` | Lọc: ưu tiên `search_query`, cân nhắc bỏ `default_calm` | EDA |
| `search_query`, `tags` | **Không** vào feature mood classifier | — |

### Classifier mood (baseline)

```
Feature:  TF-IDF(tokens) từ text_content
Label:    mood_label (query-derived, có nhiễu)
Eval:     Stratified CV accuracy / macro-F1
Kỳ vọng:  ~0.67–0.71 (text_content); ~0.53 artist-only; bỏ default_calm khi train
Không:    train trên text_crawl_context rồi báo "accuracy mood"
```

Chạy leakage check trước khi báo cáo: `python idk.py data/processed/spotify_hybrid_processed.csv`

### Recsys / chatbot (trọng tâm đánh giá)

- User text → classify **user mood** (input thật lúc inference — không leakage)
- Lọc / rank bài theo mood + **semantic similarity** trên `text_content`
- Metric: **Precision@K**, khảo sát người dùng — **không** phụ thuộc accuracy classifier bài

### Limitation (copy vào báo cáo)

> … đã phát hiện leakage query→feature (recovery 0.81→0.27; accuracy leaky ~0.99→~0.71 sau tách cột; xem ARCHIVED trong idk.py). …

### Việc cần làm ngay

1. Baseline TF-IDF trên `tokens` (từ `text_content`)
2. (Tuỳ chọn) Ablation `tokens_context` — demo leakage 1.0, không báo cáo chính
3. Class weight / oversample `stressed`, `romantic`, `energetic`
4. Lọc train: bỏ `label_source == default_calm` (1134 bài)

---

# Cập nhật — Sau điều tra leakage & nguồn lyrics

Phần này ghi lại kết quả điều tra thực nghiệm (không suy đoán) và trạng thái notebook sau khi vá. Mọi số đều từ `idk.py` chạy trên dataset thật (10,197 track) hoặc từ probe nguồn dữ liệu.

## 1. Trạng thái notebook (đã vá)

**Bước 7 đã sửa:**
- `safe_str()` — chặn bug `NaN` → token `"nan"` (xác nhận: `bool(float('nan'))==True`, `str(nan)=='nan'`).
- `GENERIC_TOKENS = {liked, songs, song, playlist, nan, unknown}` — loại nhiễu generic.
- `dedupe_tokens()` — bỏ token lặp do ghép query/tags.
- **Tách cột**: `text_content` (track + artist + album + lyrics nếu có) vs `text_crawl_context` (chỉ search: query + tags, deduped).
- `combined_text` giờ = `text_content` → không còn chứa query/tags (hết rò).
- `tokens` build từ `text_content`, đã dedupe + bỏ generic/stopword.
- `assign_label_source()` đặt **trước** khi save → `label_source` có trong CSV.
- Hook merge `lyrics`/`lyrics_status` nếu chạy `scripts/fetch_lyrics.py`.

**Bước 5 GIỮ NGUYÊN** (có chủ đích): nhãn `mood_label` vẫn suy từ query/tags/track_name. Đây là **weak supervision chấp nhận được** — không vòng lặp *nếu* feature là `text_content` (title/artist/album) tách khỏi trường sinh nhãn.

> ⚠️ **Re-run Bước 7–8** nếu CSV chưa có `text_content` / `text_crawl_context` / `label_source`. Sau re-run, `combined_text` = `text_content` — bản leaky cũ **không tái tạo** được; số TRƯỚC chỉ lưu ARCHIVED trong `idk.py`.

## 2. Kết luận nguồn lyrics (đã loại — có bằng chứng)

| Nguồn | Kết quả thật | Kết luận |
|-------|--------------|----------|
| musiXmatch (`mxm_779k_matches.txt`) | Match 258/10,197 (**2.5%**); 2020+ chỉ 0.7%; pre-2010 30.7% | Corpus 2011 → quá cũ so với playlist. Chỉ là phần thưởng cho nhạc cũ, **không phải nguồn chính** |
| Genius — search API | OK (Pumped Up Kicks, James Brown, Billie Eilish đều có hit) | Token & search **không** phải vấn đề |
| Genius — scrape trang lyrics | **403 Cloudflare 100%** trên bài Tây | Không lấy được full lyrics tự động ở quy mô. API chính thức không trả full lyrics |

> **KHÔNG dùng con số 0% trong `lyrics_sample_summary.json` làm "coverage".** Đó là probe hỏng ở tầng scrape (Cloudflare chặn), không phản ánh việc Genius có bài hay không.

**Hệ quả:** lyrics không khả thi ở quy mô đồ án → classifier mood huấn luyện trên **metadata**; gợi ý dựa trên **semantic similarity metadata + mood filter**.

## 3. Đo leakage trước/sau khi tách cột (`idk.py`) — vòng đã khép

Baseline (lớp đa số `calm`) = **0.301**.

| Probe | TRƯỚC (archived — CSV leaky đã ghi đè) | SAU (`text_content`, CSV hiện tại) |
|-------|----------------------------------------|-------------------------------------|
| [A] keyword recovery | **0.811** | **0.267** (< baseline → keyword đã rời feature) |
| [B] model accuracy | **0.993** | **0.707** |
| [C] ablation keyword | — | 0.649 |
| [D] track + artist | — | 0.671 |
| [G] artist-only | — | **0.531** |
| [E] query-only (nhóm search) | **1.000** | (query đã rời feature chính) |
| [E'] train trên `text_crawl_context` | — | **1.000** (CẤM) |

**Đọc:**
- Recovery **0.811 → 0.267** — xác nhận cắt leakage đẹp nhất (dùng dòng này, không dùng Δ=0 trên CSV post-patch).
- Accuracy leaky **0.993 → 0.707** (Δ≈0.29): phần lớn 0.993 là vòng lặp query/tags trong feature.
- [B] sau (0.707) ≈ [D] (0.671): `text_content` không còn rò query.
- [C] 0.707 → 0.649: bỏ keyword chỉ rớt nhẹ → signal chủ yếu metadata thật.
- [E]/[E'] = 1.0: nhãn vẫn từ query — **CẤM** đưa crawl_context vào feature mood.

> **Caveat:** [D]=0.671 > [G]=0.531 vì `track_name` từng nằm trong hàm gán nhãn → còn rò nhẹ. **[G] artist-only (0.531) là cận dưới phòng thủ** (artist không tham gia gán nhãn). Báo cáo: *"metadata ~0.71; artist-only 0.53 là cận dưới"* — không tuyên bố "title+artist dự đoán mood 0.70".

**label_source:** 8891 `search_query` / 1134 `default_calm` / 172 `inferred` — train nên **bỏ `default_calm`**.

## 4. Handoff cho model team

### 4.1. Cột nào dùng vào đâu

| Cột | Dùng làm feature train mood? | Ghi chú |
|-----|------------------------------|---------|
| `text_content` | ✅ **Feature chính** | track + artist + album (+ lyrics nếu có). An toàn khi inference từ metadata bài |
| `tokens` | ✅ Input TF-IDF / BoW | Sinh từ `text_content`, đã dedupe + bỏ generic/stopword |
| `text_crawl_context` | ❌ **CẤM làm feature mood** | query + tags; chỉ EDA / ablation leakage |
| `search_query`, `tags`, `tags_normalized` | ❌ | Trường sinh nhãn → rò |
| `mood_label` | 🎯 Target | Nhãn yếu (query-derived). 6 lớp |
| `label_source` | Lọc nhãn | Bỏ `default_calm` (1134 bài) khi train |
| `data_origin` | Split / phân tích | `search` vs `profile` |
| `lyrics`, `lyrics_status` | ❌ (hiện không có) | Genius scrape 403; không phụ thuộc pipeline lyrics |

### 4.2. Kỳ vọng metric (đừng overclaim)

- Báo cáo trên `text_content`: accuracy tham khảo **~0.71** — dùng **macro-F1 + per-class + confusion matrix**.
- 6 lớp lệch (calm ~30%) → macro-F1 thấp hơn accuracy; `class_weight='balanced'` hoặc oversample.
- Mốc: baseline **0.301** · artist-only **0.531** · track+artist **0.671** · text_content **0.707**.
- **KHÔNG** báo `combined_text` leaky (0.993) hay query-only (1.0) như thành tích.

### 4.3. Đoạn Limitation (copy-paste vào báo cáo)

> Hệ thống không truy cập được lyrics ở quy mô lớn: Spotify Dev Mode không cung cấp lyrics/audio features; musiXmatch chỉ khớp ~2.5% dataset (corpus 2011, quá cũ so với playlist hiện đại); Genius có search API hoạt động nhưng trang lyrics bị Cloudflare chặn scrape (403) và API chính thức không trả full lyrics. Do đó mô hình phân loại cảm xúc được huấn luyện trên metadata bài hát (tên, nghệ sĩ, album) thay vì lyrics. Nhãn mood được suy từ ngữ cảnh crawl (search query/tags) nên là nhãn yếu (weak supervision); chúng tôi đã phát hiện và định lượng label leakage (mô hình trên text gộp đạt 0.993; recovery keyword 0.811; query-only đạt 1.0), sau đó tách feature an toàn (`text_content`) khỏi trường sinh nhãn — recovery giảm còn 0.267, accuracy về ~0.71 (artist-only 0.53 là cận dưới). Hệ gợi ý dựa trên semantic similarity của metadata kết hợp lọc theo mood, đánh giá bằng Precision@K và khảo sát người dùng, không phụ thuộc độ chính xác của bộ phân loại.

### 4.4. Việc cần làm

- [x] Vá notebook Bước 7 + `idk.py` (archived before / probe hiện tại)
- [ ] Re-run Bước 7–8 nếu CSV chưa có cột mới
- [ ] `python idk.py data/processed/spotify_hybrid_processed.csv` — đính số vào báo cáo
- [ ] Train classifier trên `tokens`; báo macro-F1 + per-class + confusion matrix
- [ ] Loại `label_source == default_calm`; thử `class_weight='balanced'`
- [ ] Recsys: metadata embedding + mood filter; Precision@K + khảo sát
- [ ] **KHÔNG** đưa `text_crawl_context` / `search_query` / `tags` vào feature mood


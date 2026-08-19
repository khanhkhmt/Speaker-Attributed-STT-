# Implementation status

| Thuộc tính | Giá trị |
|---|---|
| Cập nhật | 19/08/2026 |
| Spec tham chiếu | [`production-technical-spec.md`](production-technical-spec.md) v1.0 |
| Milestone hiện tại | M0 xong · M1 **xong** · **M2 ~85%** · **M3 ~60%** · **M4 ~45%** · **M5 ~30%** — phần còn lại là gate cần model/GPU/corpus thật |
| Engine mặc định | `fake` (M0). Đặt `SASTT_ENGINE=real` để dùng adapter model thật |

Tài liệu này ghi **tình trạng thực tế** của repo. Mọi con số đều lấy từ lần chạy
thật, không ước lượng.

## 1. Milestone (spec 18)

| Milestone | Phạm vi | Trạng thái |
|---|---|---|
| M0 — Foundation & contracts | package `sastt`, config validation, domain models, ports, JSON Schema v2, fake adapters, state machine, revision/idempotency, CI | **xong** |
| M1 — Offline 2-speaker path | decode/resample, adapter pyannote + faster-whisper + MossFormer2 + CAM++, pin manifest, smoke test | **xong về luồng chức năng** — DoD, worker queue và chốt transcript thực đã kiểm chứng; benchmark/soak/accuracy evidence vẫn là nợ production |
| M2 — Linking & Voice ID | pgvector registry, enrollment quality, deletion/audit | **~85%** — API local tạo/enroll/xem/xóa đã chạy; persistent pgvector cần được chọn khi deploy |
| M3 — Near-realtime | queue/worker, backpressure, latency instrumentation | **~60%** — có WebSocket/revision/replay, bounded RAM ring + disk spool final pass, backpressure phía client và Prometheus stage/RTF/GPU-optional metrics. Tuy nhiên chưa chứng minh được stream hết audio và gán speaker ổn định trên audio thật; không đạt beta/production. Chưa có OTel, autoscale hay SLO đo trên baseline. |
| M4 — Beta/advanced overlap | SepFormer 3mix, concurrent counter, GSS, WeSep | **~45%** — router + feature gate + adapter SepFormer 3mix 8 kHz đã có, và bộ đếm người đồng thời nay chạy bằng bằng chứng diarization nên các dòng K≥3 của router lần đầu tới được. Chưa pre-stage/benchmark checkpoint 3 nguồn, chưa có GSS/WeSep. **Đang chờ quyết định hướng đi — xem mục 6.** |
| M5 — Production hardening | benchmark corpus, calibrator, load/soak, SBOM, capacity | **~30%** — calibrator release JSON, CLI benchmark/capacity/SBOM và guardrail test đã có; corpus, calibration release ký duyệt, soak/load và capacity evidence chưa có |

## 2. Gate chất lượng — lần chạy gần nhất

| Gate | Lệnh | Kết quả |
|---|---|---|
| Lint | `ruff check src tests deploy` | pass |
| Format | `ruff format --check src tests deploy` | pass |
| Type | `mypy` (strict) | pass, 62 source file |
| Test thường | `pytest` | **297 passed**, 59 deselected (lần kiểm tra 19/08/2026; không gộp model/load/db) |
| Test model | `pytest -m model` | **34 passed, 0 skipped** |
| Test hạ tầng | `pytest -m db` | **21 passed** (Postgres + Redis thật) |
| Coverage | `--cov=sastt.domain --cov=sastt.application` | 88% (spec 16.3 yêu cầu ≥85%) |

Test model chạy trên weights thật + audio thật (VoxConverse dev, public,
multi-speaker) trên 2× Tesla T4. Không còn test nào skip: 2 checkpoint pyannote
đã accept điều kiện và pin xong (mục 4).

## 3. Model weights (spec 11.2, 20)

Pin bằng `deploy/prestage_models.py`; runtime chỉ mount `/models` read-only.

| Backend | Repo | Revision | Dung lượng | Trạng thái |
|---|---|---|---:|---|
| `faster_whisper` (turbo) | `deepdml/faster-whisper-large-v3-turbo-ct2` | `4df90f75…` | 1.5 GiB | pinned + verified |
| `faster_whisper_large_v3` | `Systran/faster-whisper-large-v3` | `edaa852e…` | 2.9 GiB | pinned + verified |
| `mossformer2_ss_16k` | `alibabasglab/MossFormer2_SS_16K` | `407cb030…` | 639 MiB | pinned + verified |
| `3d_speaker_campplus` | `iic/speech_campplus_sv_zh_en_16k-common_advanced` (ModelScope) | `v1.0.0` | 27 MiB | pinned + verified |
| `pyannote-community-1` | `pyannote/speaker-diarization-community-1` | `3533c8cf…` | 33 MiB | pinned + verified |
| `pyannote_segmentation_3.0` | `pyannote/segmentation-3.0` | `e66f3d3b…` | 5.8 MiB | pinned + verified |
| `sepformer_libri3mix` | `speechbrain/sepformer-libri3mix` | — | — | adapter M4 đã có, nhưng weight chưa pre-stage/verify; chỉ chạy khi bật `three_source_beta` |
| `multidecoder_dprnn` | `JunzheJosephZhu/MultiDecoderDPRNN` | — | — | `deny` production (spec 20) |
| `gpu_gss`, `wesep` | — | — | — | không có weights / phase 2 |

Cả 6 backend của M1 đã pin và verify:
`python3 deploy/prestage_models.py --verify` → 6/6 `[ok]`.

## 4. Blocker cũ — đã gỡ

`pyannote/speaker-diarization-community-1` và `pyannote/segmentation-3.0` là
gated model. Trước đây tải về trả 403 vì tài khoản chưa accept điều kiện. Sau
khi accept, cả hai tải và pin bình thường; `SASTT_ENGINE=real` chạy được toàn bộ
pipeline offline trên audio thật.

Gỡ blocker này làm lộ 3 lỗi thật mà đường fake không thể phát hiện — đã sửa:

| Lỗi | Triệu chứng | Nguyên nhân |
|---|---|---|
| OSD chuyển đổi powerset | `mat1 and mat2 shapes cannot be multiplied (2356x3 and 7x3)` | `Inference` tự convert powerset→multilabel (hard) trước khi adapter tự convert soft; thiếu `skip_conversion=True` |
| OSD gộp chunk | timestamp phồng lên ~20× (overlap 2.5–5 s bị báo 146–292 s) | pyannote chỉ overlap-add *sau* convert; với `skip_conversion` phải tự `Inference.aggregate` theo `model.receptive_field`, nếu không output per-chunk bị đọc phẳng thành timeline 1 s/frame |
| Merge nhầm 2 người | hai nguồn của cùng vùng overlap bị gán chung một `session_speaker_id` | source bị reject tạo temporary ID nhưng không có cannot-link với source kia cùng crop, nên reconciliation gộp lại (spec 5.6, 5.8.7) |

### Cập nhật 19/08/2026 — chẩn đoán lại vùng overlap, và dựng thước đo

Đo trên một upload thật 20 phút tiếng Việt 3 người (`job_01M0C5CA41JJAFY2QSWKZFQ010`,
`SASTT_ENGINE=real`): 319 segment, 271 segment không overlap **đều có tên** trên
đúng 3 người, còn 48 segment overlap thì **41 ra `Unknown` (85%)**. Segment gán
được tên có trung vị 1080 ms, segment `Unknown` có trung vị 480 ms.

**Nguyên nhân trội không phải cosine nhiễu — phần lớn nguồn chưa từng được embed.**
`linking_minimum_speech_ms` mặc định rơi về `speaker_embedding.minimum_clean_speech_seconds`
= 1500 ms, trong khi trung vị một vùng overlap trên file đó là 660 ms. Adapter ném
`InsufficientSpeechForEmbeddingError`, `embed_buffer` trả `None`, và không có
vector nào để đưa vào Hungarian. Chẩn đoán cũ ở mục 6.1 ("dưới 500 ms embedding là
nhiễu") mô tả một hiện tượng có thật nhưng nằm *sau* chỗ pipeline dừng lại.

Đáng chú ý hơn: separation đã chạy trên cửa sổ `region ± audio.overlap_context_seconds`
(0.5 s mỗi phía), nhưng `separate_and_link` gọi `embed_buffer(source, owned)` — đúng
lõi vùng — nên phần đệm bị cắt trước khi vào CAM++. Audio cần thiết đã được tính
rồi và bị bỏ đi. Đo trên 15 vùng overlap đã tách của file đó:

| Cửa sổ dùng để embed | Vùng đạt mốc 1500 ms |
|---|---|
| `owned` — lõi vùng (hiện tại) | 3/15 = 20% |
| `padded` — cả cửa sổ đã tách | 10/15 = 67% |

**Một tương tác chưa ai ghi:** `short_source_policy: diarization_constrained` chỉ
kích hoạt khi `active_clusters == embedded`. Với vùng ngắn thì `embedded = 0`, nên
nhánh đó trơ đúng ở chỗ cần nó nhất. Nới cửa sổ embedding là điều kiện cần để nó
có tác dụng, không phải phương án thay thế.

#### Đã bổ sung

| Artefact | Vai trò |
|---|---|
| `source_linking.embedding_window: owned \| padded` | Cờ config, **mặc định `owned`** (không đổi hành vi). `padded` embed cả cửa sổ đã tách. ASR vẫn chỉ đọc `owned` — bất biến này có test — nhưng transcript cuối *không* bất biến: danh tính quyết định cách gom từ thành utterance, nên ranh giới segment dịch theo. Chưa nghiệm thu. |
| `deploy/overlap_eval.py` | Chấm điểm gán người nói trong vùng overlap so với nhãn tay. Luôn trả **bộ ba** đúng/nhầm/`Unknown`, ánh xạ session speaker ID sang nhãn bằng Hungarian một-một. Chế độ so sánh chỉ báo "cải thiện" khi đúng tăng **và** nhầm không tăng. |
| Chế độ gán nhãn trong `web/` | Bấm dòng overlap → nghe → phím `1..9` gán người, `0` = không nghe ra, tự nhảy sang dòng chưa gán. Xuất JSON đúng định dạng `overlap_eval.py` đọc. |
| `GET /v1/jobs/{id}/audio` | Trả input gốc của job để nghe lại từng segment. Giữ 12 job gần nhất, `DELETE` xóa kèm. |

#### Phép thử `padded` trên file 20 phút — kết quả KHÔNG ủng hộ việc bật

Chạy hai lượt trên cùng file, cùng ngưỡng `0.55/0.10`, chỉ khác `embedding_window`:

| | `owned` | `padded` |
|---|---:|---:|
| Nguồn overlap có cosine để so | 7 | **13** |
| Segment overlap có tên | 7 | 11 |
| Segment overlap `Unknown` | 42 (86%) | 38 (78%) |
| **Số session speaker riêng biệt** | 4 | **5** |

Cơ chế hoạt động đúng như dự đoán: gần gấp đôi số nguồn sinh được embedding. Nhưng
hai quan sát sau là lý do **không** được bật nó:

1. **Roster phình lên.** File có 3 người, `estimated_session_speakers = 3`, nhưng
   `padded` sinh ra `Speaker 5` cho 2 segment. Nhiều embedding hơn nghĩa là nhiều
   nguồn đủ điều kiện tạo temporary speaker hơn, và temporary nào không
   reconcile được thì đọng lại thành một người không có thật.
2. **Hai lượt gán tên khác nhau cho cùng một segment.** Ở 399411 ms và 400931 ms,
   `owned` nói `Speaker 1` (cos 0.455), `padded` nói `Speaker 5` (cos 0.471). Cả
   hai đều dưới ngưỡng 0.55. Ít nhất một trong hai phải sai, và không có nhãn thì
   không biết cái nào — đúng tình huống mục 6.3 mô tả.

Ngoài ra 6 nguồn mới có cosine nhưng chỉ 4 segment thêm tên: ngưỡng chấp nhận
(chưa hiệu chỉnh) nay là chỗ nghẽn tiếp theo, không còn là mốc embedding.

Kết luận: giữ `owned`. `padded` chỉ được xét lại sau khi có bộ nhãn và một
calibration release thật.

#### Baseline đầu tiên trên nhãn tay (19/08, n nhỏ)

8 segment overlap được gán nhãn tay trên `job_01M0CC9VVA5518QJ1ETJQX6Q4E` (1 nhãn
là "không nghe ra", còn 7 quyết định được):

| | đúng | nhầm | `Unknown` |
|---|---:|---:|---:|
| Tổng (n=7) | **0%** | **28.6%** | 71.4% |
| vùng < 1000 ms (n=5) | 0% | 0% | 100% |
| vùng ≥ 1000 ms (n=2) | 0% | 100% | 0% |

Hai segment duy nhất pipeline dám đặt tên đều **đảo hai người**: cùng vùng
107811 ms, source 0 được gán `Speaker 1` trong khi người nghe nói đó là
`Speaker 2`, và ngược lại. Hai dòng sai đến từ **một** quyết định hoán vị sai của
Hungarian, không phải hai lỗi độc lập.

Vùng dưới 1000 ms: 5/5 ra `Unknown`, đúng như chẩn đoán mốc embedding.

`padded` không đổi một dòng nào trong tập đã gán nhãn — 4 cái tên thêm của nó rơi
vào các segment chưa được gán.

**Lỗi trong chính công cụ đo, đã sửa.** Bản đầu của `overlap_eval.py` ánh xạ
speaker dự đoán sang nhãn bằng Hungarian tự do theo chuẩn đánh giá diarization.
Nhưng nhãn ở đây được viết bằng từ vựng roster của chính lượt chạy — người gán
nhãn nghe `Speaker 2` ở đoạn sạch rồi nói nguồn overlap là người đó. Với ánh xạ
tự do, một vụ đảo hai người nhất quán được đổi tên thành điểm tuyệt đối: công cụ
báo **đúng 28.6%, nhầm 0%** thay vì **đúng 0%, nhầm 28.6%**. Nay so sánh theo
danh tính; lượt chạy thứ hai được gióng roster bằng thời lượng nói ở vùng **không
overlap** — mốc neo độc lập với thứ đang được chấm. Có test khoá lại.

**Cảnh báo về quy mô:** 7 mẫu, 2 dòng sai đến từ 1 quyết định. Đây là mẫu thí
điểm, không phải bằng chứng về tỉ lệ. Điều nó đã làm được là lật ngược một kết
luận và lộ ra một lỗi trong thước đo — đúng việc của một mẻ thí điểm.

**Chưa nghiệm thu, và đây là điều quan trọng nhất của mục này:** `padded` mới chỉ
được chứng minh là *sinh ra embedding*, chưa được chứng minh là *gán đúng người*.
Phần đệm nằm ở vùng không chồng tiếng — nếu separator để lọt một giọng vào cả hai
nguồn thì embedding bị nhiễm chéo, và số nguồn vượt mốc tăng lên trong khi tên gán
lại sai. Vượt mốc không phải bằng chứng của một liên kết đúng (spec 18 rule 7).
Muốn kết luận thì phải có bộ nhãn — đó là lý do hai artefact còn lại tồn tại.

### Cập nhật 18/08/2026 (b) — bộ đếm người đồng thời được nạp bằng chứng

Một upload 20 phút dựng sẵn 3 người (`overlap_3speaker_vi_20min.mp3`) cho thấy
`estimate_source_count` được gọi với `CountingEvidence()` **rỗng** ở cả hai
pipeline, nên quy tắc 4 của spec §5.3 luôn bắn: giả định K=2, gắn
`count_uncertain`. Toàn bộ 55 segment overlap của job đó ghi
`estimated_concurrent_speakers = 2`, và các dòng K=3/K=4 của bảng routing trở
thành code chết — vùng 3 người bị ép qua separator 2 nguồn.

Thêm `CountingEvidence.diarization_active_speakers` (số cluster mà diarization
báo đang nói trong vùng) và quy tắc 3b tương ứng, xếp **dưới** TS-VAD và
multichannel vì nó không có confidence hiệu chỉnh — và không bịa ra một cái:
`confidence=None`, `count_uncertain=True`, method `diarization_activity`.

Chạy lại đúng file đó: 4 segment nay ghi `estimated_concurrent_speakers = 3`, đi
đường `MIXTURE_ASR_UNSUPPORTED` với `source_track = NULL` thay vì bị tách làm hai
nguồn giả, và job lần đầu tiên mang cảnh báo `unsupported_concurrency`. Router
không phải sửa gì — nó đã đúng từ trước, chỉ chưa bao giờ được kích hoạt.

Kèm theo: `/v1/jobs/{id}/result` ở nhánh nạp lại từ PostgreSQL (topology queue)
thiếu số người trong phiên nên console luôn hiện `—`. Nay cả hai nhánh trả
`session_speakers`, đếm từ segment và loại sink `Unknown`.

### Cập nhật 18/08/2026 (a) — ngôn ngữ chốt theo phiên, linking dùng nhãn diarization

Một upload 15 phút tiếng Việt chạy bằng `SASTT_ENGINE=real` cho 86/91 segment
overlap ra `Unknown` và 20 segment chứa chữ Hán/Hangul/Cyrillic. Điều tra bằng đo
đạc trên chính weight đang chạy tìm ra hai nguyên nhân độc lập.

**Ngôn ngữ nhận dạng lại trên từng crop.** `asr.language: null` được truyền vào
mọi lần `transcribe`, kể cả crop 0.4 s của nguồn overlap đã tách; log ghi 274 lần
detect ra 11 ngôn ngữ trên một file đơn ngữ. Đo lại theo độ dài crop trên chính
audio đó: 0.3 s đúng 18%, 1 s đúng 68%, 2 s đúng 82%, 30 s đúng 100%; text rác
chỉ xuất hiện ở crop ≤1 s. Pipeline nay chốt ngôn ngữ **một lần cho cả phiên** từ
speech đã gộp (`asr.language_detection.mode: auto_once`), không đủ tự tin thì
không chốt và thêm cảnh báo `session_language_uncertain`. Chạy lại đúng file đó:
text rác 20 → **0**, số lần detect 274 → **0**, transcript ngoài overlap giữ
nguyên trong sai số 0.1% (13 487 → 13 473 ký tự).

**Gán nguồn overlap.** Ranh giới thành/bại là độ dài chứ không phải chất lượng
model: segment overlap hỏng dài trung bình 395 ms, segment thành công 1 984 ms.
Đo CAM++ trên audio sạch cho thấy dưới 500 ms embedding là nhiễu — ở 300 ms
cosine cùng người (0.046) thấp hơn khác người (0.077). pyannote thì phân biệt
được ≥2 người ở **100%** các vùng đó (32/32 cặp nguồn đồng thời), nhưng nhánh
overlap không dùng thông tin này. Nay `link_sources` nhận `candidate_keys` từ
`regular_tracks` và chỉ chấm điểm với người đang nói
(`source_linking.restrict_to_active_clusters`, mặc định bật), và
`source_linking.min_embedding_ms` tách mốc **so** với centroid khỏi mốc **dựng**
centroid.

Ngưỡng `0.55/0.10` không còn hardcode ở `api/http.py` và `workers/offline_worker.py`;
cả hai đọc `configs/linking-thresholds.demo.yaml` qua `load_linking_overlay`, hoặc
`SASTT_LINKING_THRESHOLDS` nếu được đặt. File đó tự khai `status: unapproved`.

**Chưa nghiệm thu:** hai cơ chế gán nguồn độc lập (embedding + Hungarian có ràng
buộc, và đối chiếu năng lượng theo vùng nói riêng) chỉ đồng thuận 3/6 trên bài
toán nhị phân — đúng mức ngẫu nhiên. Không có nhãn người gán thì không kết luận
được cơ chế nào đúng, nên `source_linking.short_source_policy` mặc định vẫn là
`unknown`; nhánh `diarization_constrained` đã hiện thực nhưng tắt.

### Cập nhật 14/08/2026 — chặn transcript không khả dĩ

Một file upload 20 phút chạy bằng `SASTT_ENGINE=real` đã phát hiện ASR có thể sinh
một câu dài trong cửa sổ vài trăm mili-giây ở vùng overlap. Đây là lỗi logic kiểm
tra đầu ra, không phải transcript mẫu được pipeline nạp vào runtime. Pipeline nay
đặt physical-plausibility gate cho cả nguồn đã tách, mixture fallback và non-overlap:

- có ít nhất 100 ms speech do VAD xác nhận khi xuất text;
- từ token thứ tư, tổng speech VAD **và** khoảng thời gian timestamp của các từ đều
  phải đạt ít nhất 60 ms/từ;
- không đạt thì không gán speaker/không xuất câu, giữ audio nguồn, thêm warning
  `unreliable_separated_transcript`, `unreliable_mixture_transcript` hoặc
  `unreliable_non_overlap_transcript`, và trả `DEGRADED_SUCCEEDED`.

Lượt chạy kiểm chứng `job_01M0009…` không còn bản câu lỗi ở cả hai mốc đã quan
sát. Đây là guard cấu trúc, không phải cam kết WER hay thay thế benchmark.

Ngoài ra `torch.device("cuda")` được đổi thành `cuda:0` trong adapter pyannote và
CAM++: `clearvoice` gọi `torch.cuda.set_device()` toàn cục khi chọn GPU trống
nhất, làm adapter khác bám theo và chia tensor lên 2 GPU (`weight is on cuda:1,
different from other tensors on cuda:0`).

## 5. Đã kiểm chứng trên model thật (T4 16 GB)

Đo bằng `pytest -m model`, audio thật từ VoxConverse. Đây là **smoke test**,
không phải benchmark — gate DER/SI-SDRi/WER thuộc spec 16.4/16.5.

| Thành phần | Quan sát |
|---|---|
| faster-whisper turbo | load 4.3 s, decode RTF ≈ 0.09 trên T4 (`int8_float16`), có word timestamp |
| MossFormer2 | tách 2 người thật, SI-SDR cải thiện so với mixture ở cả 2 nguồn |
| CAM++ | same-speaker cosine 0.77–0.95, cross-speaker 0.06–0.23 |
| Hungarian linking | danh tính bám theo giọng kể cả khi đảo thứ tự source (S03 trên audio thật) |
| ffmpeg decoder | decode WAV/FLAC/MP3/M4A/Ogg, giữ channel gốc, tạo mono 16 kHz |
| pyannote community-1 | đếm đúng 2 speaker toàn phiên, trả regular + exclusive tracks |
| pyannote segmentation-3.0 | vùng overlap 2393–5819 ms so với dựng thật 2500–5000 ms |

### DoD M1 — một file thật chạy hết pipeline

Gửi qua HTTP (`POST /v1/jobs`, `SASTT_ENGINE=real`), audio dựng từ VoxConverse:
clean A 0–2.5 s · overlap 2.5–5 s · clean B 5–7.5 s.

```text
 start    end  ovl  src  status       label                 text
    60   2200  no   —    anonymous    Speaker 1             Các bạn hãy đăng ký kênh…
  2393   5713  yes  0    provisional  Temporary Speaker 1   Well, yeah, I mean, it totally is…
  2393   5073  yes  1    provisional  Temporary Speaker 2   [nội dung overlap đã ẩn]
  5819   7319  no   —    anonymous    Speaker 1             [nội dung overlap đã ẩn]
```

Hai segment overlap giữ nguyên timestamp chồng nhau và mang **hai**
`session_speaker_id` khác nhau (spec 0.1.7). Mọi confidence là `null` +
`confidence_status="uncalibrated"`. Retry cùng `Idempotency-Key` trả lại đúng
`job_id` cũ với `created=false`, không chạy lại job (S13).

Đây vẫn là **smoke test**, không phải benchmark: gate DER/SI-SDRi/WER cần corpus
của spec 16.4.

### Ma trận input đã chạy thật (spec 1.1, 3, 5.1.4)

`tests/model/test_spec_conformance.py`, weights thật, 2× Tesla T4:

| Hạng mục spec | Đã chạy | Kết quả |
|---|---|---|
| 1.1 container | WAV, FLAC, MP3, M4A/AAC, Ogg/Opus | cả 5 chạy hết pipeline |
| 1.1 sample rate | 8 / 16 / 44.1 / 48 kHz | pass |
| 1.1 + 5.1.4 kênh | 1, 2, 4, 6, 8 | giữ nguyên bản gốc, mono 16 kHz là derivative |
| 3 — E2E RTF | 5.5 phút audio, một job/GPU | **RTF 0.297** (mục tiêu `<= 0.50`) |
| 12 / S09 | production + research flag | từ chối khởi động, đúng error |

RTF đo trên T4 với overlap ratio thấp; spec 3 nói con số này phải được xác nhận
bằng **load test** (spec 16.1.6) chứ không phải một lần chạy — chưa làm.

**Không có test nào khẳng định số người nhận diện được.** Đó là câu hỏi accuracy,
thuộc benchmark spec 16.4/16.5, và spec 21.1 còn để ngỏ diarization default. Đo
thực tế: 2 người → nhận 2, 3 người → nhận 3, nhưng **4 người → nhận 3 và 5 người
→ nhận 3** (đếm thiếu). Con số này được ghi lại làm mốc, không được dùng làm gate
và không sửa code để làm nó đẹp lên (spec 18 rule 5).

### Near-realtime trên model thật (spec 4.2, FR-011)

Stream PCM 40 ms qua `StreamingSession`, weights thật:

```text
events: transcript.provisional 7 · transcript.final 4 · pipeline.warning 1 · session.finalized 1
final segments 4 · overlap 2 · distinct overlap speakers 2
replay(from 0) → 14 events (S12)
xRT 0.86 · max push latency 4529 ms
```

Đường này **trước đó hỏng hoàn toàn** trên model thật: final pass đưa PCM không
header vào decoder ffmpeg (`ffprobe could not read the input`). Fake decoder
chấp nhận nên fake test không phát hiện. Đã sửa bằng cách bọc RIFF/WAVE.

> **Trạng thái cập nhật — chưa được chấp nhận.** Lần test tay gần nhất với audio
> thực vẫn thấy luồng live không ổn định: audio có hai người nói đồng thời nhưng
> kết quả thường là `Speaker 1` cùng `Temporary Speaker 1`, đôi khi xuất hiện
> `Temporary Speaker 2` hoặc `Unknown`; người dùng cũng quan sát stream kết thúc
> trước khi toàn bộ file được thể hiện trong kết quả. Vì vậy số liệu smoke ở trên
> chỉ chứng minh transport/event cơ bản, **không chứng minh** speaker continuity,
> overlap attribution, hay end-of-stream completeness.
>
> Đã bổ sung các sửa chữa chưa đủ bằng chứng nghiệm thu: giới hạn rolling window,
> điều tiết `WebSocket.bufferedAmount` ở client, final pass trên toàn bộ PCM đã
> spool và reconcile nhãn provisional sau finalization. Cần chạy lại cùng một
> corpus audio thật, lưu frame/byte/audio-duration từ log và đối chiếu transcript
> cuối trước khi nâng trạng thái M3. Không được coi việc thay model là cách che
> lỗi logic/transport này.

### ASR final `large-v3` (spec 0.2)

Chạy lần đầu tiên — trước đó pin 2.9 GiB nhưng chưa thực thi dòng nào:

| | turbo (realtime) | large-v3 (final) |
|---|---|---|
| decode 5 s audio | 0.75 s (RTF 0.150) | 0.87 s (RTF 0.175) |
| text | giống nhau | giống nhau |
| word timestamp cuối | 4320 ms | 3900 ms |

Chậm hơn ~17% mà text không khác trên mẫu này; timestamp lệch 420 ms. Chưa trả
lời được spec 21.6 (turbo hay large-v3 cho tiếng Việt) — cần benchmark.

## 6. Công việc đang dở — cần chọn một hướng

**Trạng thái:** chờ quyết định. Không có việc nào khác đang bị chặn bởi mục này,
nhưng mọi cải thiện tiếp theo cho vùng overlap đều nằm sau nó.

### 6.1 Vấn đề còn lại

Sau các thay đổi ngày 18/08, vùng overlap vẫn còn hai giới hạn mà không cấu hình
nào gỡ được:

1. **Đoạn ngắn không định danh được.** Trên file 15 phút, 34/41 segment overlap
   vẫn `Unknown`; trên file 3 người 20 phút là 43/50.
   **Chẩn đoán này đã được sửa ngày 19/08 — xem cập nhật bên dưới.** Bản cũ ghi
   nguyên nhân là "dưới 500 ms embedding là nhiễu", nhưng phần lớn nguồn ngắn
   chưa từng được embed: mốc tối thiểu là 1500 ms còn trung vị vùng overlap là
   660 ms, nên adapter trả `InsufficientSpeechForEmbeddingError` và không có
   vector nào để so. Giới hạn thông tin ở mục 5 vẫn đúng cho vùng thực sự ngắn,
   nhưng nó không phải nguyên nhân trội.
2. **Ba người nói cùng lúc không tách được.** Bộ đếm nay báo đúng K=3 và router
   trả `MIXTURE_ASR_UNSUPPORTED` — trung thực, nhưng vẫn là không có transcript
   phân theo người cho vùng đó. MossFormer2 chỉ tách 2 nguồn.

### 6.2 Hai hướng, chọn một

| | **A · Stage SepFormer 3mix** | **B · Chuyển sang TS-ASR (DiCoW)** |
|---|---|---|
| Bản chất | Bổ sung separator 3 nguồn cho đường hiện tại | Bỏ hẳn khâu tách nguồn và gán nguồn |
| Giải quyết | Giới hạn 2 (ba người đồng thời) | Cả giới hạn 1 và 2 |
| Việc phải làm | `prestage_models.py` cho `sepformer_libri3mix`, rà licence (`beta_only`, `revision: null`), bật `three_source_beta`, nâng `max_supported_concurrent_speakers: 3`, benchmark | Manifest + adapter mới, rà licence, wiring port ASR, một lượt decode mỗi người |
| Rủi ro | SepFormer libri3mix là 8 kHz, phải resample; chất lượng trên tiếng Việt chưa biết | Thay đổi kiến trúc lớn nhất; chất lượng phụ thuộc mạnh vào diarization |
| Chi phí T4 | Thêm một model thường trú; VRAM hiện chỉ còn ~1–2 GB mỗi GPU | Backbone whisper-large-v3-turbo, fp16; N lượt decode cho N người |
| Giữ được gì | Toàn bộ pipeline hiện tại | Diarization giữ nguyên; separation + linking thành code chết |

### 6.3 Điều kiện chung cho cả hai

Cả A lẫn B đều **không nghiệm thu được nếu không có bộ dữ liệu có nhãn**. Đo đạc
ngày 18/08 cho thấy hai cơ chế gán nguồn độc lập chỉ đồng thuận 3/6 trên bài toán
nhị phân — đúng mức ngẫu nhiên — nên hiện không có cách nào so sánh A với B, hay
so sánh bất kỳ phương án nào với hiện trạng, ngoài việc đếm số segment có tên.
Đếm số lượng không phải đo độ chính xác.

Vì lý do đó `source_linking.short_source_policy` vẫn mặc định `unknown`: bật
`diarization_constrained` làm `Unknown` giảm 34 → 16 trên file mẫu, nhưng không
có gì chứng minh 19 cái tên mới là đúng.

**Việc cần làm trước tiên, bất kể chọn A hay B:** 20–30 phút audio cùng miền, gán
nhãn ai nói câu nào trong vùng overlap. Xem mục 10 để biết vị trí của nó trong
danh sách nợ.

## 7. Fullstack — hạ tầng đã chạy thật (spec 10, 11.1, 11.3)

| Thành phần | Trạng thái |
|---|---|
| PostgreSQL 14 + pgvector 0.8.0 | chạy; 11 bảng theo spec 10.2 |
| Redis 6 | chạy; 8 queue theo spec 11.3 |
| Migration | `deploy/migrate.py`, checksum, idempotent |
| Job/Event store | `PostgresJobStore`, `PostgresEventStore` |
| Voice registry | `PgVectorVoiceRegistry` — HNSW cosine, tenant-scoped, audit |
| Queue | `RedisTaskQueue` — at-least-once, backpressure, requeue task của worker chết |
| Worker | `sastt.workers.offline_worker` — process riêng, SIGTERM graceful, đọc `audio_key` từ object storage |
| Object storage | `S3ObjectStore` (S3/MinIO), tenant prefix, SSE request; compose có `minio-init` tạo bucket trước ingest |
| Metrics | `/metrics` Prometheus text exposition; guard cấm label chứa text/tên/embedding vẫn áp dụng |
| Container | 5 Dockerfile spec 11.1 + `docker-compose.yml` (có Redis/Postgres/MinIO) |

Test hạ tầng chạy trên Postgres/Redis thật: **21 passed** (`pytest -m db`).

Đường đi end-to-end đã chứng minh, engine `real`:

```text
1. API tạo job trong PostgreSQL            job_01KZZBKMFV... QUEUED
2. API đẩy vào Redis queue                 speaker.batch depth=1
3. Worker process nhận và xử lý            24.3 s, depth về 0
4. Đọc kết quả lại từ PostgreSQL           SUCCEEDED, 4 segment
      60-2200   overlap=False  Speaker 1
    2393-5713   overlap=True   Temporary Speaker 1
    2393-5073   overlap=True   Temporary Speaker 2
    5819-7319   overlap=False  Speaker 1
```

**Cập nhật wiring:** đặt `SASTT_JOB_RUNNER=queue` khiến API tạo job idempotent
trong PostgreSQL, lưu audio theo key `jobs/<job_id>/input` ở S3/MinIO rồi enqueue
`speaker.batch`; worker chỉ nhận object key, không nhận bytes hoặc local path. Test
API khóa SHA-256 input, tenant scope và queue payload; `/metrics` export theo
Prometheus. Docker/compose vẫn **chưa build/chạy thử được ở máy này** vì không có
Docker daemon, nên không được coi đây là bằng chứng fullstack GPU/MinIO đã chạy.

**Vẫn chưa xong:** auth/TLS production, OTel tracing, load/soak và benchmark.

## 8. Scenario acceptance (spec 16.2)

Chạy trên fake adapters, không tải weights:

| ID | Nội dung | Trạng thái |
|---|---|---|
| S01 | 2–5 người, không overlap | pass |
| S02 | 2 người overlap giữa phiên | pass |
| S03 | separator đảo thứ tự source | pass (cả fake và model thật) |
| S04 | overlap ngay giây đầu | pass — temporary ID → revision → final |
| S11 | separator lỗi | pass — retry crop nhỏ hơn rồi degraded, không mất audio |
| S12 | realtime reconnect | pass — replay đúng, không trùng final |
| S13 | retry idempotent | pass |
| S05–S07, S15 | enrollment quality, reject uncalibrated, tenant isolation | API local pass; Voice ID inference vẫn fail-closed tới khi có calibration |
| S08 | 3-source/beta | route/feature-gate + fake 3-source wiring đã có; acceptance model thật chờ pre-stage SepFormer |
| S09–S10, S14 | đa kênh/GSS, WeSep/model revision | chưa triển khai adapter production; giữ feature gate fail-closed |

## 9. Lệch spec — đã rào, không giấu

| Chỗ lệch | Lý do | Rào chắn |
|---|---|---|
| Tenant lấy từ header `X-Tenant-Id` | auth thật chưa có | `create_app(environment=production)` raise ngay (spec 14.2) |
| Demo console `web/` + route `/v1/demo/*` | công cụ dev để test tay | không thuộc cây spec 17; banner luôn nói engine đang dùng |
| Threshold linking trong demo được set sẵn | spec để `null` và fail closed | chỉ áp trong `create_app` demo, config gốc vẫn `null` |
| `ports/audio.py`, `ports/fusion.py` ngoài cây spec 17 | spec 9 bắt buộc 3 port này | ghi chú trong `ports/__init__.py` |
| Một môi trường Python cho mọi model | tiện phát triển | spec 11.1 yêu cầu tách image; cài `clearvoice` đã hạ numpy 2.5→1.26 |

## 10. Nợ kỹ thuật đã biết

- **Chưa có bộ dữ liệu có nhãn cho vùng overlap** — nợ chặn nhiều thứ nhất hiện
  nay. Không có nó thì không nghiệm thu được hướng A hay B ở mục 6, không đặt
  được ngưỡng linking từ số đo, và không bật được
  `source_linking.short_source_policy: diarization_constrained`. Quy mô tối thiểu:
  20–30 phút audio cùng miền, gán nhãn ai nói câu nào trong vùng overlap. Khác với
  benchmark corpus 10–20 giờ của spec 16.4: bộ này để **hiệu chỉnh**, không để
  công bố số.
- **Chưa tách image theo worker** (spec 11.1). Xung đột numpy khi cài chung
  pyannote + clearvoice là bằng chứng cho việc phải tách ở production.
- **Chưa có calibration release được duyệt.** Code đã có `FileConfidenceCalibrator` đọc release JSON versioned, nhưng config mặc định vẫn không trỏ release; do đó mọi output mặc định vẫn `null` + `confidence_status="uncalibrated"` và Voice ID fail closed.
- **Chưa có benchmark corpus** 10–20 giờ (spec 16.4) → chưa được phát biểu bất kỳ
  con số accuracy nào.
- **OSD adapter đã chạy thật** nhưng chưa calibration: pyannote.audio 4.x bỏ
  `OverlappedSpeechDetection`, nên adapter tự chạy segmentation model, tự
  convert powerset→multilabel (soft), tự `Inference.aggregate` rồi binarise theo
  hysteresis onset/offset của spec 5.2. Đường này đã kiểm chứng trên weights
  thật, nhưng onset/offset vẫn là seed value của spec 5.2 — chốt sau benchmark
  (spec 21.4).
- **CAM++ checkpoint chưa chốt**: dùng bản `zh_en` vì phiên là tiếng Việt xen
  tiếng Anh, nhưng không bản nào train trên tiếng Việt → `benchmark_pending`
  trong manifest, chốt sau benchmark (spec 21.2).
- **Nguồn gốc bản convert `faster-whisper-large-v3-turbo`**: SYSTRAN không phát
  hành bản CTranslate2 cho turbo, đang dùng bản convert cộng đồng (MIT). Licence
  review phải xác minh riêng lớp này (spec 20).
- **GPU pinning là `cuda:0` cứng** trong adapter pyannote/CAM++ để tránh
  `clearvoice` kéo tensor sang GPU khác. Đúng cho một worker một GPU (spec 11.1),
  nhưng khi tách worker thì device phải lấy từ config chứ không hard-code.
- **Near-realtime chưa có evidence E2E trên audio thật.** Cần kiểm tra tuần tự:
  frame nhận/gửi và thời lượng PCM, backlog/đóng WebSocket, final pass có bao phủ
  đủ audio, rồi mới đánh giá separator/diarizer/linker bằng DER, overlap recall,
  speaker-confusion và tỷ lệ `Unknown`.

## 11. Nợ còn lại sau khi đóng M1

| Việc | Vì sao chưa xong |
|---|---|
| **Compose chưa build/chạy thử** | môi trường hiện tại không có Docker daemon/CLI; cần chạy integration thật với Postgres, Redis, MinIO và GPU image |
| **Chưa có auth/TLS** | tenant vẫn lấy từ header `X-Tenant-Id`; `create_app(environment=production)` từ chối khởi động (spec 14.1–14.2) |
| **Chưa có OTel tracing/autoscale** | `/metrics` Prometheus đã có; stream ghi stage duration/RTF, buffer và GPU VRAM khi PyTorch/CUDA hiện diện, nhưng chưa có trace exporter hay policy autoscale queue-age |
| **Chưa có load/soak evidence** | `deploy/capacity_report.py` tính p95 và không pass nếu thiếu số đo; còn RTF/latency/VRAM 30–60 phút phải chạy trên baseline GPU thật |
| **Chưa có benchmark corpus/release** | `deploy/benchmark_report.py` và confidence calibration release đã có, nhưng corpus 10–20h + duyệt threshold/model release chưa có |
| **SBOM chưa vào CI/release** | `deploy/generate_sbom.py` tạo inventory dependency + model manifest local; cần artifact ký/scan trong pipeline release |
| **Registry persistent chưa nối vào app** | API Voice Registry chạy local bằng registry in-memory để không phụ thuộc Docker; deploy cần inject `PgVectorVoiceRegistry` + auth context thay vì state local |

## 11.1 Artefact M3–M5 mới

- `StreamingSession` chỉ giữ ring buffer trong RAM; PCM đầy đủ được spool tạm để final pass, nên kiểm thử có thể chứng minh bộ đệm RAM bị chặn.
- ASR mặc định Whisper auto-detect; console/API có language hint per-job (`auto`, `vi`, `en`) và đưa hint vào `config_version`, tránh ép file tiếng Anh sang tiếng Việt.
- `sastt.adapters.speechbrain.SepFormerLibri3MixSeparator` là adapter beta K=3, yêu cầu thư mục weight local và `three_source_beta`; không tải model ở runtime.
- `FileConfidenceCalibrator` chỉ xuất confidence khi có release JSON hợp lệ; chưa cấu hình release thì vẫn null/fail-closed.
- `deploy/benchmark_report.py`, `deploy/capacity_report.py`, `deploy/generate_sbom.py` tạo evidence local. Report capacity đánh dấu `pending` khi thiếu sample, không tự xác nhận SLO.

Ví dụ:

```bash
python3 deploy/generate_sbom.py --output artifacts/sbom.json
python3 deploy/benchmark_report.py evidence.jsonl --release-id bench_2026_08 --output artifacts/benchmark.json
python3 deploy/capacity_report.py load-measurements.json --output artifacts/capacity.json
```

## 12. Chạy thử

Cần: Python 3.10+, ffmpeg, NVIDIA GPU + CUDA 12. Đo trên 2× Tesla T4.

```bash
pip install -e ".[dev,api]"
# adapter model thật (không nằm trong extras vì spec 11.1 muốn tách image):
pip install "pyannote.audio==4.0.7" faster-whisper modelscope clearvoice addict
pip install "numpy<2.0,>=1.24.3"   # clearvoice ghim numpy 1.x, cài sau cùng
```

Tải weights (cần HF token đã accept điều kiện 2 model pyannote gated):

```bash
HF_TOKEN=... python3 deploy/prestage_models.py --all --models-dir /models
python3 deploy/prestage_models.py --verify     # hash lại weights trên đĩa
```

```bash
SASTT_ENGINE=real uvicorn --factory sastt.api.http:create_app \
    --app-dir src --port 8000
# http://localhost:8000 — chọn kịch bản, chạy offline hoặc near-realtime
# GET /readyz cho biết engine, config version và backend nào đã pin
```

Gửi audio thật:

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: key-001' \
  -H 'X-Tenant-Id: tenant-demo' \
  -d "{\"audio_base64\": \"$(base64 -w0 file.wav)\"}"
curl "http://127.0.0.1:8000/v1/jobs/<job_id>/result" -H 'X-Tenant-Id: tenant-demo'
```

```bash
pytest                 # test thường, không tải weights, không cần HF token
pytest -m model        # 32 test, cần weights trong /models
ruff check src tests deploy && ruff format --check src tests deploy && mypy
python3 deploy/prestage_models.py --list      # xem backend nào đã pin
```

# Implementation status

| Thuộc tính | Giá trị |
|---|---|
| Cập nhật | 14/08/2026 |
| Spec tham chiếu | [`production-technical-spec.md`](production-technical-spec.md) v1.0 |
| Milestone hiện tại | M0 xong · M1 ~95% · **M2 ~70%** · **M3 ~60%** — hạ tầng fullstack đã chạy |
| Engine mặc định | `fake` (M0). Đặt `SASTT_ENGINE=real` để dùng adapter model thật |

Tài liệu này ghi **tình trạng thực tế** của repo. Mọi con số đều lấy từ lần chạy
thật, không ước lượng.

## 1. Milestone (spec 18)

| Milestone | Phạm vi | Trạng thái |
|---|---|---|
| M0 — Foundation & contracts | package `sastt`, config validation, domain models, ports, JSON Schema v2, fake adapters, state machine, revision/idempotency, CI | **xong** |
| M1 — Offline 2-speaker path | decode/resample, adapter pyannote + faster-whisper + MossFormer2 + CAM++, pin manifest, smoke test | **~95%** — DoD pass; còn nợ mục 10 |
| M2 — Linking & Voice ID | pgvector registry, enrollment quality, deletion/audit | **~70%** — registry + schema + audit chạy thật; chưa nối vào API |
| M3 — Near-realtime | queue/worker, backpressure, latency instrumentation | **~60%** — WebSocket/revision/replay chạy trên model thật; Redis queue + worker chạy thật; chưa có autoscale/metric export |
| M4 — Beta/advanced overlap | SepFormer 3mix, concurrent counter, GSS, WeSep | chưa bắt đầu |
| M5 — Production hardening | benchmark corpus, calibrator, load/soak, SBOM, capacity | chưa bắt đầu |

## 2. Gate chất lượng — lần chạy gần nhất

| Gate | Lệnh | Kết quả |
|---|---|---|
| Lint | `ruff check src tests deploy` | pass |
| Format | `ruff format --check src tests deploy` | pass |
| Type | `mypy` (strict) | pass, 53 file |
| Test thường | `pytest` | **206 passed**, 57 deselected |
| Test model | `pytest -m model` | **32 passed, 0 skipped** |
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
| `sepformer_libri3mix` | `speechbrain/sepformer-libri3mix` | — | — | beta, chưa cần (M4) |
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
  2393   5073  yes  1    provisional  Temporary Speaker 2   Hãy subscribe cho kênh La La School…
  5819   7319  no   —    anonymous    Speaker 1             Hãy subscribe cho kênh La La School…
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

### ASR final `large-v3` (spec 0.2)

Chạy lần đầu tiên — trước đó pin 2.9 GiB nhưng chưa thực thi dòng nào:

| | turbo (realtime) | large-v3 (final) |
|---|---|---|
| decode 5 s audio | 0.75 s (RTF 0.150) | 0.87 s (RTF 0.175) |
| text | giống nhau | giống nhau |
| word timestamp cuối | 4320 ms | 3900 ms |

Chậm hơn ~17% mà text không khác trên mẫu này; timestamp lệch 420 ms. Chưa trả
lời được spec 21.6 (turbo hay large-v3 cho tiếng Việt) — cần benchmark.

## 6. Fullstack — hạ tầng đã chạy thật (spec 10, 11.1, 11.3)

| Thành phần | Trạng thái |
|---|---|
| PostgreSQL 14 + pgvector 0.8.0 | chạy; 11 bảng theo spec 10.2 |
| Redis 6 | chạy; 8 queue theo spec 11.3 |
| Migration | `deploy/migrate.py`, checksum, idempotent |
| Job/Event store | `PostgresJobStore`, `PostgresEventStore` |
| Voice registry | `PgVectorVoiceRegistry` — HNSW cosine, tenant-scoped, audit |
| Queue | `RedisTaskQueue` — at-least-once, backpressure, requeue task của worker chết |
| Worker | `sastt.workers.offline_worker` — process riêng, SIGTERM graceful |
| Container | 5 Dockerfile spec 11.1 + `docker-compose.yml` (9 service) |

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

**Chưa xong:** Docker chưa build được ở máy này (không có daemon) nên Dockerfile
và compose mới chỉ được kiểm tra cú pháp; API vẫn chạy job in-process thay vì
đẩy vào queue; chưa có auth/TLS; chưa export metric ra Prometheus; MinIO/object
storage mới khai báo trong compose, chưa có adapter.

## 7. Scenario acceptance (spec 16.2)

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
| S05–S10, S14, S15 | Voice ID, đa kênh, model revision, cross-tenant | chờ M2/M4 |

## 8. Lệch spec — đã rào, không giấu

| Chỗ lệch | Lý do | Rào chắn |
|---|---|---|
| Job chạy in-process, chưa có queue/worker | topology spec 11.1/11.3 thuộc M3 | ghi rõ trong `api/http.py` |
| Tenant lấy từ header `X-Tenant-Id` | auth thật chưa có | `create_app(environment=production)` raise ngay (spec 14.2) |
| Demo console `web/` + route `/v1/demo/*` | công cụ dev để test tay | không thuộc cây spec 17; banner luôn nói engine đang dùng |
| Threshold linking trong demo được set sẵn | spec để `null` và fail closed | chỉ áp trong `create_app` demo, config gốc vẫn `null` |
| `ports/audio.py`, `ports/fusion.py` ngoài cây spec 17 | spec 9 bắt buộc 3 port này | ghi chú trong `ports/__init__.py` |
| Một môi trường Python cho mọi model | tiện phát triển | spec 11.1 yêu cầu tách image; cài `clearvoice` đã hạ numpy 2.5→1.26 |

## 9. Nợ kỹ thuật đã biết

- **Chưa tách image theo worker** (spec 11.1). Xung đột numpy khi cài chung
  pyannote + clearvoice là bằng chứng cho việc phải tách ở production.
- **Chưa có calibration**. Mọi confidence là `null` với
  `confidence_status="uncalibrated"`; threshold linking/Voice ID vẫn `null` và
  fail closed (spec 5.10, 18 rule 7).
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

## 10. Còn nợ để đóng M1

| Việc | Vì sao chưa xong |
|---|---|
| **API vẫn chạy job in-process** | queue + worker đã chạy thật, nhưng `api/http.py` chưa đẩy job vào Redis. Đây là việc nối dây còn lại của spec 11.1 |
| **Docker chưa build thử** | máy phát triển không có Docker daemon; 5 Dockerfile + compose mới chỉ kiểm tra cú pháp |
| **Chưa có auth/TLS** | tenant vẫn lấy từ header `X-Tenant-Id`; `create_app(environment=production)` từ chối khởi động (spec 14.1–14.2) |
| **Chưa export metric** | chỉ `InMemoryMetrics`; spec 13.1 cần Prometheus/OTel để autoscale theo queue age |
| **Chưa có object storage adapter** | MinIO đã khai báo trong compose nhưng chưa có adapter `ObjectStore` dùng S3 |
| **Chưa có load test** | RTF 0.297 là *một* lần chạy; spec 3 yêu cầu xác nhận bằng load test (spec 16.1.6), gồm p95, soak 30–60 phút và giới hạn VRAM/GPU 80% |

## 11. Chạy thử

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
pytest                 # 206 test, không tải weights, không cần HF token
pytest -m model        # 12 test, cần weights trong /models
ruff check src tests deploy && ruff format --check src tests deploy && mypy
python3 deploy/prestage_models.py --list      # xem backend nào đã pin
```

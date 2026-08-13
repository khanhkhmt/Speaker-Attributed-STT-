# Implementation status

| Thuộc tính | Giá trị |
|---|---|
| Cập nhật | 13/08/2026 |
| Spec tham chiếu | [`production-technical-spec.md`](production-technical-spec.md) v1.0 |
| Milestone hiện tại | M0 xong · M1 phần lớn xong, chặn ở 2 checkpoint gated |
| Engine mặc định | `fake` (M0). Đặt `SASTT_ENGINE=real` để dùng adapter model thật |

Tài liệu này ghi **tình trạng thực tế** của repo. Mọi con số đều lấy từ lần chạy
thật, không ước lượng.

## 1. Milestone (spec 18)

| Milestone | Phạm vi | Trạng thái |
|---|---|---|
| M0 — Foundation & contracts | package `sastt`, config validation, domain models, ports, JSON Schema v2, fake adapters, state machine, revision/idempotency, CI | **xong** |
| M1 — Offline 2-speaker path | decode/resample, adapter pyannote + faster-whisper + MossFormer2 + CAM++, pin manifest, smoke test | **~85%** — chặn ở pyannote gated |
| M2 — Linking & Voice ID | pgvector registry, enrollment quality, deletion/audit | chưa bắt đầu |
| M3 — Near-realtime | queue/worker, backpressure, latency instrumentation | một phần (WebSocket + revision/replay đã chạy in-process) |
| M4 — Beta/advanced overlap | SepFormer 3mix, concurrent counter, GSS, WeSep | chưa bắt đầu |
| M5 — Production hardening | benchmark corpus, calibrator, load/soak, SBOM, capacity | chưa bắt đầu |

## 2. Gate chất lượng — lần chạy gần nhất

| Gate | Lệnh | Kết quả |
|---|---|---|
| Lint | `ruff check src tests deploy` | pass |
| Format | `ruff format --check src tests deploy` | pass |
| Type | `mypy` (strict) | pass, 46 file |
| Test thường | `pytest` | **206 passed**, 16 deselected |
| Test model | `pytest -m model` | **9 passed, 3 skipped** (skip có lý do) |
| Coverage | `--cov=sastt.domain --cov=sastt.application` | 88% (spec 16.3 yêu cầu ≥85%) |

Test model chạy trên weights thật + audio thật (VoxConverse dev, public,
multi-speaker). 3 test skip vì thiếu checkpoint pyannote — **không** fallback
sang fake rồi báo pass (spec 18 rule 6).

## 3. Model weights (spec 11.2, 20)

Pin bằng `deploy/prestage_models.py`; runtime chỉ mount `/models` read-only.

| Backend | Repo | Revision | Dung lượng | Trạng thái |
|---|---|---|---:|---|
| `faster_whisper` (turbo) | `deepdml/faster-whisper-large-v3-turbo-ct2` | `4df90f75…` | 1.5 GiB | pinned + verified |
| `faster_whisper_large_v3` | `Systran/faster-whisper-large-v3` | `edaa852e…` | 2.9 GiB | pinned + verified |
| `mossformer2_ss_16k` | `alibabasglab/MossFormer2_SS_16K` | `407cb030…` | 639 MiB | pinned + verified |
| `3d_speaker_campplus` | `iic/speech_campplus_sv_zh_en_16k-common_advanced` (ModelScope) | `v1.0.0` | 27 MiB | pinned + verified |
| `pyannote-community-1` | `pyannote/speaker-diarization-community-1` | — | — | **chặn: chưa accept terms** |
| `pyannote_segmentation_3.0` | `pyannote/segmentation-3.0` | — | — | **chặn: chưa accept terms** |
| `sepformer_libri3mix` | `speechbrain/sepformer-libri3mix` | — | — | beta, chưa cần (M4) |
| `multidecoder_dprnn` | `JunzheJosephZhu/MultiDecoderDPRNN` | — | — | `deny` production (spec 20) |
| `gpu_gss`, `wesep` | — | — | — | không có weights / phase 2 |

Vì 2 checkpoint pyannote chưa pin, `validate_for_environment(..., PRODUCTION)`
**từ chối khởi động** — đúng hành vi spec yêu cầu.

## 4. Blocker duy nhất

`pyannote/speaker-diarization-community-1` và `pyannote/segmentation-3.0` là
gated model. Token HF của tài khoản `khanh99` có `canReadGatedRepos: True`,
nhưng tải file vẫn trả 403 *"you are not in the authorized list"* → tài khoản
**chưa bấm accept điều kiện** trên trang model.

Cách gỡ:

1. Đăng nhập HF, mở 2 trang model, điền form điều kiện và accept (gated `auto`,
   duyệt ngay).
2. Chạy `python3 deploy/prestage_models.py --only diarization osd --models-dir /models`
3. Chạy `pytest -m model` — 3 test đang skip sẽ chạy, gồm cả DoD end-to-end của M1.

Sau đó `SASTT_ENGINE=real` sẽ chạy được toàn bộ pipeline trên audio thật.

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

## 6. Scenario acceptance (spec 16.2)

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

## 7. Lệch spec — đã rào, không giấu

| Chỗ lệch | Lý do | Rào chắn |
|---|---|---|
| Job chạy in-process, chưa có queue/worker | topology spec 11.1/11.3 thuộc M3 | ghi rõ trong `api/http.py` |
| Tenant lấy từ header `X-Tenant-Id` | auth thật chưa có | `create_app(environment=production)` raise ngay (spec 14.2) |
| Demo console `web/` + route `/v1/demo/*` | công cụ dev để test tay | không thuộc cây spec 17; banner luôn nói engine đang dùng |
| Threshold linking trong demo được set sẵn | spec để `null` và fail closed | chỉ áp trong `create_app` demo, config gốc vẫn `null` |
| `ports/audio.py`, `ports/fusion.py` ngoài cây spec 17 | spec 9 bắt buộc 3 port này | ghi chú trong `ports/__init__.py` |
| Một môi trường Python cho mọi model | tiện phát triển | spec 11.1 yêu cầu tách image; cài `clearvoice` đã hạ numpy 2.5→1.26 |

## 8. Nợ kỹ thuật đã biết

- **Chưa tách image theo worker** (spec 11.1). Xung đột numpy khi cài chung
  pyannote + clearvoice là bằng chứng cho việc phải tách ở production.
- **Chưa có calibration**. Mọi confidence là `null` với
  `confidence_status="uncalibrated"`; threshold linking/Voice ID vẫn `null` và
  fail closed (spec 5.10, 18 rule 7).
- **Chưa có benchmark corpus** 10–20 giờ (spec 16.4) → chưa được phát biểu bất kỳ
  con số accuracy nào.
- **OSD adapter chưa chạy thật**: pyannote.audio 4.x bỏ
  `OverlappedSpeechDetection`, nên adapter tự chạy segmentation model rồi
  binarise theo hysteresis onset/offset của spec 5.2. Đường này **chưa được
  kiểm chứng** vì weights còn gated.
- **CAM++ checkpoint chưa chốt**: dùng bản `zh_en` vì phiên là tiếng Việt xen
  tiếng Anh, nhưng không bản nào train trên tiếng Việt → `benchmark_pending`
  trong manifest, chốt sau benchmark (spec 21.2).
- **Nguồn gốc bản convert `faster-whisper-large-v3-turbo`**: SYSTRAN không phát
  hành bản CTranslate2 cho turbo, đang dùng bản convert cộng đồng (MIT). Licence
  review phải xác minh riêng lớp này (spec 20).

## 9. Chạy thử

```bash
pip install -e ".[dev,api]"
uvicorn --factory sastt.api.http:create_app --app-dir src --port 8000
# http://localhost:8000 — chọn kịch bản, chạy offline hoặc near-realtime
```

```bash
pytest                 # 206 test, không tải weights, không cần HF token
pytest -m model        # cần weights trong /models; skip có lý do nếu thiếu
ruff check src tests deploy && mypy
python3 deploy/prestage_models.py --list      # xem backend nào đã pin
python3 deploy/prestage_models.py --verify    # hash lại weights trên đĩa
```

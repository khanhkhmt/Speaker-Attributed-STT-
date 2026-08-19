# Hướng dẫn chạy và vận hành Speaker-Attributed STT

Tài liệu này là runbook cho repository `sastt`: từ chạy test, demo giao diện,
chạy model thật, đến triển khai API + Redis + PostgreSQL + MinIO + worker GPU.
Đặc tả sản phẩm là nguồn yêu cầu gốc tại
[`production-technical-spec.md`](production-technical-spec.md). Trạng thái các
phần chưa nghiệm thu nằm tại
[`implementation-status.md`](implementation-status.md).

> Phạm vi hiện tại: luồng offline 2 nguồn (`M1`) chạy với model thật. Những
> chức năng beta/production nêu ở cuối tài liệu không được coi là đã sẵn sàng
> chỉ vì dịch vụ khởi động được.

## 1. Hai chế độ chạy

| Chế độ | Biến môi trường | Dùng cho | Không dùng cho |
|---|---|---|---|
| `fake` | `SASTT_ENGINE=fake` (mặc định) | Test, demo structural với scenario tích hợp sẵn | Chép lời audio thật hoặc đánh giá model |
| `real` | `SASTT_ENGINE=real` | Audio thật với weight đã pin tại `/models` | Chạy khi weight, dependency hoặc GPU chưa sẵn sàng |

`fake` không phải model rút gọn. Nó dùng adapter deterministic để kiểm thử hợp
đồng và state machine. API chủ động trả `MODEL_NOT_READY` nếu gửi audio thật
vào fake engine.

Luồng offline đúng theo spec §8.1 là:

```text
QUEUED → PREPROCESSING → DIARIZING → TRANSCRIBING
       → SEPARATING (chỉ khi có overlap) → LINKING → FUSING
       → SUCCEEDED | DEGRADED_SUCCEEDED | FAILED | CANCELLED
```

`FUSING` là bước cuối: gắn word/turn vào speaker, giữ các đoạn chồng thời gian,
gom thành utterance và tạo transcript canonical. Nó không có nghĩa là job đã
xong. Giao diện poll cho đến khi job đi vào terminal state; với file dài, không
đóng trang chỉ vì job đang ở một stage trong vài phút.

`DEGRADED_SUCCEEDED` vẫn là kết quả hợp lệ nhưng có một nhánh phải fail-safe. Với
overlap, pipeline chỉ đưa transcript vào kết quả khi cả VAD và timestamp từng từ
cho thấy mật độ lời nói khả dĩ (từ thứ tư trở đi cần tối thiểu 60 ms/từ). Nếu
không đạt, pipeline không gán speaker/không xuất câu đó, giữ nguyên audio nguồn
và trả warning `unreliable_separated_transcript`,
`unreliable_mixture_transcript` hoặc `unreliable_non_overlap_transcript`.

## 2. Điều kiện trước khi chạy

### 2.1 Phần mềm

- Git.
- Python 3.10+; repository đã được kiểm tra với Python 3.12.
- `ffmpeg` có trong `PATH` để decode/probe WAV, FLAC, MP3, M4A/AAC và Ogg/Opus.
- Với model test/local: NVIDIA driver, CUDA tương thích và GPU NVIDIA.
- Với topology container: Docker Engine, Docker Compose v2 và NVIDIA Container
  Toolkit.

Kiểm tra host/container GPU trước khi chạy worker thật:

```bash
docker --version
docker compose version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi
```

Lệnh cuối phải thấy GPU trong container. Nếu không, chưa chạy được worker GPU.

### 2.2 Tài khoản model và dung lượng

Hai model pyannote là gated. Tài khoản Hugging Face phải accept điều khoản của
`pyannote/speaker-diarization-community-1` và
`pyannote/segmentation-3.0`, sau đó tạo token có quyền đọc.

Sáu weight lõi M1 cần stage trước runtime:

| Khóa stage | Thư mục mặc định | Vai trò |
|---|---|---|
| `diarization` | `pyannote-community-1` | Diarization |
| `osd` | `pyannote-segmentation-3.0` | Overlap speech detection |
| `asr_realtime` | `faster-whisper-large-v3-turbo` | ASR realtime/default |
| `asr_final` | `faster-whisper-large-v3` | ASR final tùy chọn |
| `separation_two_source` | `mossformer2-ss-16k` | Tách overlap hai nguồn |
| `embedding` | `campplus` | Speaker embedding/linking |

Weight không được commit vào Git. Chuẩn bị vài GiB dung lượng đĩa và mount thư
mục model read-only khi chạy worker.

## 3. Chuẩn bị mã nguồn và môi trường Python

Từ thư mục repository:

```bash
git clone <URL_REPOSITORY> sastt
cd sastt

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,api]"
```

Các dependency trên đủ cho test, fake engine và API development. Để chạy
API/worker theo queue ở local cần thêm:

```bash
pip install "psycopg[binary,pool]" redis boto3
```

Để thử real engine trong **một môi trường local duy nhất** (chỉ phù hợp debug,
không phải layout production):

```bash
pip install "pyannote.audio==4.0.7" faster-whisper modelscope clearvoice addict
pip install "numpy<2.0,>=1.24.3"
```

`clearvoice` ghim NumPy 1.x trong khi pyannote có dependency khác ở một số
phiên bản. Production phải dùng image worker tách biệt; không nâng/hạ NumPy
ngẫu nhiên trong môi trường đang chạy.

## 4. Kiểm tra mã trước khi khởi động

Các gate không cần model, Hub token, GPU, database hay Redis:

```bash
pytest -q
ruff check src tests deploy
ruff format --check src tests deploy
mypy
pytest --cov=sastt.domain --cov=sastt.application --cov-report=term
```

Kiểm tra manifest/weight trên worker:

```bash
python3 deploy/prestage_models.py --list
python3 deploy/prestage_models.py --verify --models-dir /models
```

`--verify` hash lại từng file so với manifest. Không sửa manifest bằng tay để
che mismatch; stage lại weight để script cập nhật revision/SHA-256 có kiểm soát.

Model smoke test cần đủ dependency, weight, GPU và dữ liệu test:

```bash
pytest -m model -q
```

Model smoke pass không phải benchmark accuracy; benchmark corpus, calibration và
load evidence là các gate riêng.

Lần kiểm tra mã gần nhất (19/08/2026): `ruff check`, format, `mypy` (62 file) và
`pytest -q` đều pass (**303 passed, 59 deselected**). `pytest -m model`
**34 passed** trên weight thật và `pytest -m db` **21 passed** trên PostgreSQL +
Redis thật, cả hai đo ngày 18/08; máy ngày 19/08 không có Postgres/Redis nên gate
`db` chưa chạy lại.

## 5. Chạy nhanh giao diện development (fake engine)

Đây là đường nhanh nhất để kiểm tra API, UI, scenario, JSON contract và realtime
event structure mà không tải model:

```bash
source .venv/bin/activate
SASTT_ENGINE=fake \
uvicorn --factory sastt.api.http:create_app --app-dir src --host 127.0.0.1 --port 8000
```

Mở <http://127.0.0.1:8000>. Chọn một scenario có sẵn rồi bấm **Chạy chép lời**.
Không upload audio thật ở chế độ này.

Probe từ terminal khác:

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/v1/demo/scenarios
```

`/healthz` chỉ xác nhận process HTTP sống. `/readyz` trả engine, config version
và trạng thái pin manifest; dùng `/readyz` trước khi kết luận real engine sẵn
sàng. Dừng server bằng `Ctrl+C` tại terminal chạy Uvicorn.

## 6. Stage và xác minh model thật

Chỉ tải model ở bước chuẩn bị; runtime production không được tự download.
Ví dụ stage vào `/models`:

```bash
export HF_TOKEN='<Hugging-Face-read-token>'
python3 deploy/prestage_models.py --all --models-dir /models
python3 deploy/prestage_models.py --verify --models-dir /models
unset HF_TOKEN
```

Hoặc chỉ stage một nhóm:

```bash
python3 deploy/prestage_models.py --only diarization osd --models-dir /models
```

Sau stage, xác nhận sáu target lõi đều `[ok]`. Manifest `gpu_gss`, `wesep`,
`sepformer_libri3mix` và `multidecoder_dprnn` không pin trong mặc định là bình
thường: chúng thuộc phase/beta/research khác. Vì vậy UI có thể ghi `6/10 model
manifests pinned`; đó không phải lỗi M1 nếu sáu model ở §2.2 đã verified.

## 7. Chạy real engine đơn tiến trình (debug/local)

Đường này không cần Postgres, Redis hay MinIO nếu bỏ `SASTT_JOB_RUNNER=queue`.
API xử lý job trong process và giữ kết quả trong memory; phù hợp debug một máy,
không phù hợp restart/scale.

```bash
source .venv/bin/activate
export SASTT_ENGINE=real
uvicorn --factory sastt.api.http:create_app --app-dir src --host 0.0.0.0 --port 8000
```

Kiểm tra:

```bash
curl -fsS http://127.0.0.1:8000/readyz
```

Gửi một file. Không nhét base64 của file lớn trực tiếp vào command line vì có
thể vượt giới hạn `ARG_MAX`; tạo JSON request trong file tạm bằng Python chuẩn:

```bash
python3 - ./sample.wav >/tmp/sastt-job-request.json <<'PY'
import base64
import json
import pathlib
import sys

audio = pathlib.Path(sys.argv[1]).read_bytes()
print(json.dumps({"audio_base64": base64.b64encode(audio).decode(), "language": "auto"}))
PY

curl -sS -X POST http://127.0.0.1:8000/v1/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: local-sample-001' \
  -H 'X-Tenant-Id: tenant-demo' \
  --data-binary @/tmp/sastt-job-request.json
```

Xóa file tạm khi không cần nữa:

```bash
rm /tmp/sastt-job-request.json
```

Lưu `job_id` trả về. `language` có thể là `auto`, `vi` hoặc `en`; `auto` là mặc
định. Dùng key idempotency mới nếu muốn tạo job mới; dùng lại key với audio khác
sẽ nhận `409` thay vì chạy lặp không an toàn.

Lấy trạng thái/kết quả:

```bash
curl -sS -H 'X-Tenant-Id: tenant-demo' \
  http://127.0.0.1:8000/v1/jobs/<job_id>

curl -sS -H 'X-Tenant-Id: tenant-demo' \
  http://127.0.0.1:8000/v1/jobs/<job_id>/result
```

Trong development, `X-Tenant-Id` chỉ là trợ giúp demo. Production không được
tin header này; ứng dụng từ chối khởi động production nếu auth context thật
chưa được cung cấp.

## 8. Chạy topology queue ở local (API + worker + hạ tầng)

Luồng upload có persistence là:

```text
Browser/API → PostgreSQL (job) + MinIO (audio) → Redis speaker.batch → worker → PostgreSQL (result)
```

### 8.1 Cấu hình hạ tầng

Khuyến nghị Compose ở §9. Nếu hạ tầng đã được cung cấp bên ngoài, export đúng
endpoint/credential trước khi chạy API và worker:

```bash
export DATABASE_URL='postgresql://sastt:<password>@127.0.0.1:5432/sastt'
export REDIS_URL='redis://127.0.0.1:6379/0'
export S3_ENDPOINT_URL='http://127.0.0.1:9000'
export S3_BUCKET='sastt-audio'
export S3_PREFIX='sastt'
export AWS_ACCESS_KEY_ID='<access-key>'
export AWS_SECRET_ACCESS_KEY='<secret-key>'
export AWS_REGION='us-east-1'
```

Khởi tạo schema **trước** worker/API:

```bash
python3 deploy/migrate.py --status
python3 deploy/migrate.py
python3 deploy/migrate.py --status
```

Migration idempotent và kiểm tra checksum. Nếu migration `applied` nhưng
checksum khác, dừng lại: không sửa migration lịch sử trên database đang dùng.
Bucket S3/MinIO phải tồn tại trước khi API nhận upload.

### 8.2 Khởi động worker trước API

Terminal thứ nhất, với cùng `DATABASE_URL`, `REDIS_URL`, S3 credential,
`SASTT_ENGINE=real` và quyền đọc `/models`:

```bash
source .venv/bin/activate
export SASTT_ENGINE=real
python3 -m sastt.workers.offline_worker --queues speaker.batch asr.batch
```

Worker sẽ log `consuming speaker.batch, asr.batch`. Không chạy hai worker dùng
cùng GPU nếu chưa chủ động phân chia VRAM/device.

Terminal thứ hai chạy API queue:

```bash
source .venv/bin/activate
export SASTT_ENGINE=real
export SASTT_JOB_RUNNER=queue
uvicorn --factory sastt.api.http:create_app --app-dir src --host 0.0.0.0 --port 8000
```

API tạo job idempotent, lưu input theo key `jobs/<job_id>/input` ở object storage
và enqueue `speaker.batch`. Worker chạy pipeline rồi lưu result PostgreSQL.

### 8.3 Theo dõi job đúng cách

Poll endpoint đến terminal state, không chỉ đợi một khoảng cố định:

```bash
JOB_ID='<job_id>'
while :; do
  STATE="$(curl -fsS -H 'X-Tenant-Id: tenant-demo' \
    "http://127.0.0.1:8000/v1/jobs/${JOB_ID}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])')"
  printf '%s\n' "$STATE"
  case "$STATE" in
    SUCCEEDED|DEGRADED_SUCCEEDED|FAILED|CANCELLED) break ;;
  esac
  sleep 3
done
```

Chỉ gọi `/result` khi state là `SUCCEEDED`/`DEGRADED_SUCCEEDED`. `409 job is
<state>, not finished` nghĩa là pipeline chưa xong, không phải transcript mất.
Với `FAILED`, xem `error_code` trên endpoint job và log worker.

Để dừng an toàn, gửi `SIGTERM` hoặc `Ctrl+C` tới worker; worker hoàn thành task
đang cầm rồi thoát. Không `kill -9` trừ sự cố; task chưa ack sẽ được recovery
theo cơ chế queue at-least-once khi worker phù hợp khởi động lại.

## 9. Chạy bằng Docker Compose (khuyến nghị cho topology)

Từ root repository, đặt vị trí model host và secret development qua shell hoặc
file `.env` **không commit**:

```bash
export MODELS_DIR=/models
export POSTGRES_PASSWORD='<development-password>'
export MINIO_USER='sastt'
export MINIO_PASSWORD='<development-minio-password>'
```

Stage weight vào `MODELS_DIR` trước. Sau đó:

```bash
docker compose -f deploy/docker/docker-compose.yml up -d postgres redis minio
docker compose -f deploy/docker/docker-compose.yml run --rm migrate
docker compose -f deploy/docker/docker-compose.yml up -d api speaker-worker
```

Kiểm tra service/probe/log:

```bash
docker compose -f deploy/docker/docker-compose.yml ps
docker compose -f deploy/docker/docker-compose.yml logs --tail=100 api
docker compose -f deploy/docker/docker-compose.yml logs --tail=100 speaker-worker
curl -fsS http://127.0.0.1:8000/readyz
```

`api` publish cổng `8000`. PostgreSQL, Redis và MinIO trong compose hiện không
publish cổng host; truy cập chúng qua network Compose hoặc thêm port mapping
trong override development, không sửa trực tiếp cấu hình production.

Compose còn định nghĩa `asr-worker`, `fusion-worker` và `gss-worker` để phản
ánh topology mục tiêu. Với offline queue hiện tại API enqueue `speaker.batch`,
nên `speaker-worker` là worker bắt buộc để xử lý upload. `gss-worker` chỉ chạy
khi bật profile và feature tương ứng:

```bash
docker compose -f deploy/docker/docker-compose.yml --profile gss up -d gss-worker
```

Worker image mount `/models:ro`, chạy offline Hub (`HF_HUB_OFFLINE=1`) và mỗi
GPU worker được gán một thiết bị. Không mount writeable model directory vào
worker runtime.

Để dừng stack nhưng giữ dữ liệu volume:

```bash
docker compose -f deploy/docker/docker-compose.yml down
```

`down -v` xóa volume Postgres/Redis/MinIO; chỉ dùng sau khi xác nhận muốn xóa
toàn bộ dữ liệu development.

## 10. Giao diện web và API

Giao diện ở `http://<host>:8000/`; không có bước build frontend riêng. Endpoint
dùng thường xuyên:

| Method | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/healthz` | Liveness HTTP |
| `GET` | `/readyz` | Engine/config/model-manifest readiness |
| `GET` | `/metrics` | Prometheus text exposition |
| `GET` | `/v1/demo/scenarios` | Danh sách scenario development |
| `POST` | `/v1/jobs` | Tạo job; bắt buộc `Idempotency-Key` |
| `GET` | `/v1/jobs/{job_id}` | State/warning/error job |
| `GET` | `/v1/jobs/{job_id}/result` | Kết quả final v2 |
| `GET` | `/v1/jobs/{job_id}/audio` | Audio input của job, để nghe lại một segment |
| `DELETE` | `/v1/jobs/{job_id}` | Hủy job chưa terminal hoặc xóa artifact local theo state API |
| `POST` | `/v1/sessions` | Tạo session realtime/WebSocket development |
| `POST` | `/v1/sessions/{session_id}/finalize` | Finalize session |
| `GET` | `/v1/sessions/{session_id}/result` | Transcript realtime final |

Trang web cache JavaScript trong browser. Sau khi deploy thay đổi
`web/index.html`, dùng hard refresh (`Ctrl+Shift+R` hoặc `Cmd+Shift+R`) trước
khi kiểm tra UI mới.

### 10.1 Nghe lại một segment

Bấm vào một dòng transcript để nghe đúng khoảng `start_ms`–`end_ms` của dòng đó.
Thanh player có hai chế độ: *Chỉ đoạn đã bấm* dừng ở cuối segment, *Phát liên
tục* chạy tiếp và tự highlight dòng đang phát.

Audio phát ra là **input gốc của job**, không phải nguồn đã tách. Một nguồn tách
là sản phẩm của model; phát nó như thể đó là bản ghi sẽ trình bày sai việc
pipeline đã làm. Vì vậy vùng overlap sẽ nghe thấy cả hai người cùng lúc — đó là
đúng, và là cách kiểm tra bằng tai xem việc gán nguồn có hợp lý không.

Console giữ audio của **12 job gần nhất** (`MAX_RETAINED_JOB_AUDIO`). Cũ hơn
mức đó, hoặc sau khi restart API ở chế độ đơn tiến trình, endpoint trả `404` và
player ghi rõ audio không còn được giữ — chạy lại job nếu cần nghe. `DELETE
/v1/jobs/{job_id}` xóa luôn audio đã giữ.

Audio thô là dữ liệu sinh trắc (spec 10.3, 14.4). Bound ở trên chỉ chặn RAM của
môi trường development; deployment thật giữ audio trong object storage theo
retention policy chứ không giữ trong tiến trình API.

### 10.2 Gán nhãn vùng chồng tiếng

Vùng overlap là chỗ duy nhất pipeline đang hỏng, và không thể cải thiện nó nếu
không phân biệt được một *cái tên đúng* với một *cái tên*. Console có chế độ gán
nhãn để tạo bộ đối chứng đó.

1. Mở một job đã xong, bấm **◐ Gán nhãn** ở thanh công cụ transcript.
2. Đặt bộ lọc về **Chỉ đồng thời** để chỉ thấy các dòng cần gán.
3. Bấm **♪** cạnh mỗi tên để nghe giọng mẫu của người đó — 5 giây đầu của đoạn
   nói sạch dài nhất mà họ có trong phiên. Đây là cách biết `Speaker 2` là ai
   trước khi gán. Rê chuột lên tên sẽ thấy câu họ nói trong đoạn mẫu đó.
4. Bấm một dòng → nó phát đúng đoạn audio đó.
5. Gõ `1`…`9` để gán người tương ứng, hoặc `0` nếu **không nghe ra ai nói**.
6. Sau mỗi lần gán, con trỏ tự nhảy sang dòng chưa gán kế tiếp và phát luôn.

| Phím | Tác dụng |
|---|---|
| `1`–`9` | Gán người nói thứ n |
| `n` | Vùng này thực ra chỉ có **một** người nói |
| `l` | Dòng này lẫn chữ của **hai** người — luồng tách bị rò |
| `0` | Không nghe ra — không tính vào điểm số |
| `Shift`+`1`–`9` | Nghe lại giọng mẫu của người thứ n |
| `Space` | Nghe lại dòng hiện tại |
| `↑` `↓` hoặc `k` `j` | Dòng trước / dòng sau |

Nhãn lưu trong `localStorage` theo `job_id`, nên đóng tab không mất. Bấm **Xuất
nhãn** để tải file JSON.

#### Nguyên tắc gán

Mỗi dòng là **một luồng đã tách**, không phải một khoảng thời gian. Separator cắt
vùng chồng tiếng thành hai luồng và chép lời riêng từng luồng, nên câu hỏi là:

> **Ai đã nói câu hiển thị ở dòng này?**

Không phải "trong khoảng này ai nói nhiều hơn". Anh không nghe được luồng đã tách
— player cố tình phát audio gốc — nên **cột Nội dung là thứ xác định dòng đó là
luồng nào**. Nghe đoạn audio, rồi gán câu chữ đó cho giọng mình nghe ra.

Vài chữ ở mép đoạn rơi vào chỗ chồng tiếng là bình thường và **không** đổi câu
trả lời: một luồng tách ra vốn phải chở đúng một người xuyên suốt vùng đó.

Trong thực tế hai giọng đồng thời là hai người khác nhau — nhưng đó là *thực tế*,
không phải *đầu ra của separator*. Separator hoàn toàn có thể nhét cùng một người
vào cả hai luồng. Nếu nghe kỹ mà thấy cả hai dòng đều là một giọng, hãy gán **cùng
một tên cho cả hai**: đó là câu trả lời trung thực, và nó ghi lại một lỗi tách
nguồn thay vì che đi.

Bốn câu trả lời không phải tên người, mỗi cái đo một tầng khác nhau của pipeline:

| | Nghĩa | Nó đo cái gì |
|---|---|---|
| `n` — chỉ 1 người | OSD báo có chồng tiếng nhưng thật ra không | Lỗi **phát hiện overlap** |
| `l` — lẫn 2 giọng | Một dòng chứa chữ của cả hai người | Lỗi **tách nguồn** |
| `0` — không nghe ra | Có hai người nhưng tai không tách được ai là ai | **Trần của bài toán** |
| bỏ trống | Chưa gán | Chưa có dữ liệu |

Cả ba đều **không** bị tính là model sai — chúng được đếm riêng, vì mỗi cái chỉ về
một khâu khác nhau. Dùng `l` dè dặt: chỉ khi không tên nào đúng cho cả dòng. Còn
"phần lớn dòng này là giọng X" thì cứ gán X. Đoán bừa cho đủ mới làm hỏng bộ nhãn.

### 10.3 Chấm điểm gán người nói trong vùng overlap

```bash
python3 deploy/overlap_eval.py result.json --labels voxlane-labels-<job_id>.json
python3 deploy/overlap_eval.py before.json after.json --labels labels.json
```

Công cụ luôn trả **bộ ba** — đúng · nhầm · `Unknown` — chứ không bao giờ một con
số. Lý do: đổi `Unknown` trung thực lấy một cái tên sai tự tin sẽ làm "số segment
có tên" đẹp lên trong khi sản phẩm tệ đi. Ở chế độ so sánh, nó chỉ báo **CẢI
THIỆN** khi *đúng tăng* **và** *nhầm không tăng*.

Session speaker ID sinh ngẫu nhiên mỗi lần chạy, nên công cụ ánh xạ ID dự đoán
sang nhãn bằng Hungarian một-một trước khi chấm. Ánh xạ một-một nghĩa là một
người bị tách làm hai session speaker vẫn bị tính sai.

### 10.4 Thử cửa sổ embedding rộng hơn (chưa nghiệm thu)

Mặc định pipeline chỉ embed đúng lõi vùng overlap, nên vùng ngắn hơn 1500 ms
không sinh được embedding nào và chỉ có thể ra `Unknown`. Đặt
`source_linking.embedding_window: padded` để embed cả cửa sổ đã tách
(`vùng ± audio.overlap_context_seconds`) — audio đó separator đã tạo sẵn.

```yaml
# calibration-overlay.yaml, trỏ tới bằng SASTT_LINKING_THRESHOLDS
source_linking:
  accept_threshold: 0.55
  ambiguous_margin: 0.10
  embedding_window: padded
```

**Chưa được nghiệm thu trên dữ liệu có nhãn.** Phần đệm nằm ở vùng không chồng
tiếng; nếu separator để lọt một giọng vào cả hai nguồn thì embedding bị nhiễm
chéo và pipeline sẽ gán tên sai một cách tự tin. Đo bằng §10.3 trước khi bật cho
bất kỳ việc gì thật.

## 11. Quan sát và xử lý lỗi

### 11.1 Checklist khi upload không chạy

1. `curl -fsS http://127.0.0.1:8000/healthz` phải trả `{"status":"ok"}`.
2. `/readyz` phải báo `engine: real` nếu upload audio thật.
3. `python3 deploy/prestage_models.py --verify --models-dir /models` phải pass
   cả sáu target core trên máy worker.
4. Kiểm tra worker có process sống và log `consuming ...`.
5. API/worker phải dùng cùng PostgreSQL, Redis, bucket, prefix và credential;
   khác tenant/prefix khiến worker không đọc được input object.
6. Poll `/v1/jobs/<job_id>`; đừng suy diễn lỗi chỉ từ một state trung gian.
7. Xem log API, worker, GPU (`nvidia-smi`) và queue depth/age qua `/metrics`.

### 11.2 Triệu chứng phổ biến

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `MODEL_NOT_READY` | Fake engine, thiếu dependency/weight, path `/models` sai hoặc manifest chưa pin | Đặt `SASTT_ENGINE=real`, stage + `--verify`, kiểm tra mount và log worker |
| Job mãi `QUEUED` | Không có worker consume `speaker.batch`, Redis URL khác hoặc queue worker sai | Khởi động `speaker-worker`/worker local với cùng `REDIS_URL`; xem log/metrics |
| State giữa pipeline rất lâu | File dài, GPU bận/thiếu VRAM hoặc model đang xử lý | Theo dõi worker log + `nvidia-smi`; không retry đồng thời job cũ chưa terminal |
| `FUSING` | Đang ghép kết quả cuối, hoặc worker chạy code cũ | Worker hiện tại cập nhật stage tại thời điểm chạy; nếu giữ bất thường, xem log/version rồi restart graceful |
| `unsupported_concurrency` | Ba người trở lên nói cùng lúc; separator hai nguồn không xử lý được | Vùng đó trả transcript mixture, không phân theo người. Đúng hành vi, không phải lỗi cấu hình — xem giới hạn ở §13 |
| `session_language_uncertain` | Nhận dạng ngôn ngữ không đạt `min_probability` nên không chốt cho cả phiên | Kiểm tra audio có đủ speech sạch không; hoặc pin `asr.language` thủ công cho job đó |
| `unreliable_*_transcript` / `DEGRADED_SUCCEEDED` | ASR sinh số từ không khả dĩ cho VAD/timestamp (thường ở overlap hoặc khoảng lặng) | Giữ audio nguồn để nghe lại; không coi câu bị chặn là lời nói thật. Kiểm tra model/VAD trước khi điều chỉnh ngưỡng, rồi retry bằng key mới nếu cần. |
| UI “Lỗi” nhưng API job chưa terminal | Browser chạy JS cũ hoặc lỗi mạng phía client | Hard refresh; `/v1/jobs/<job_id>` là nguồn state chuẩn |
| `409` idempotency key | Key đã dùng cho audio khác | Tạo key mới; không tái sử dụng key cũ với payload khác |
| `409 job is ..., not finished` ở `/result` | Gọi result quá sớm | Poll state đến `SUCCEEDED`/`DEGRADED_SUCCEEDED` |
| Lỗi S3/MinIO/object not found | Bucket chưa tạo, endpoint/credential/prefix/tenant không khớp | Tạo bucket, đối chiếu toàn bộ `S3_*`/`AWS_*` ở API và worker |
| Worker chết lúc import model | Dependency GPU xung đột | Dùng Dockerfile worker tương ứng; không trộn asr/speaker stack tùy tiện |
| `/readyz` ghi 6/10 pinned | Bốn manifest là beta/phase 2/deny | Xác nhận sáu manifest core M1; không bật feature beta nếu chưa stage weight |

### 11.3 Jobs cũ và retry

Queue là at-least-once. Worker bị dừng trong task có thể để task trong processing
list; worker khởi động lại recovery task hết visibility timeout. Không xóa Redis
list, Postgres row hay object storage bằng lệnh ad-hoc khi chưa xác định `job_id`,
tenant, state và retention policy. Với job thất bại, giữ log/metadata trước khi
retry bằng **idempotency key mới**.

## 12. Kiểm thử hạ tầng và release evidence

Test PostgreSQL/Redis được đánh dấu riêng, để test mặc định không cần dịch vụ:

```bash
export SASTT_TEST_DATABASE_URL='postgresql://sastt:<password>@127.0.0.1:5432/sastt_test'
export SASTT_TEST_REDIS_URL='redis://127.0.0.1:6379/1'
pytest -m db -q
```

Dùng database/Redis namespace test riêng; không chạy test contract vào dữ liệu
ứng dụng đang vận hành.

Tạo evidence, không phải tự động phê duyệt production:

```bash
mkdir -p artifacts
python3 deploy/generate_sbom.py --output artifacts/sbom.json
python3 deploy/benchmark_report.py evidence.jsonl \
  --release-id bench_YYYY_MM --output artifacts/benchmark.json
python3 deploy/capacity_report.py load-measurements.json \
  --output artifacts/capacity.json
```

Thiếu đo đạc thì capacity report phải ở `pending`; không thay nó bằng số liệu
ước lượng.

## 13. Giới hạn phải biết trước khi đưa ra production

- Chưa có auth/TLS/production tenant context hoàn chỉnh. `X-Tenant-Id` chỉ dùng
  development.
- Confidence mặc định là `null`/`uncalibrated`; Voice ID fail-closed cho đến
  khi có calibration release được duyệt.
- Chưa có benchmark corpus 10–20 giờ, calibration được duyệt, load/soak
  evidence hoặc capacity evidence. Không đưa ra claim DER/WER/accuracy/SLO
  production.
- Near-realtime có transport/revision/replay nhưng chưa được nghiệm thu độ ổn
  định speaker continuity, overlap attribution và end-of-stream trên corpus
  thật.
- SepFormer 3-source, GPU-GSS, WeSep và Multi-Decoder DPRNN không phải đường
  production mặc định. Giữ feature gate tắt nếu chưa có model/evidence tương
  ứng.
- Voice Registry trong API hiện là local/in-memory; persistent pgvector cần
  wiring/auth phù hợp trước triển khai nhiều replica.
- **Gán người nói trong vùng chồng tiếng chưa dùng được cho sản phẩm.** Baseline
  đo trên nhãn tay của một file 20 phút: **đúng 0% · nhầm 28.6% · `Unknown`
  71.4%** trên 7 dòng quyết định được. Mẫu quá nhỏ để kết luận tỉ lệ, nhưng đủ để
  nói rằng phần ngoài overlap chạy tốt còn phần overlap thì chưa. Cách đo lại nằm
  ở §10.2–10.3.
- `source_linking.short_source_policy: diarization_constrained` **chưa được
  nghiệm thu trên dữ liệu có nhãn**. Bật nó là đánh đổi `Unknown` trung thực lấy
  rủi ro gán sai tên một cách tự tin. Giữ `unknown` cho đến khi có bộ đánh giá.
- `source_linking.embedding_window: padded` cũng **chưa nghiệm thu**. Nó đưa số
  vùng vượt mốc embedding từ 3/15 lên 10/15, nhưng không đổi dòng nào trong tập đã
  gán nhãn và sinh thêm một speaker thứ năm trong phiên ba người. Xem §10.4.
- **Ba người nói cùng lúc chưa tách được.** Bộ đếm nay báo đúng K=3 và router trả
  `MIXTURE_ASR_UNSUPPORTED` kèm cảnh báo `unsupported_concurrency` — vùng đó có
  transcript mixture nhưng không phân theo người. MossFormer2 chỉ tách 2 nguồn và
  weight 3 nguồn chưa được stage. Việc chọn separator 3 nguồn hay chuyển sang
  TS-ASR **không còn là bước kế tiếp** — hai việc rẻ hơn đứng trước nó, xem
  [`implementation-status.md` mục 6](implementation-status.md).
- **Đoạn overlap ngắn vẫn ra `Unknown`.** Nguyên nhân trội không phải embedding
  nhiễu mà là **không có embedding nào**: mốc tối thiểu 1500 ms trong khi trung vị
  vùng overlap là 660 ms, nên adapter từ chối tạo vector. Giới hạn thông tin dưới
  500 ms vẫn có thật nhưng nằm sau chỗ pipeline dừng. Chi tiết và số đo ở
  [`implementation-status.md` mục 6](implementation-status.md).

## 13.1 Ngôn ngữ ASR (`asr.language_detection`)

Whisper nhận dạng ngôn ngữ từ đúng đoạn audio được đưa vào, nên chạy nhận dạng
lại trên từng crop khiến nó phải quyết định từ vài trăm mili-giây của một nguồn
overlap đã tách. Đo trên một file tiếng Việt 15 phút: crop 0.3 s trả đúng 18%,
crop 30 s trả đúng 100%, và các lượt sai ngôn ngữ chính là nơi model sinh ra
credit phụ đề học thuộc (`한글자막 by …`, `Субтитры создавал …`).

| `mode` | Hành vi |
|---|---|
| `auto_once` (mặc định) | Nhận dạng một lần từ `sample_seconds` speech đã gộp, rồi dùng lại cho mọi lần gọi |
| `fixed` | Dùng `asr.language`, không nhận dạng. Bắt buộc phải đặt `asr.language` |
| `per_segment` | Khôi phục hành vi cũ, chỉ dùng cho phiên thật sự đa ngôn ngữ |

Nếu xác suất dưới `min_probability`, pipeline **không** chốt, thêm cảnh báo
`session_language_uncertain` và để backend tự quyết từng lần — thà không chốt
còn hơn chốt sai cho cả phiên.

## 13.2 Ngưỡng linking (`source_linking`)

Ngưỡng không còn hardcode trong API và worker. Cả hai đọc cùng một file để không
trôi lệch nhau:

```bash
export SASTT_LINKING_THRESHOLDS=/path/to/calibration-release.yaml
```

Mặc định development dùng `configs/linking-thresholds.demo.yaml`. **File đó không
phải calibration release**: hai con số trong đó chưa từng được đo. Không có file
nào được cấu hình thì ngưỡng giữ `null` và pipeline fail-closed.

Hai knob còn lại:

- `min_embedding_ms` — lượng speech tối thiểu để **so** một nguồn với centroid
  đã có. Khác với `speaker_embedding.minimum_clean_speech_seconds`, thứ quy định
  lượng speech để **dựng** centroid mới. `null` dùng chung mốc dựng centroid.
- `restrict_to_active_clusters` (mặc định bật) — chỉ chấm điểm nguồn với những
  người mà diarization báo đang nói trong vùng đó. Nếu diarization không có ý
  kiến về vùng đó, ma trận giữ nguyên thay vì ép tất cả thành `Unknown`.

## 14. Bàn giao vận hành tối thiểu

Trước khi nhận một deployment, người vận hành nên lưu lại:

1. Commit SHA repository và `config_version` từ `/readyz`.
2. Output `deploy/prestage_models.py --verify` cùng revision/SHA model manifest.
3. Phiên bản container image, CUDA/driver và `nvidia-smi`.
4. Migration status, phiên bản PostgreSQL/pgvector, Redis và bucket policy.
5. URL metric/alert, queue depth/age, log retention và owner xử lý incident.
6. Kết quả test, model smoke, benchmark, load/soak và SBOM đúng release đó.
7. Xác nhận backup/retention/xóa dữ liệu audio và biometric template theo policy.

Không thay thế các mục trên bằng screenshot giao diện “ready”: readiness chỉ là
một trong các điều kiện vận hành.

# MetaHRM Face Recognition Service

## Tổng Quan Dự Án

**Tên dịch vụ:** MetaHRM Face Recognition Service  
**Phiên bản:** 1.0.0  
**Ngôn ngữ:** Python 3.10+  
**Framework:** FastAPI 0.104.1  
**Loại:** RESTful Microservice  

**Mục đích:** Microservice nhận diện khuôn mặt phục vụ hệ thống chấm công của MetaHRM. Nhân viên đăng ký khuôn mặt một lần (enrollment), sau đó xác thực danh tính (verification) mỗi khi check-in. Service này xử lý toàn bộ phần AI/ML — phát hiện mặt, trích xuất embedding và so sánh — còn việc lưu trữ embedding và ghi nhận chấm công do Spring Boot backend đảm nhiệm.

---

## Cấu Trúc Thư Mục

```
metahrm-face-service/
├── app/
│   ├── main.py                          # Khởi động FastAPI, lifespan, middleware
│   ├── api/
│   │   └── routes/
│   │       ├── enrollment.py            # POST /api/face/enroll
│   │       ├── verification.py          # POST /api/face/verify
│   │       └── health.py                # GET /health, GET /
│   ├── core/
│   │   ├── config.py                    # Cấu hình toàn cục (Pydantic Settings)
│   │   ├── face_detector.py             # MTCNN wrapper — phát hiện khuôn mặt
│   │   └── face_recognizer.py           # FaceNet wrapper — tạo embedding
│   ├── models/
│   │   ├── request.py                   # DTO request (FaceEnrollRequest, FaceVerifyRequest)
│   │   └── response.py                  # DTO response (FaceEnrollResponse, FaceVerifyResponse)
│   ├── services/
│   │   ├── enrollment_service.py        # Business logic đăng ký khuôn mặt
│   │   ├── verification_service.py      # Business logic xác thực khuôn mặt
│   │   └── anomaly_detector.py          # Phát hiện bất thường (chất lượng, giả mạo,...)
│   └── utils/
│       ├── image_processor.py           # Xử lý ảnh: decode, validate, quality score
│       ├── distance_calculator.py       # Tính khoảng cách Euclidean và Cosine
│       └── logger.py                    # Cấu hình loguru
├── tests/
│   ├── test_api_integration.py          # Integration test toàn bộ API
│   ├── test_distance_calculator.py
│   ├── test_face_detector.py
│   ├── test_face_recognizer.py
│   ├── test_image_processor.py
│   ├── test_logger.py
│   ├── test_config.py
│   └── MetaHRM_FaceService.postman_collection.json
├── data/sample_faces/                   # Ảnh mẫu để test
├── logs/face_service.log                # File log tự động xoay vòng
├── weights/                             # Model weights (tự tải lần đầu ~100MB)
├── .env / .env.example                  # Biến môi trường
├── Dockerfile                           # Multi-stage build
├── docker-compose.yml
└── requirements.txt
```

---

## Tech Stack

### Framework & Server
| Thư viện | Phiên bản | Mục đích |
|----------|-----------|---------|
| FastAPI | 0.104.1 | Web framework async, tự động sinh Swagger docs |
| Uvicorn | 0.24.0 | ASGI server chạy FastAPI |

### AI / ML
| Thư viện | Phiên bản | Mục đích |
|----------|-----------|---------|
| facenet-pytorch | 2.5.3 | FaceNet — mô hình InceptionResnetV1 tạo embedding 512 chiều |
| mtcnn | 0.1.1 | Multi-task Cascaded CNN — phát hiện khuôn mặt và landmarks |
| torch | 2.1.0 | PyTorch — nền tảng deep learning |
| torchvision | 0.16.0 | Utilities xử lý ảnh cho PyTorch |

### Xử Lý Ảnh
| Thư viện | Phiên bản | Mục đích |
|----------|-----------|---------|
| opencv-python | 4.8.1.78 | Decode ảnh, resize, xử lý pixel |
| Pillow | 10.1.0 | PIL Image — chuyển đổi format ảnh |
| numpy | 1.24.3 | Tính toán ma trận embedding |

### Tính Khoảng Cách
| Thư viện | Phiên bản | Mục đích |
|----------|-----------|---------|
| scipy | 1.11.4 | Cosine distance, Euclidean distance |
| scikit-learn | 1.3.2 | Machine learning utilities hỗ trợ |

### Validation & Config
| Thư viện | Phiên bản | Mục đích |
|----------|-----------|---------|
| pydantic | 2.5.0 | Validate request/response data |
| pydantic-settings | 2.1.0 | Đọc cấu hình từ .env |
| python-dotenv | 1.0.0 | Load biến môi trường |

### Logging & Testing
| Thư viện | Phiên bản | Mục đích |
|----------|-----------|---------|
| loguru | 0.7.2 | Logging nâng cao, tự xoay vòng file |
| pytest | 7.4.3 | Test framework |
| httpx | 0.25.1 | HTTP client dùng trong test |
| pytest-asyncio | 0.21.1 | Hỗ trợ test async |

---

## Cấu Hình (.env)

```env
# Ứng dụng
APP_NAME=MetaHRM Face Recognition Service
APP_VERSION=1.0.0
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Mô hình AI
FACENET_MODEL=vggface2              # Hoặc 'casia-webface'
MTCNN_MIN_FACE_SIZE=20
MTCNN_THRESHOLDS=0.6,0.7,0.7      # 3 giai đoạn của MTCNN
MTCNN_FACTOR=0.709

# Ngưỡng nhận diện
EUCLIDEAN_THRESHOLD=1.0            # < 1.0 → cùng người
COSINE_THRESHOLD=0.6               # > 0.6 → cùng người
MIN_FACE_CONFIDENCE=0.95           # Độ tin cậy tối thiểu của MTCNN

# Xử lý ảnh
MAX_IMAGE_SIZE=10485760            # 10 MB
ALLOWED_EXTENSIONS=jpg,jpeg,png
TARGET_FACE_SIZE=160               # Input size FaceNet
MIN_IMAGE_WIDTH=200
MIN_IMAGE_HEIGHT=200
MAX_IMAGE_WIDTH=4000
MAX_IMAGE_HEIGHT=4000

# Anomaly Detection
MIN_IMAGE_QUALITY_SCORE=0.7
MAX_FACES_ALLOWED=1
MIN_FACE_AREA_RATIO=0.05

# CORS & Logging
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:5173
LOG_LEVEL=INFO
LOG_FILE=logs/face_service.log
```

---

## API Endpoints

### GET `/health`
Kiểm tra trạng thái service.

**Query Parameters:**
- `include_model_info` (bool, default=false) — Kèm thông tin mô hình AI

**Response:**
```json
{
  "status": "healthy",
  "service": "MetaHRM Face Recognition Service",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:00:00",
  "model_info": null
}
```

---

### POST `/api/face/enroll`
Đăng ký khuôn mặt nhân viên.

**Request:**
```json
{
  "employee_id": 123,
  "image_base64": "iVBORw0KGgoAAAANS...",
  "is_primary": true,
  "note": "Ghi chú tuỳ chọn"
}
```

**Response thành công:**
```json
{
  "success": true,
  "message": "Face enrolled successfully",
  "data": {
    "employee_id": 123,
    "embedding": [0.123, -0.456, ...],
    "face_confidence": 0.99,
    "quality_score": 0.85,
    "face_quality_metrics": {
      "confidence": 0.99,
      "size": 45000,
      "area_ratio": 0.25,
      "symmetry_score": 0.88,
      "brightness": 0.65
    },
    "is_primary": true,
    "detection_info": {
      "bounding_box": [x, y, w, h],
      "keypoints": {}
    },
    "anomalies": null
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

**Luồng xử lý:**
1. Decode base64 → numpy array RGB
2. Validate kích thước ảnh (200×200 đến 4000×4000)
3. Tính quality score (brightness, sharpness, contrast)
4. Phát hiện khuôn mặt bằng MTCNN
5. Validate đúng 1 mặt, confidence ≥ 0.95
6. Cắt vùng mặt với margin
7. Tạo embedding 512 chiều bằng FaceNet
8. Tính face quality metrics (symmetry, brightness)
9. Phát hiện bất thường
10. Trả về embedding + metadata

---

### POST `/api/face/verify`
Xác thực danh tính khi chấm công.

**Request:**
```json
{
  "employee_id": 123,
  "image_base64": "iVBORw0KGgoAAAANS...",
  "known_embeddings": [
    [0.123, -0.456, ...],
    [0.234, -0.567, ...]
  ],
  "verification_threshold": 80.0
}
```

**Response:**
```json
{
  "success": true,
  "message": "Verification completed successfully",
  "is_match": true,
  "confidence": 95.67,
  "details": {
    "euclidean_distance": 0.45,
    "cosine_similarity": 0.89,
    "best_match_index": 0,
    "method": "BOTH",
    "num_comparisons": 2,
    "face_confidence": 0.98,
    "quality_score": 0.82,
    "all_comparisons": [...]
  },
  "anomalies": null,
  "timestamp": "2024-01-15T08:30:00"
}
```

**Luồng xử lý:**
1. Decode ảnh → validate kích thước
2. Tính quality score
3. Phát hiện và validate khuôn mặt
4. Tạo embedding từ ảnh hiện tại
5. So sánh với từng embedding đã lưu (known_embeddings)
6. Áp dụng ngưỡng kép: Euclidean + Cosine
7. Tính confidence tổng hợp
8. Phát hiện bất thường
9. Trả về kết quả khớp + confidence

**Điều kiện xác nhận là cùng người (AND logic):**
- Euclidean distance < 1.0 **VÀ**
- Cosine similarity > 0.6 **VÀ**
- Confidence ≥ verification_threshold (mặc định 80%)

**Công thức tính confidence:**
```
euc_score = max(0, 1 - distance / 2.0)
confidence = (cosine_similarity × 0.6 + euc_score × 0.4) × 100
```

---

## Models AI

### MTCNN — Phát Hiện Khuôn Mặt
**File:** [app/core/face_detector.py](app/core/face_detector.py)

MTCNN (Multi-task Cascaded Convolutional Neural Network) gồm 3 giai đoạn:
- Stage 1: RPN (Region Proposal Network) — đề xuất vùng có thể là mặt
- Stage 2: R-CNN — lọc và tinh chỉnh vùng
- Stage 3: Hiệu chỉnh bounding box và keypoints

**Đầu ra:** bounding box `[x, y, w, h]`, keypoints (mắt, mũi, miệng), confidence score

**Phương thức chính:**
- `detect_faces(image)` → danh sách khuôn mặt phát hiện được
- `get_largest_face(image)` → cắt 1 mặt lớn nhất với margin
- `validate_single_face(image)` → kiểm tra đúng 1 mặt, đủ điều kiện
- `get_face_quality_metrics(image, face_info)` → symmetry, brightness, area_ratio

---

### FaceNet — Tạo Embedding
**File:** [app/core/face_recognizer.py](app/core/face_recognizer.py)

**Mô hình:** InceptionResnetV1 (từ facenet-pytorch)  
**Pretrained:** VGGFace2 (9,131 người, 3.31 triệu ảnh)  
**Input:** ảnh 160×160 RGB  
**Output:** vector 512 chiều (embedding)

**Preprocessing pipeline:**
```
Ảnh gốc → Resize(160×160) → ToTensor() → Normalize([0.5, 0.5, 0.5])
```

**Phương thức chính:**
- `get_embedding(face_image)` → numpy array (512,)
- `get_embedding_from_raw_image(image, detector)` → detect + embed trong 1 bước
- `batch_get_embeddings(face_images)` → xử lý nhiều ảnh
- `get_model_info()` → thông tin mô hình, device, số tham số

**Device:** Tự động chọn GPU nếu có (CUDA), ngược lại dùng CPU

---

## Services & Business Logic

### EnrollmentService
**File:** [app/services/enrollment_service.py](app/services/enrollment_service.py)

Điều phối toàn bộ quy trình đăng ký khuôn mặt: từ decode ảnh → phát hiện mặt → tạo embedding → phát hiện bất thường → trả về kết quả.

### VerificationService
**File:** [app/services/verification_service.py](app/services/verification_service.py)

Điều phối quy trình xác thực: decode ảnh → detect mặt → tạo embedding → so sánh với known_embeddings → tính confidence → kiểm tra ngưỡng.

### AnomalyDetector
**File:** [app/services/anomaly_detector.py](app/services/anomaly_detector.py)

Phát hiện các bất thường trong ảnh và kết quả xác thực:

| Loại bất thường | Mức độ | Điều kiện kích hoạt |
|----------------|--------|---------------------|
| NO_FACE | HIGH | Không phát hiện mặt |
| MULTIPLE_FACES | HIGH | Phát hiện > 1 mặt |
| LOW_CONFIDENCE | MEDIUM | face_confidence < 0.95 |
| LOW_QUALITY | MEDIUM | quality_score < 0.7 |
| FACE_TOO_SMALL | HIGH | area_ratio < 0.05 |
| FACE_MISMATCH | CRITICAL | Kết quả verify = false |
| SUSPICIOUS_PATTERN | MEDIUM | Khớp nhưng confidence < 80% |

---

## Utilities

### ImageProcessor
**File:** [app/utils/image_processor.py](app/utils/image_processor.py)

- `decode_base64_image(base64_string)` → numpy array RGB (hỗ trợ data URI)
- `validate_image_dimensions(image)` → kiểm tra min/max width và height
- `calculate_image_quality_score(image)` → float [0.0–1.0]
  - Brightness variance: trọng số 0.3
  - Sharpness (Laplacian): trọng số 0.4
  - Contrast (std deviation): trọng số 0.3
- `resize_image(image, target_size)` → dùng LANCZOS interpolation
- `normalize_image(image)` → normalize về [-1, 1] theo công thức `(pixel - 127.5) / 127.5`

### DistanceCalculator
**File:** [app/utils/distance_calculator.py](app/utils/distance_calculator.py)

- `euclidean_distance(emb1, emb2)` → float, ngưỡng: < 1.0 = cùng người
- `cosine_similarity(emb1, emb2)` → float [0–1], ngưỡng: > 0.6 = cùng người
- `is_same_person(emb1, emb2)` → `{is_match, euclidean_distance, cosine_similarity, confidence, method}`
- `find_best_match(query_embedding, known_embeddings)` → tìm embedding khớp nhất trong danh sách

### Logger
**File:** [app/utils/logger.py](app/utils/logger.py)

Dùng **loguru** với cấu hình:
- Console: màu sắc, format rõ ràng
- File: `logs/face_service.log`
- Xoay vòng: mỗi 10 MB
- Giữ lại: 30 ngày
- Nén: ZIP cho file cũ

---

## Request/Response Models

### Request Models (`app/models/request.py`)

**FaceEnrollRequest**
| Field | Type | Ràng buộc |
|-------|------|-----------|
| employee_id | int | > 0 |
| image_base64 | str | min_length=100, định dạng base64 |
| is_primary | bool | default=True |
| note | str (optional) | max_length=500 |

**FaceVerifyRequest**
| Field | Type | Ràng buộc |
|-------|------|-----------|
| employee_id | int | > 0 |
| image_base64 | str | min_length=100 |
| known_embeddings | List[List[float]] | min 1 embedding |
| verification_threshold | float (optional) | 0–100 |

### Response Models (`app/models/response.py`)

**FaceEnrollResponse** — embedding + metadata + anomalies  
**FaceVerifyResponse** — is_match + confidence + distances + anomalies  
**HealthCheckResponse** — status + model_info  
**ErrorResponse** — error type + message + timestamp

---

## Khởi Động & Lifespan

**File:** [app/main.py](app/main.py)

**Startup sequence:**
1. Log thông tin service và cấu hình
2. Khởi tạo `FaceDetector` (load MTCNN)
3. Khởi tạo `FaceRecognizer` (tải weights FaceNet ~100MB lần đầu)
4. Lưu vào `app.state` để inject vào routes
5. Log sẵn sàng

**Middleware:**
- CORS: cho phép các origin trong `ALLOWED_ORIGINS`
- Exception handler toàn cục: trả 500 với chi tiết (DEBUG) hoặc thông báo chung (production)

**Docs tự động:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Tích Hợp với Spring Boot Backend

Service này là một thành phần trong kiến trúc microservice của MetaHRM. Spring Boot backend chịu trách nhiệm:
- Lưu trữ embedding vào database
- Gọi `/api/face/enroll` khi nhân viên đăng ký
- Truy xuất embedding đã lưu rồi gọi `/api/face/verify` khi chấm công
- Ghi nhận kết quả chấm công và anomaly

**Luồng Enrollment (Spring → Face Service):**
```
Frontend upload ảnh → Spring nhận → encode base64 → gọi /api/face/enroll
→ nhận embedding 512 chiều → lưu vào bảng employee_faces
```

**Luồng Verification (Spring → Face Service):**
```
Nhân viên check-in → Spring lấy known_embeddings từ DB
→ gọi /api/face/verify với ảnh + embeddings
→ nhận {is_match, confidence} → ghi nhận chấm công hoặc anomaly
```

---

## Hiệu Năng

| Chỉ số | Giá trị ước tính |
|--------|-----------------|
| Thời gian Enrollment | 300–500ms |
| Thời gian Verification | 300–500ms |
| Decode ảnh | ~50ms |
| Phát hiện mặt (MTCNN) | ~100–150ms |
| Tạo embedding (FaceNet) | ~150–200ms |
| Tính khoảng cách | < 10ms |
| Độ chính xác (VGGFace2) | ~99.5% |
| False Accept Rate | < 0.1% |
| False Reject Rate | < 1% |
| RAM cho models | ~150–200MB |

**GPU:** Tự động phát hiện CUDA. Nếu có GPU, inference nhanh hơn 2–3x.

---

## Docker Deployment

**Dockerfile:** Multi-stage build

- Stage builder: `python:3.10-slim` + gcc/g++ để build packages
- Stage runtime: `python:3.10-slim` + `libgl1` + `libglib2.0-0` (yêu cầu của OpenCV)
- Health check: `GET /health` mỗi 30 giây
- Port: 8000

**docker-compose.yml:**
```yaml
services:
  face-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./weights:/app/weights
      - ./data:/app/data
    restart: unless-stopped
```

**Chạy Docker:**
```bash
docker-compose up -d
docker logs metahrm-face-service
curl http://localhost:8000/health
```

---

## Cài Đặt Local

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Cài dependencies
pip install -r requirements.txt

# Copy và chỉnh .env
copy .env.example .env

# Chạy server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Yêu cầu hệ thống:**
- Python 3.10+
- RAM tối thiểu 4GB (khuyến nghị 8GB)
- GPU NVIDIA với CUDA (tuỳ chọn, tăng tốc 2–3x)

---

## Testing

```bash
# Chạy toàn bộ test
pytest tests/ -v -s

# Chỉ chạy integration test
pytest tests/test_api_integration.py -v

# Test một module cụ thể
pytest tests/test_distance_calculator.py -v
```

**Postman Collection:** `tests/MetaHRM_FaceService.postman_collection.json`  
Bao gồm tất cả endpoints với test scripts và biến môi trường cấu hình sẵn.

---

## Hạn Chế Hiện Tại

1. **Chỉ xử lý 1 khuôn mặt:** Từ chối ảnh có nhiều người (by design)
2. **Chỉ nhận base64:** Chưa có API upload file trực tiếp
3. **Không phát hiện giả mạo (liveness):** Chưa chống replay attack bằng ảnh/video
4. **Không lưu database:** Embedding do Spring Boot quản lý hoàn toàn
5. **Không batch API:** Enrollment/verification từng ảnh một

## Cải Tiến Đề Xuất

1. Thêm liveness detection (phát hiện chớp mắt, xoay đầu)
2. Hỗ trợ nhận diện mặt nghiêng/quay
3. Thêm API batch enrollment
4. Thêm authentication (API key, JWT)
5. Thêm rate limiting
6. Thêm Prometheus metrics endpoint
7. Thêm embedding caching cho verification lặp lại
8. Thêm phân tích texture để chống spoofing

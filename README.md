# 🤖 MetaHRM Face Recognition Service

Face Recognition microservice cho hệ thống chấm công MetaHRM sử dụng FaceNet và MTCNN.

## 📋 Mục lục

- [Tính năng](#tính-năng)
- [Tech Stack](#tech-stack)
- [Cài đặt](#cài-đặt)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Performance](#performance)

---

## ✨ Tính năng

- ✅ **Face Enrollment**: Đăng ký khuôn mặt nhân viên
- ✅ **Face Verification**: Xác thực khuôn mặt khi chấm công
- ✅ **Anomaly Detection**: Phát hiện bất thường (nhiều mặt, chất lượng thấp...)
- ✅ **RESTful API**: FastAPI với Swagger UI
- ✅ **Docker Support**: Containerized deployment
- ✅ **High Accuracy**: ~99.5% trên VGGFace2 dataset

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI 0.104+ |
| **Face Detection** | MTCNN |
| **Face Recognition** | FaceNet (Inception-ResNet-V1) |
| **Pretrained Model** | VGGFace2 (9K identities, 3.31M images) |
| **Embedding Dimension** | 512-D vector |
| **Distance Metrics** | Euclidean Distance + Cosine Similarity |
| **Python** | 3.10+ |

---

## 🚀 Cài đặt

### Prerequisites

- Python 3.10+
- pip
- virtualenv (recommended)

### Installation Steps

```bash
# 1. Clone repository
git clone <repo-url>
cd metahrm-face-service

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env

# 5. Run server
python -m uvicorn app.main:app --reload

# Server chạy tại http://localhost:8000+

# Swagger UI: http://localhost:8000/docs
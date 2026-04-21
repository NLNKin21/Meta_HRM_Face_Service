"""
Tạo test images từ webcam hoặc download samples
"""

import cv2
import base64
from pathlib import Path
import requests


def capture_from_webcam(num_photos=3):
    """Chụp ảnh từ webcam"""
    output_dir = Path("data/sample_faces")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("📸 Press SPACE to capture, ESC to exit")
    
    count = 0
    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Capture Face - Press SPACE', frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            filename = output_dir / f"my_face_{count+1}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"✅ Saved: {filename}")
            count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Captured {count} photos")


def download_sample_images():
    """Download sample celebrity faces"""
    output_dir = Path("data/sample_faces")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    urls = {
        "obama": "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg",
        "einstein": "https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg",
    }
    
    for name, url in urls.items():
        try:
            response = requests.get(url)
            if response.status_code == 200:
                output_path = output_dir / f"{name}.jpg"
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded: {output_path}")
        except Exception as e:
            print(f"❌ Failed {name}: {e}")


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


if __name__ == "__main__":
    print("1. Capture from webcam")
    print("2. Download sample images")
    choice = input("Choose (1/2): ")
    
    if choice == "1":
        capture_from_webcam(3)
    else:
        download_sample_images()
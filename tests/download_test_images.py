"""
Download sample images để test
"""
import requests
from pathlib import Path

def download_test_images():
    """Download một vài ảnh từ internet"""
    
    # Sample face images (free to use)
    urls = [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
    ]
    
    output_dir = Path("data/sample_faces")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, url in enumerate(urls):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                output_path = output_dir / f"test_{idx+1}.jpg"
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded: {output_path}")
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")

if __name__ == "__main__":
    download_test_images()
import os
import json
import base64
import requests
from dotenv import load_dotenv

from src.utils.path import OUT_DIR, DATA_DIR


load_dotenv()


VL_OUT_DIR = OUT_DIR / "ocr"
VL_OUT_DIR.mkdir(exist_ok=True)

BASE_URL = os.getenv("OCR_BASE_URL", "http://localhost:8080")

image_path = DATA_DIR / "test.jpeg"

with open(image_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {
    "file": file_data,
    "fileType": 1,
    "returnWordBox": True,
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    # "textDetThresh": 0.02,
    # "textDetBoxThresh": 0.01,
    # "textDetUnclipRatio": 1.5,
    # "textDetLimitType": "max",
    # "textDetLimitSideLen": 5000,
    # "textRecScoreThresh": 0.0,
}

response = requests.post(f"{BASE_URL}/ocr", json=payload)

assert response.status_code == 200
result = response.json()["result"]
ocr_json_path = VL_OUT_DIR / "ocr_result.json"
ocr_json_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8"
)
print(f"OCR result saved at {ocr_json_path}")
for i, res in enumerate(result["ocrResults"]):
    # print(res["prunedResult"])
    ocr_img_path = VL_OUT_DIR / f"ocr_{i}.jpg"
    with open(ocr_img_path, "wb") as f:
        f.write(base64.b64decode(res["ocrImage"]))
    print(f"Output image saved at {ocr_img_path}")


import os
import base64
import requests
import pathlib
from dotenv import load_dotenv

from src.utils.path import OUT_DIR, DATA_DIR


load_dotenv()

VL_OUT_DIR = OUT_DIR / "vl"
VL_OUT_DIR.mkdir(exist_ok=True)

BASE_URL = os.getenv("OCR_BASE_URL", "http://localhost:8080")

image_path = DATA_DIR / "test.jpeg"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data, # Base64编码的文件内容或者文件URL
    "fileType": 1, # 文件类型，1表示图像文件
}

response = requests.post(BASE_URL + "/layout-parsing", json=payload)
assert response.status_code == 200, (response.status_code, response.text)

result = response.json()["result"]
pages = []
for i, res in enumerate(result["layoutParsingResults"]):
    pages.append({"prunedResult": res["prunedResult"], "markdownImages": res["markdown"].get("images")})
    for img_name, img in res["outputImages"].items():
        img_path = VL_OUT_DIR / f"{img_name}_{i}.jpg"
        pathlib.Path(img_path).parent.mkdir(exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")

payload = {
    "pages": pages,
    "concatenatePages": True,
}

response = requests.post(BASE_URL + "/restructure-pages", json=payload)
assert response.status_code == 200, (response.status_code, response.text)

result = response.json()["result"]
res = result["layoutParsingResults"][0]
print(res["prunedResult"])
md_dir = VL_OUT_DIR / "markdown"
md_dir.mkdir(exist_ok=True)
(md_dir / "doc.md").write_text(res["markdown"]["text"], encoding="utf-8")
for img_path, img in res["markdown"]["images"].items():
    img_path = md_dir / img_path
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(base64.b64decode(img))
print(f"Markdown document saved at {md_dir / 'doc.md'}")
from paddleocr import LayoutDetection
from paddlex.utils.fonts import PINGFANG_FONT

from src.utils.path import DATA_DIR, MODELS_DIR, OUT_DIR, FONT_DIR


PINGFANG_FONT._local_path = str(FONT_DIR / 'PingFang-SC-Regular.ttf')

LAYOUT_DIR = OUT_DIR / 'layout'
LAYOUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def get_images():
    """Get all images in the data directory."""
    return [
        p
        for p in DATA_DIR.rglob("*")
        if p.suffix.lower() in IMG_EXTS and p.is_file()
    ]


layout_detector = LayoutDetection(
    model_name='PP-DocLayout_plus-L',
    model_dir=str(MODELS_DIR / 'PP-DocLayout_plus-L'),
    device="cpu",
    enable_mkldnn=False,
)


def main():
    img_files = get_images()
    for img_file in img_files:
        output = layout_detector.predict(str(img_file), batch_size=1, layout_nms=True)
        for res in output:
            res.print()
            res.save_to_img(save_path=str(LAYOUT_DIR / img_file.name))
            res.save_to_json(save_path=str(LAYOUT_DIR / f"{img_file.stem}.json"))


if __name__ == "__main__":
    main()

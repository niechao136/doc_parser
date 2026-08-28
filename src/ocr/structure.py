from paddleocr import PPStructureV3
from paddlex.utils.fonts import PINGFANG_FONT

from src.utils.path import DATA_DIR, MODELS_DIR, OUT_DIR, FONT_DIR


PINGFANG_FONT._local_path = str(FONT_DIR / 'PingFang-SC-Regular.ttf')


STRUCTURE_DIR = OUT_DIR / 'structure'
STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)


pipeline = PPStructureV3(
    # 版面检测：复用你已经在用的模型
    layout_detection_model_dir=str(MODELS_DIR / "PP-DocLayout_plus-L"),
    layout_detection_model_name="PP-DocLayout_plus-L",

    # 表格识别：这是你需要的核心功能，打开
    use_table_recognition=True,
    doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",  # PP-LCNet_x1_0_doc_ori
    doc_orientation_classify_model_dir=str(MODELS_DIR / "PP-LCNet_x1_0_doc_ori"),
    textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
    textline_orientation_model_dir=str(MODELS_DIR / "PP-LCNet_x1_0_textline_ori"),
    table_classification_model_dir=str(MODELS_DIR / "PP-LCNet_x1_0_table_cls"),
    table_classification_model_name="PP-LCNet_x1_0_table_cls",
    wired_table_structure_recognition_model_dir=str(MODELS_DIR / "SLANeXt_wired"),
    wired_table_structure_recognition_model_name="SLANeXt_wired",
    wireless_table_structure_recognition_model_dir=str(MODELS_DIR / "SLANet_plus"),
    wireless_table_structure_recognition_model_name="SLANet_plus",
    wired_table_cells_detection_model_dir=str(MODELS_DIR / "RT-DETR-L_wired_table_cell_det"),
    wired_table_cells_detection_model_name="RT-DETR-L_wired_table_cell_det",
    wireless_table_cells_detection_model_dir=str(MODELS_DIR / "RT-DETR-L_wireless_table_cell_det"),
    wireless_table_cells_detection_model_name="RT-DETR-L_wireless_table_cell_det",

    # OCR 文字检测识别：表单里肯定要用
    text_detection_model_dir=str(MODELS_DIR / "PP-OCRv6_medium_det"),
    text_detection_model_name="PP-OCRv6_medium_det",
    text_recognition_model_dir=str(MODELS_DIR / "PP-OCRv6_medium_rec"),
    text_recognition_model_name="PP-OCRv6_medium_rec",

    # 用不到的模块显式关闭，省去加载和推理时间
    use_doc_orientation_classify=False,   # 如果你的扫描件本来就是正的，不需要方向分类
    use_doc_unwarping=False,              # 不需要做去畸变（UVDoc），除非是拍照件有透视变形
    use_textline_orientation=False,
    use_seal_recognition=False,           # 不涉及印章
    use_formula_recognition=False,        # 不涉及公式
    use_chart_recognition=False,          # 不涉及图表转表格
    use_region_detection=False,           # 一般表单不需要额外的 region 检测层

    device="cpu",
    lang="ch",
    enable_mkldnn=False,
)


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def get_images():
    """Get all images in the data directory."""
    return [
        p
        for p in DATA_DIR.rglob("*")
        if p.suffix.lower() in IMG_EXTS and p.is_file()
    ]


def main():
    img_files = get_images()
    for img_file in img_files:
        output = pipeline.predict(str(img_file), return_word_box=True)
        for res in output:
            res.print()
            # 每张原图对应一个子目录，避免多页/多图之间互相覆盖
            img_out_dir = STRUCTURE_DIR / img_file.stem
            img_out_dir.mkdir(parents=True, exist_ok=True)
            res.save_to_img(save_path=str(img_out_dir))
            res.save_to_json(save_path=str(img_out_dir / f"{img_file.stem}.json"))


if __name__ == "__main__":
    main()


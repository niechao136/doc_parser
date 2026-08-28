import json
import time
from pathlib import Path
from paddleocr import PaddleOCR

from src.utils.path import MODELS_DIR, DATA_DIR, OUT_DIR


model = "PP-OCRv6"

def build_ocr_model(tier: str = "medium") -> PaddleOCR:
    """Build the PaddleOCR model with specified configurations."""
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        enable_mkldnn=False,
        text_detection_model_name=f"{model}_{tier}_det",
        text_detection_model_dir=str(MODELS_DIR / f"{model}_{tier}_det"),
        text_recognition_model_name=f"{model}_{tier}_rec",
        text_recognition_model_dir=str(MODELS_DIR / f"{model}_{tier}_rec"),
    )

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
    enable_mkldnn=False,
    text_detection_model_name="PP-OCRv6_medium_det",
    text_detection_model_dir=str(MODELS_DIR / "PP-OCRv6_medium_det"),
    text_recognition_model_name="PP-OCRv6_medium_rec",
    text_recognition_model_dir=str(MODELS_DIR / "PP-OCRv6_medium_rec"),
)


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def get_images():
    """Get all images in the data directory."""
    return [
        p
        for p in DATA_DIR.rglob("*")
        if p.suffix.lower() in IMG_EXTS and p.is_file()
    ]


def run_single_tier(img_files: list[Path], tier: str = "medium"):
    print(f"\n{'=' * 50}")
    print(f"档位: {tier} |  图片数: {len(img_files)}")
    print(f"{'=' * 50}")

    t_init_start = time.perf_counter()
    t_init = time.perf_counter() - t_init_start
    print(f"模型加载耗时: {t_init:.2f}s")

    tier_out_dir = OUT_DIR / tier
    tier_out_dir.mkdir(parents=True, exist_ok=True)

    per_image_records = []
    total_start = time.perf_counter()
    ocr_model = build_ocr_model(tier)

    for idx, img_path in enumerate(img_files, 1):
        t0 = time.perf_counter()
        try:
            result = ocr_model.predict(str(img_path), return_word_box=True)
            elapsed = time.perf_counter() - t0

            # 统计识别到的文本行数，并汇总所有文字（便于快速人工核对效果）
            texts = []
            for res in result:
                res.save_to_img(str(tier_out_dir))
                res.save_to_json(str(tier_out_dir))
                # 不同版本字段名可能是 rec_texts 或类似结构，做个兼容取值
                res_dict = res.json if hasattr(res, "json") else {}
                texts.extend(res_dict.get("rec_texts", []) if isinstance(res_dict, dict) else [])

            status = "ok"
        except Exception as e:
            elapsed = time.perf_counter() - t0
            texts = []
            status = f"error: {e}"

        print(f"[{idx}/{len(img_files)}] {img_path.name}: {elapsed:.3f}s  ({status})")

        per_image_records.append(
            {
                "file": img_path.name,
                "elapsed_sec": round(elapsed, 4),
                "status": status,
                "text_line_count": len(texts),
                "texts_preview": texts[:5],  # 只存前5行做速览，避免json过大
            }
        )

    total_elapsed = time.perf_counter() - total_start
    ok_records = [r for r in per_image_records if r["status"] == "ok"]
    avg_elapsed = sum(r["elapsed_sec"] for r in ok_records) / len(ok_records) if ok_records else 0

    summary = {
        "tier": tier,
        "model_init_sec": round(t_init, 3),
        "image_count": len(img_files),
        "success_count": len(ok_records),
        "fail_count": len(img_files) - len(ok_records),
        "total_inference_sec": round(total_elapsed, 3),
        "avg_per_image_sec": round(avg_elapsed, 4),
        "images": per_image_records,
    }

    report_path = tier_out_dir / f"report_{tier}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[{tier}] 汇总: 成功 {summary['success_count']}/{summary['image_count']}，"
        f"总耗时 {summary['total_inference_sec']}s，平均每张 {summary['avg_per_image_sec']}s")
    print(f"[{tier}] 详细报告已保存: {report_path}")

    return summary


def main():
    img_files = get_images()

    all_summaries = []
    summary = run_single_tier(img_files, tier="medium")
    all_summaries.append(summary)

    if len(all_summaries) > 1:
        print(f"\n{'=' * 50}")
        print("模型速度对比")
        print(f"{'=' * 50}")
        print(f"{'档位':<10}{'加载耗时(s)':<15}{'平均每张(s)':<15}{'总耗时(s)':<12}")
        for s in all_summaries:
            print(f"{s['tier']:<10}{s['model_init_sec']:<15}{s['avg_per_image_sec']:<15}{s['total_inference_sec']:<12}")

        compare_path = OUT_DIR / "compare_report.json"
        with open(compare_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"\n对比报告已保存: {compare_path}")

    print(f"\n全部完成，结果目录: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    print("Hello from doc-parser!")
    main()


"""
根据「表单栏位提取结果」(fields.json) 和「OCR 结果（含字符级坐标）」(ocr_result.json)，
计算每个栏位在原图上应该显示的坐标：

- 非选择类栏位（Text / Number / Date ...）：记录 x, y, height
    y / height 直接取该栏位标题所在 OCR 行的 box
    x 通常取标题冒号右边界再留出 10px 间隙；特殊内嵌横线栏位取横线起点

- 关系栏位如果表单已有默认值，position 仍是填写起点，另记录 overwritePosition
    作为清除默认文字的区域

- 选择类栏位（SingleChoice / MultiChoice）：为每个选项记录 x, y, width, height
    优先使用选项文字前面紧邻的 □ / ○ 符号的坐标
    如果 OCR 没有识别出符号（漏检/看漏），则用选项第一个字的框，
    按字符高度估算一个等大的方框位置，往左推算出来
"""

import difflib
import json
import opencc

from src.utils.path import OUT_DIR


CHECKBOX_GLYPHS = {"□", "○", "◯", "☐"}


FIELD_DIR = OUT_DIR / "field"
FIELD_PATH = FIELD_DIR / "field.json"
OCR_RESULT_PATH = OUT_DIR / "medium" / "test_res.json"


s2t = opencc.OpenCC('s2t.json')


FIELD_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "wound_care_details": (
        "給予術後護理指導(見衛教單：",
        "給予街後護理指導(見衛教單：",
    ),
    "tube_specific_info": ("管路照護方法（見衛教單：",),
}


OPTION_ALIASES: dict[str, tuple[str, ...]] = {
    "外傭": ("外備",),
    "流質": ("流臂",),
    "低鈉": ("低釣",),
    "每天換藥": ("每天换藥次",),
    "換藥方式": ("○换藥方式：(見衛教單)",),
    "有引流管": ("□有引流管條",),
    "緊急就醫狀況": (
        "緊急就罂狀況",
        "緊急就罂狀況：将殊獠烧(>38℃)、急性疼痛、痛口红腫、不正常出血、",
    ),
    "給予預約回診單": ("□給予預約回診單",),
}


DATE_FIELD_LAYOUTS: dict[str, tuple[tuple[str, int, int], ...]] = {
    "discharge_date": (
        ("year", 160, 64),
        ("month", 240, 30),
        ("day", 287, 39),
    ),
}


TEXT_FIELD_GAP = 10


INLINE_FIELD_LAYOUTS: dict[str, tuple[str, str]] = {
    "dressing_frequency": ("每天換藥", "次"),
    "drain_tube_count": ("有引流管", "條"),
}


DEFAULT_VALUE_FIELDS: dict[str, str] = {
    "relationship": "本人",
}


FIELD_SEARCH_BOUNDARIES: dict[str, tuple[str, str]] = {
    "living_status_detail": ("回家", "出院居住狀態"),
    "primary_caregiver": ("居家主要照顧者", "活動能力"),
    "institution_type": ("機構", "活動能力"),
    "activity_ability": ("活動能力", "飲食"),
    "mobility_aid_type": ("輔具使用指導", "飲食"),
    "dietary_requirements": ("飲食", "傷口"),
    "wound_care_instructions": ("傷口", "管路"),
    "tube_care": ("管路", "健康照護需求"),
    "health_care_needs": ("健康照護需求", "當你有任何問題時"),
}


def normalize(s: str) -> str:
    """去掉空格之类的干扰字符，并把常见异体字统一，方便模糊匹配。"""
    s = s2t.convert(s)
    return "".join(s.split()).casefold()


def fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _match_score(actual: str, target: str) -> float:
    prefix = actual[: len(target)]
    return max(fuzzy_ratio(actual, target), fuzzy_ratio(prefix, target))


def _find_colon_boundary(line: dict, start_idx: int) -> int | None:
    for token, box in zip(line["tokens"][start_idx:], line["token_boxes"][start_idx:]):
        token = token.strip()
        colon_idx = next(
            (index for index, char in enumerate(token) if char in {":", "："}),
            None,
        )
        if colon_idx is None:
            continue
        if colon_idx == len(token) - 1:
            return box[2]
        char_width = (box[2] - box[0]) / len(token)
        return round(box[0] + char_width * (colon_idx + 1))
    return None


def _find_default_value_start(
    line: dict, start_idx: int, default_value: str
) -> tuple[int, int] | None:
    default_value_n = normalize(default_value)
    value_start_idx = None

    for index in range(start_idx, len(line["tokens"])):
        token = line["tokens"][index]
        colon_idx = next(
            (offset for offset, char in enumerate(token) if char in {":", "："}),
            None,
        )
        if colon_idx is None:
            continue

        remainder_n = normalize(token[colon_idx + 1 :])
        if remainder_n.startswith(default_value_n):
            box = line["token_boxes"][index]
            char_width = (box[2] - box[0]) / len(token)
            value_start = round(box[0] + char_width * (colon_idx + 1))
            return value_start, box[2]
        value_start_idx = index + 1
        break

    if value_start_idx is None:
        return None

    for index in range(value_start_idx, len(line["tokens"])):
        value_n = ""
        for end in range(index, len(line["tokens"])):
            value_n += normalize(line["tokens"][end])
            if len(value_n) >= len(default_value_n):
                if value_n.startswith(default_value_n):
                    return (
                        line["token_boxes"][index][0],
                        line["token_boxes"][end][2],
                    )
                break
    return None


def _find_best_match(
    targets: tuple[str, ...],
    lines: list[dict],
    preferred_y: float | None = None,
):
    best = None
    best_key = None

    for target_idx, target in enumerate(targets):
        target_n = normalize(target)
        if not target_n:
            continue

        for line in lines:
            tokens = line["tokens"]
            for start in range(len(tokens)):
                acc = ""
                for end in range(start, len(tokens)):
                    acc += tokens[end]
                    acc_n = normalize(acc)
                    if len(acc_n) >= len(target_n):
                        score = _match_score(acc_n, target_n)
                        candidate_key = (
                            score,
                            len(target_n),
                            (
                                -abs(line["box"][1] - preferred_y)
                                if preferred_y is not None
                                else 0
                            ),
                            -target_idx,
                            -line["line_idx"],
                            -start,
                        )
                        if best_key is None or candidate_key > best_key:
                            best_key = candidate_key
                            best = (score, line, start, end)
                        break

    return best


def flatten_ocr(ocr_data: dict) -> list[dict]:
    """把 OCR 结果整理成便于查找的结构：每一行的整体文本/坐标框，
    以及行内逐字符（token）的文本和坐标。"""
    lines = []
    for i, text in enumerate(ocr_data["rec_texts"]):
        lines.append(
            {
                "line_idx": i,
                "text": text,
                "box": ocr_data["rec_boxes"][i],  # [x0, y0, x1, y1]
                "tokens": ocr_data["text_word"][i],
                "token_boxes": ocr_data["text_word_boxes"][i],
            }
        )
    return lines


def find_label_position(
    label: str,
    lines: list[dict],
    min_ratio: float = 0.6,
    aliases: tuple[str, ...] | None = None,
):
    """
    在所有 OCR 行里，找一段 token 序列跟 label 最匹配的位置。
    返回 (line, end_token_idx)。
    """
    targets = (label, *(aliases or ()))
    best = _find_best_match(targets, lines)
    if best is None or best[0] < min_ratio:
        return None
    return best[1], best[3]


def compute_text_field_position(
    label: str,
    lines: list[dict],
    aliases: tuple[str, ...] | None = None,
    field_key: str | None = None,
):
    found = find_label_position(label, lines, aliases=aliases)
    if not found:
        return None
    line, end_idx = found
    x = _find_colon_boundary(line, end_idx)
    if x is None:
        x = line["token_boxes"][end_idx][2]
    x += TEXT_FIELD_GAP
    y0, y1 = line["box"][1], line["box"][3]
    return {"x": x, "y": y0, "height": y1 - y0}


def compute_default_value_overwrite_position(
    field_key: str, label: str, lines: list[dict]
) -> dict[str, int] | None:
    default_value = DEFAULT_VALUE_FIELDS.get(field_key)
    if default_value is None:
        return None

    found = find_label_position(label, lines)
    if not found:
        return None

    line, end_idx = found
    value_bounds = _find_default_value_start(line, end_idx, default_value)
    if value_bounds is None:
        return None

    x0, x1 = value_bounds
    y0, y1 = line["box"][1], line["box"][3]
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def compute_inline_field_position(
    field_key: str, lines: list[dict]
) -> dict[str, int] | None:
    layout = INLINE_FIELD_LAYOUTS.get(field_key)
    if layout is None:
        return None

    prefix, suffix = layout
    suffix_n = normalize(suffix)
    for line in lines:
        prefix_match = _find_best_match((prefix,), [line])
        if prefix_match is None or prefix_match[0] < 0.8:
            continue

        prefix_end_idx = prefix_match[3]
        for suffix_idx in range(prefix_end_idx + 1, len(line["tokens"])):
            if normalize(line["tokens"][suffix_idx]).startswith(suffix_n):
                x = line["token_boxes"][suffix_idx - 1][2]
                y0, y1 = line["box"][1], line["box"][3]
                return {"x": x, "y": y0, "height": y1 - y0}
    return None


def compute_date_field_positions(
    field_key: str, label: str, lines: list[dict]
) -> dict[str, dict[str, int]] | None:
    layout = DATE_FIELD_LAYOUTS.get(field_key)
    if layout is None:
        return None

    found = find_label_position(label, lines)
    if not found:
        return None

    line, _ = found
    y0, y1 = line["box"][1], line["box"][3]
    return {
        part_name: {
            "x": x,
            "y": y0,
            "width": width,
            "height": y1 - y0,
        }
        for part_name, x, width in layout
    }


def find_option_position(
    option_text: str,
    lines: list[dict],
    min_ratio: float = 0.55,
    y_range: tuple[float, float] | None = None,
    aliases: tuple[str, ...] | None = None,
    preferred_y: float | None = None,
):
    """
    在（可选的 y_range 范围内的）行里搜索这个选项文字，返回它前面 checkbox 符号的坐标；
    找不到符号时，按选项首字的框往左推一个估算框。

    y_range: (y_start, y_end)，限制只在这个纵向区间内的 OCR 行里查找，
            用来避免"其他："这类会在表单里反复出现的选项文字被错误地
            匹配到别的栏位上。不传则不限制（全局搜索）。
    """
    candidate_lines = lines
    if y_range is not None:
        y_start, y_end = y_range
        candidate_lines = [
            line for line in lines if y_start <= line["box"][1] < y_end
        ]

    targets = (option_text, *(aliases or ()))
    best = _find_best_match(
        targets, candidate_lines, preferred_y=preferred_y
    )
    if best is None or best[0] < min_ratio:
        return None

    _, line, start_idx, _ = best
    first_box = line["token_boxes"][start_idx]

    checkbox_idx = None
    if line["tokens"][start_idx].strip() in CHECKBOX_GLYPHS:
        checkbox_idx = start_idx
    elif (
        start_idx - 1 >= 0
        and line["tokens"][start_idx - 1].strip() in CHECKBOX_GLYPHS
    ):
        checkbox_idx = start_idx - 1

    if checkbox_idx is not None:
        cb_box = line["token_boxes"][checkbox_idx]
        return {
            "x": cb_box[0],
            "y": cb_box[1],
            "width": cb_box[2] - cb_box[0],
            "height": cb_box[3] - cb_box[1],
            # "inferred": False,
        }

    # 没有符号：按第一个字的框高度估算一个等大方框，放在它左边
    char_h = first_box[3] - first_box[1]
    box_size = char_h
    return {
        "x": first_box[0] - box_size,
        "y": first_box[1],
        "width": box_size,
        "height": char_h,
        # "inferred": True,
    }


def compute_field_anchors(fields: list[dict], lines: list[dict]) -> list[float | None]:
    """
    尝试用每个栏位自己的 label 在 OCR 里找到对应行，记录该行的 y0 作为"锚点"。
    有些栏位（比如内部拆分出来的子栏位）标题本身并没有印在表单上，
    找不到就是 None，后面用相邻栏位的锚点插值。
    """
    anchors = []
    for field in fields:
        aliases = FIELD_LABEL_ALIASES.get(field["fieldKey"])
        found = find_label_position(
            field["label"], lines, min_ratio=0.6, aliases=aliases
        )
        anchors.append(found[0]["box"][1] if found else None)
    return anchors


def get_field_search_window(
    field_key: str, default_window: tuple[float, float], lines: list[dict]
) -> tuple[float, float]:
    boundaries = FIELD_SEARCH_BOUNDARIES.get(field_key)
    if boundaries is None:
        return default_window

    start_found = find_label_position(boundaries[0], lines, min_ratio=0.8)
    end_found = find_label_position(boundaries[1], lines, min_ratio=0.8)
    if start_found is None or end_found is None:
        return default_window

    start_y = start_found[0]["box"][1]
    end_y = end_found[0]["box"][1]
    if start_y >= end_y:
        return default_window
    return start_y, end_y


def fill_search_windows(
    anchors: list[float | None], bottom: float, back_margin: float = 40
) -> list[tuple[float, float]]:
    """
    把 anchors 里的 None 用前后最近的真实锚点插值，
    给每个栏位算出一个 (y_start, y_end) 搜索区间：
    - y_start：自己的锚点往前留一点余量（back_margin），
    因为有些栏位是合并单元格，选项可能印在标题所在行的上方
    - y_end：往后找下一个"明显不同"的锚点（跳过跟自己几乎同一行的相邻栏位锚点）

    这是一个基于 OCR 行 y 坐标的启发式方法，不是真正按表格单元格边界切分，
    遇到复杂的合并单元格布局时可能仍需要人工微调 back_margin，
    或者改用 PPStructureV3 的 cell_box_list 精确切分（见函数说明）。
    """
    n = len(anchors)
    windows = []
    for i in range(n):
        # 起点：自身锚点，否则往前找最近一个有锚点的栏位
        y_start = anchors[i]
        j = i
        while y_start is None and j > 0:
            j -= 1
            y_start = anchors[j]
        if y_start is None:
            y_start = 0

        # 终点：往后找下一个明显不同（差距超过 back_margin）的锚点，
        # 避免因为相邻栏位标题刚好印在同一行而把窗口挤没
        y_end = bottom
        for k in range(i + 1, n):
            char = anchors[k]
            if char is not None:
                if char > y_start + back_margin / 2:
                    y_end = char
                    break
        windows.append((y_start - back_margin, y_end))
    return windows


def extract_field_positions(fields_data: dict, ocr_data: dict) -> list[dict]:
    lines = flatten_ocr(ocr_data)
    fields = fields_data["fields"]

    doc_bottom = max(line["box"][3] for line in lines)
    anchors = compute_field_anchors(fields, lines)
    windows = fill_search_windows(anchors, doc_bottom)

    results = []
    for field, window, anchor in zip(fields, windows, anchors):
        field_type = field["type"]
        label = field["label"]
        entry = {
            "fieldKey": field["fieldKey"],
            "label": label,
            "type": field_type,
        }

        if field_type in ("SingleChoice", "MultiChoice"):
            options = field.get("options") or []
            option_positions = {}
            search_window = get_field_search_window(
                field["fieldKey"], window, lines
            )
            for opt in options:
                pos = find_option_position(
                    opt,
                    lines,
                    y_range=search_window,
                    aliases=OPTION_ALIASES.get(opt),
                    preferred_y=anchor,
                )
                option_positions[opt] = pos  # 找不到时记 None，方便发现问题
            entry["options"] = option_positions
        else:
            if field_type == "Date":
                date_positions = compute_date_field_positions(
                    field["fieldKey"], label, lines
                )
                if date_positions:
                    first_position = next(iter(date_positions.values()))
                    entry["position"] = {
                        key: first_position[key] for key in ("x", "y", "height")
                    }
                    entry["positions"] = date_positions
                else:
                    entry["position"] = compute_text_field_position(
                        label,
                        lines,
                        aliases=FIELD_LABEL_ALIASES.get(field["fieldKey"]),
                        field_key=field["fieldKey"],
                    )
            else:
                entry["position"] = compute_inline_field_position(
                    field["fieldKey"], lines
                ) or compute_text_field_position(
                    label,
                    lines,
                    aliases=FIELD_LABEL_ALIASES.get(field["fieldKey"]),
                    field_key=field["fieldKey"],
                )

                overwrite_position = compute_default_value_overwrite_position(
                    field["fieldKey"], label, lines
                )
                if overwrite_position is not None:
                    entry["overwritePosition"] = overwrite_position

        results.append(entry)

    return results


def main():
    fields_data = json.loads(FIELD_PATH.read_text(encoding="utf-8"))
    ocr_data = json.loads(OCR_RESULT_PATH.read_text(encoding="utf-8"))

    positions = extract_field_positions(fields_data, ocr_data)

    out_path = FIELD_DIR / "field_positions.json"
    out_path.write_text(
        json.dumps(positions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"写入 {out_path}，共 {len(positions)} 个栏位")


if __name__ == "__main__":
    main()
"""
根据「表单栏位提取结果」(fields.json) 和「OCR 结果（含字符级坐标）」(ocr_result.json)，
计算每个栏位在原图上应该显示的坐标。

本版本相较上一版的主要变化（标准化解析流程）：
- 引入 ValueRegionType：不再按字段业务类型（Text/Number/Date/...）决定用哪种算法，
    而是按"定位这块区域要依赖什么锚点信号"分类（冒号 / checkbox 符号 / 异常间距 / ...）。
    分类本身是纯几何判断，不调用 LLM，保证"归到哪一类"是确定性的、可复现的；
    不确定性只留在"类型内部具体选哪个候选"这一层（模糊匹配 -> LLM 兜底）。
- 每个字段 / 每个选项的最终结果都附带 confidence（high/medium/low/unresolved）、
    method（用了哪种算法/是否升级过 LLM）、needsReview（是否建议人工复核），
    而不是只有一个坐标或 None。
- 解析过程中打印详细日志：锚点匹配方式与分数、区域类型判定、每种类型的处理结果、
    每个选项的定位过程，最后打印一份按 (region_type, confidence) 分组的汇总统计。

使用前需要根据你实际的本地 vLLM / Gemma4 部署方式，调整"LLM 客户端配置"部分的
base_url / model，以及 `_call_llm` 里 `with_structured_output` 的调用方式。
"""

import json
import os
import difflib
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Type, TypeVar

import opencc
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.utils.path import OUT_DIR

load_dotenv()


CHECKBOX_GLYPHS = {"□", "○", "◯", "☐"}
DATE_CHAR_MARKERS = {"年", "月", "日"}

FIELD_DIR = OUT_DIR / "field"
FIELD_PATH = FIELD_DIR / "field_116.json"
OCR_RESULT_PATH = OUT_DIR / "ocr" / "ocr_result.json"

TEXT_FIELD_GAP = 10  # 冒号右边留白
GAP_ANOMALY_SEARCH_WINDOW = 6  # 异常间距只在 label 结束后这么多个 token 内搜索
GAP_ANOMALY_MIN_SAMPLES = 3      # 至少要有这么多个正常字间距样本，统计才可靠
GAP_ANOMALY_MAD_K = 3.5          # 修正 z-score 的经典离群阈值
GAP_ANOMALY_FLOOR_RATIO = 0.8    # 无论如何，异常间距至少要有这么宽（相对行高）才考虑
GAP_ANOMALY_FALLBACK_RATIO = 1.6 # 样本不足、无法做统计时，退化用的绝对倍数阈值
GAP_ANOMALY_STRONG_MARGIN = 1.3  # strength 超过 MAD_K 的这个倍数，才算"很有把握"

s2t = opencc.OpenCC("s2t.json")


# ------------------------- LLM 客户端配置 -------------------------

_LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
_LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4")
_LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
_LLM_MAX_CONTEXT_TOKENS = int(os.environ.get("LLM_MAX_CONTEXT_TOKENS", "8192"))
_LLM_OUTPUT_RESERVE_TOKENS = 512

MAX_CANDIDATES = int(os.environ.get("LLM_MAX_CANDIDATES", "5"))
MAX_DATE_CONTEXT_LINES = int(os.environ.get("LLM_DATE_CONTEXT_LINES", "6"))
COMPOSITE_DATE_INPUT_OFFSET_RATIO = 0.5
COMPOSITE_DATE_SEPARATOR_WIDTH_RATIO = 0.3
COMPOSITE_DATE_SEPARATOR_GAP_RATIO = 0.1

LABEL_MATCH_CONFIDENCE = 0.85
LABEL_MATCH_MIN_CONSIDER = 0.35

OPTION_MATCH_CONFIDENCE = 0.85
OPTION_MATCH_MIN_CONSIDER = 0.4
OPTION_AMBIGUOUS_GAP = 0.08
LABEL_AMBIGUOUS_GAP = 0.08
OPTION_SECTION_GAP_TOLERANCE_RATIO = 0.25
OPTION_SECTION_MIN_GAP_TOLERANCE = 2
LEFT_INLINE_INPUT_OFFSET_RATIO = 0.45
CONTROL_REFERENCE_MAX_TEXT_DISTANCE_RATIO = 3
CONTROL_REFERENCE_MIN_TEXT_DISTANCE = 32
CONTROL_ALIGNMENT_MIN_TEXT_DISTANCE_RATIO = 1.5
MAX_OPTION_CONTEXT_TOKENS = 8

_llm_client = ChatOpenAI(
    base_url=_LLM_BASE_URL,
    api_key=_LLM_API_KEY,  # type: ignore
    model=_LLM_MODEL,
    temperature=0,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
        "skip_reasoning": True,
        "skip_special_tokens": True,
    },
)


class LabelMatchResult(BaseModel):
    matched_idx: int | None = Field(description="所选候选行下标（从0开始）；都不匹配则为 null")


class OptionMatchResult(BaseModel):
    matched_idx: int | None = Field(description="真正属于该栏位的候选下标；都不属于则为 null")


class OptionGroupMatchResult(BaseModel):
    matched_indices: list[int | None] = Field(description="按目标选项顺序返回候选下标；无匹配为 null")


class DefaultValueResult(BaseModel):
    has_default_value: bool = Field(description="冒号后的文字是否是需要清除的预填默认答案")


class InlineMarkersResult(BaseModel):
    candidate_idx: int | None = Field(description="所选候选空白区编号；没有可填写区则为 null")


class DateMarkersResult(BaseModel):
    year_marker_idx: int | None = Field(description="“年”字所在 token 编号，找不到为 null")
    month_marker_idx: int | None = Field(description="“月”字所在 token 编号，找不到为 null")
    day_marker_idx: int | None = Field(description="“日”字所在 token 编号，找不到为 null")
    year_input_idx: int | None = Field(default=None, description="年输入区候选编号，没有则为 null")
    month_input_idx: int | None = Field(default=None, description="月输入区候选编号，没有则为 null")
    day_input_idx: int | None = Field(default=None, description="日输入区候选编号，没有则为 null")


def _estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 1.2))


def _check_budget(system_prompt: str, user_prompt: str) -> None:
    total = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
    budget = _LLM_MAX_CONTEXT_TOKENS - _LLM_OUTPUT_RESERVE_TOKENS
    if total > budget:
        raise ValueError(
            f"LLM 输入预估 {total} tokens，超过预算 {budget}"
            f"（上下文上限 {_LLM_MAX_CONTEXT_TOKENS}）。"
            "请调小 LLM_MAX_CANDIDATES 或缩短候选上下文后重试。"
        )


T = TypeVar("T", bound=BaseModel)


def _call_llm(system_prompt: str, user_prompt: str, schema: Type[T]) -> T | None:
    field_names = list(schema.model_fields)
    system_with_format = (
        f"{system_prompt}\n"
        "只返回一个合法 JSON 对象，不要输出 Markdown、解释文字或额外字段。"
        f"返回对象必须直接包含字段：{'、'.join(field_names)}。"
    )
    try:
        _check_budget(system_with_format, user_prompt)
        structured_llm = _llm_client.model_copy(update={"disable_streaming": True}).with_structured_output(
            schema, method="json_mode", include_raw=False,
        )
        raw = structured_llm.invoke(
            [
                {"role": "system", "content": system_with_format},
                {"role": "user", "content": user_prompt},
            ]
        )
        if isinstance(raw, schema):
            return raw
        return schema.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - 打印后按"未匹配"处理，方便排查
        print(f"    [llm] 调用失败，本次判定按未匹配处理：{exc}")
        return None


# ------------------------- 标准化：值域类型 & 解析结果 -------------------------


class ValueRegionType(str, Enum):
    """字段"该往哪填值"这件事，按定位所依赖的锚点信号分类，
    而不是按字段的业务类型分类——业务类型和定位算法并不是一一对应的。"""

    COLON_ANCHORED = "colon_anchored"                  # label 后紧跟冒号
    GLYPH_OPTION = "glyph_option"                        # SingleChoice/MultiChoice 的 checkbox 选项
    GLYPH_BEFORE_LABEL = "glyph_before_label"            # Boolean 字段，checkbox 紧邻在 label 前
    GLYPH_DATE_PART = "glyph_date_part"                  # 日期分段（斜杠 或 年/月/日 字符）
    GAP_ANOMALY = "gap_anomaly"                          # 无冒号无符号，但有明显异常宽的字间距
    INLINE_BEFORE_LABEL = "inline_before_label"          # 标签右侧有分隔符，输入区在标签左侧
    GLYPH_MISSING_ESTIMATE = "glyph_missing_estimate"    # 以上锚点都没找到，靠 LLM/估算兜底
    OVERWRITE_REGION = "overwrite_region"                # 冒号后已有文字，需判断是否为待清除默认值


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRESOLVED = "unresolved"


@dataclass
class PositionResult:
    position: dict | None
    confidence: Confidence
    method: str
    needs_review: bool = False
    extra: dict | None = None  # 额外信息，例如日期的三段 positions

    def to_dict(self) -> dict:
        result = {
            "position": self.position,
            "confidence": self.confidence.value,
            "method": self.method,
            "needsReview": self.needs_review,
        }
        if self.extra:
            result.update(self.extra)
        return result


# ------------------------- 调试打印 -------------------------

_STATS: Counter = Counter()  # (region_type, confidence) -> count
_INDENT = "    "


def _log(msg: str, indent: int = 1) -> None:
    print(f"{_INDENT * indent}{msg}")


def _log_header(field: dict) -> None:
    print(f"\n[{field.get('order', '?'):>3}] {field.get('key')} ({field.get('type')}) label={field.get('label')!r}")


def _record_stat(region_type: ValueRegionType, confidence: Confidence) -> None:
    _STATS[(region_type, confidence)] += 1


def print_summary() -> None:
    print("\n===== 汇总统计（按 region_type / confidence 分组）=====")
    if not _STATS:
        print("  (无数据)")
        return
    by_type: dict[ValueRegionType, Counter] = {}
    for (region_type, confidence), count in _STATS.items():
        by_type.setdefault(region_type, Counter())[confidence] += count
    for region_type, counter in by_type.items():
        total = sum(counter.values())
        breakdown = ", ".join(f"{c.value}={n}" for c, n in counter.items())
        print(f"  {region_type.value:<24} total={total:<4} ({breakdown})")


# ------------------------- 通用文本处理 -------------------------


def normalize(s: str) -> str:
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
        colon_idx = next((index for index, char in enumerate(token) if char in {":", "："}), None)
        if colon_idx is None:
            continue
        if colon_idx == len(token) - 1:
            return box[2]
        char_width = (box[2] - box[0]) / len(token)
        return round(box[0] + char_width * (colon_idx + 1))
    return None


def _line_remainder_after_colon(line: dict, start_idx: int) -> dict | None:
    tokens = line["tokens"]
    for i in range(start_idx, len(tokens)):
        token = tokens[i]
        colon_idx = next((k for k, ch in enumerate(token) if ch in {":", "："}), None)
        if colon_idx is None:
            continue

        box = line["token_boxes"][i]
        if colon_idx < len(token) - 1:
            char_width = (box[2] - box[0]) / len(token)
            x0 = round(box[0] + char_width * (colon_idx + 1))
            remainder_start = i
        else:
            x0 = box[2]
            remainder_start = i + 1

        remainder_tokens = tokens[remainder_start:]
        remainder_boxes = line["token_boxes"][remainder_start:]
        if not remainder_tokens:
            return None
        return {"text": "".join(remainder_tokens), "x0": x0, "x1": remainder_boxes[-1][2]}
    return None


def _find_top_candidates(
    targets: tuple[str, ...],
    lines: list[dict],
    top_k: int = MAX_CANDIDATES,
    preferred_y: float | None = None,
):
    scored = []
    for target in targets:
        target_n = normalize(target)
        if not target_n:
            continue
        for line in lines:
            tokens = line["tokens"]
            if not tokens:
                continue
            for start in range(len(tokens)):
                acc = ""
                for end in range(start, len(tokens)):
                    acc += tokens[end]
                    acc_n = normalize(acc)
                    if len(acc_n) >= len(target_n):
                        score = _match_score(acc_n, target_n)
                        scored.append((score, line, start, end))
                        break
                else:
                    if acc_n:
                        scored.append((_match_score(acc_n, target_n), line, start, len(tokens) - 1))

    if not scored:
        return []

    def sort_key(item):
        score, line, _, _ = item
        y_bonus = -abs(line["box"][1] - preferred_y) if preferred_y is not None else 0
        return (score, y_bonus, -line["line_idx"])

    scored.sort(key=sort_key, reverse=True)

    seen_lines = set()
    deduped = []
    for candidate in scored:
        line_idx = candidate[1]["line_idx"]
        if line_idx in seen_lines:
            continue
        seen_lines.add(line_idx)
        deduped.append(candidate)
        if len(deduped) >= top_k:
            break
    return deduped


def _option_match_score(actual: str, target: str) -> float:
    actual_n = normalize(actual)
    actual_without_controls = "".join(char for char in actual_n if char not in CHECKBOX_GLYPHS)
    prefix = actual_without_controls[: len(target)]
    return max(fuzzy_ratio(actual_without_controls, target), fuzzy_ratio(prefix, target))


def _find_option_candidates(
    option_text: str,
    lines: list[dict],
    top_k: int = MAX_CANDIDATES,
    preferred_y: float | None = None,
):
    target_n = normalize(option_text)
    if not target_n:
        return []

    scored = []
    max_span = max(MAX_OPTION_CONTEXT_TOKENS, len(target_n) + 4)
    for line in lines:
        tokens = line["tokens"]
        segments = []
        segment_start = 0
        for token_idx, token in enumerate(tokens):
            if token.strip() not in CHECKBOX_GLYPHS:
                continue
            if segment_start < token_idx:
                segments.append((segment_start, token_idx))
            segment_start = token_idx + 1
        if segment_start < len(tokens):
            segments.append((segment_start, len(tokens)))

        for segment_start, segment_end in segments:
            segment_candidate = None
            for start in range(segment_start, segment_end):
                best_for_start = None
                accumulated = ""
                max_end = min(segment_end, start + max_span)
                for end in range(start, max_end):
                    accumulated += tokens[end]
                    score = _option_match_score(accumulated, target_n)
                    candidate = (score, line, start, end)
                    candidate_text = normalize(accumulated)
                    candidate_key = (
                        score
                        + (0.2 if start > 0 and tokens[start - 1].strip() in CHECKBOX_GLYPHS else 0)
                        + (0.2 if end == segment_end - 1 and len(candidate_text) >= len(target_n) else 0),
                        -abs(len(candidate_text) - len(target_n)),
                        end == segment_end - 1 and len(candidate_text) >= len(target_n),
                        -end,
                    )
                    if best_for_start is None or candidate_key > best_for_start[0]:
                        best_for_start = (candidate_key, candidate)

                if best_for_start is not None:
                    if segment_candidate is None or best_for_start[0] > segment_candidate[0]:
                        segment_candidate = best_for_start
            if segment_candidate is not None:
                scored.append(segment_candidate[1])

    def sort_key(item):
        score, line, start, end = item
        boundary_score = score
        if start > 0 and line["tokens"][start - 1].strip() in CHECKBOX_GLYPHS:
            boundary_score += 0.2
        if any(token.strip() in CHECKBOX_GLYPHS for token in line["tokens"][start + 1 : end + 1]):
            boundary_score -= 0.2
        y_bonus = -abs(line["box"][1] - preferred_y) if preferred_y is not None else 0
        return (boundary_score, y_bonus, -line["line_idx"], -start, -end)

    scored.sort(key=sort_key, reverse=True)
    seen_candidates = set()
    deduped = []
    for candidate in scored:
        candidate_key = (candidate[1]["line_idx"], candidate[2], candidate[3])
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)
        deduped.append(candidate)
        if len(deduped) >= top_k:
            break
    return deduped


def flatten_ocr(ocr_data: dict) -> list[dict]:
    lines = []
    ocrResults = ocr_data.get("ocrResults", [])
    ocrResult = ocrResults[0] if ocrResults else {}
    prunedResult = ocrResult.get("prunedResult", {})
    rec_texts = ocr_data.get("rec_texts", prunedResult.get("rec_texts", []))
    for i, text in enumerate(rec_texts):
        lines.append(
            {
                "line_idx": i,
                "text": text,
                "box": ocr_data.get("rec_boxes", prunedResult.get("rec_boxes", []))[i],
                "tokens": ocr_data.get("text_word", prunedResult.get("text_word", []))[i],
                "token_boxes": ocr_data.get("text_word_boxes", prunedResult.get("text_word_boxes", []))[i],
            }
        )
    return lines


# ------------------------- LLM 兜底裁决 -------------------------


def _field_context_text(field_context: dict | str | None) -> str:
    if field_context is None:
        return "unknown"
    if isinstance(field_context, str):
        return field_context

    parts = []
    for key in (
        "key", "label", "type", "order", "field_index",
        "anchor_line_idx", "anchor_y", "previous_anchor_y", "next_anchor_y",
    ):
        value = field_context.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    for relation in ("previous_fields", "next_fields"):
        rel_fields = field_context.get(relation) or []
        if rel_fields:
            parts.append(
                f"{relation}="
                + ", ".join(f"{item.get('order', '?')}:{item.get('key', '')}/{item.get('label', '')}" for item in rel_fields)
            )
    options = field_context.get("options") or []
    if options:
        parts.append("options=" + ", ".join(str(option) for option in options))
    return "; ".join(parts) or "unknown"


def _line_token_text(line: dict) -> str:
    return " ".join(f"[{index}]{token}" for index, token in enumerate(line["tokens"]))


def _llm_resolve_label(label: str, candidates: list, field_context: dict | str | None):
    lines_desc = "\n".join(
        f"{i}. line={candidate[1]['line_idx']} y={candidate[1]['box'][1]} "
        f"text={candidate[1]['text']!r} tokens={_line_token_text(candidate[1])}"
        for i, candidate in enumerate(candidates)
    )
    system = (
        "You resolve a form field label from OCR rows. OCR may contain recognition errors, "
        "duplicate labels, or labels embedded in longer rows. Use the field metadata and the "
        "candidate row context to choose the occurrence belonging to that field. Return null "
        "only when none belongs to the field."
    )
    user = (
        f"Field metadata: {_field_context_text(field_context)}\n"
        f"Target label: {label}\n"
        f"Candidate rows (index starts at 0):\n{lines_desc}"
    )
    result = _call_llm(system, user, LabelMatchResult)
    if result is None or result.matched_idx is None:
        return None
    if 0 <= result.matched_idx < len(candidates):
        return result.matched_idx
    return None


def _llm_resolve_option_group(
    options: list[str], candidates: list, field_context: dict | str | None, lines: list[dict],
) -> list[int | None] | None:
    by_line_idx = {line["line_idx"]: line for line in lines}
    candidate_descriptions = []
    for candidate_idx, (score, line, start_idx, end_idx) in enumerate(candidates):
        preceding_token = line["tokens"][start_idx - 1].strip() if start_idx > 0 else None
        following_token = line["tokens"][end_idx + 1].strip() if end_idx + 1 < len(line["tokens"]) else None
        before = by_line_idx.get(line["line_idx"] - 1)
        after = by_line_idx.get(line["line_idx"] + 1)
        candidate_descriptions.append(
            f"{candidate_idx}. line={line['line_idx']} y={line['box'][1]} "
            f"text={line['text']!r} selected={''.join(line['tokens'][start_idx:end_idx + 1])!r} "
            f"range=[{start_idx}]-[{end_idx}] score={score:.3f} "
            f"control_before={preceding_token in CHECKBOX_GLYPHS} "
            f"control_after={following_token in CHECKBOX_GLYPHS} "
            f"before_row={before['text'] if before else ''!r} "
            f"after_row={after['text'] if after else ''!r}"
        )

    option_descriptions = "\n".join(f"{index}. {option}" for index, option in enumerate(options))
    system = (
        "You resolve all options belonging to one form field from OCR. OCR may contain typos, "
        "omitted control glyphs, and several options on one row. Assign each target option to "
        "the candidate occurrence belonging to this field. Use the field metadata, option order, "
        "anchor row, line number, selected token range, control-boundary flags, and neighboring "
        "rows. Do not reuse one physical candidate for different target options unless the OCR "
        "clearly shows that they are the same occurrence. Return one candidate index per target "
        "option, in the same order; use null only when no candidate belongs to that option."
    )
    user = (
        f"Field metadata: {_field_context_text(field_context)}\n"
        f"Target options (index starts at 0):\n{option_descriptions}\n"
        f"Candidate occurrences (index starts at 0):\n{chr(10).join(candidate_descriptions)}"
    )
    result = _call_llm(system, user, OptionGroupMatchResult)
    if result is None or len(result.matched_indices) != len(options):
        return None
    return [index if index is None or 0 <= index < len(candidates) else None for index in result.matched_indices]


def _llm_detect_default_value(label: str, remainder_text: str) -> DefaultValueResult | None:
    system = (
        "你在协助解析医院表单。栏位标题冒号后面如果印着一段文字，"
        "可能是表单预先印好、需要在填写时清除的默认答案，也可能只是说明文字。"
        "请判断这段文字是否属于需要被覆盖清除的默认答案。"
    )
    user = f"栏位标题：{label}\n冒号后文字：{remainder_text}"
    return _call_llm(system, user, DefaultValueResult)


def _llm_find_inline_markers(
    label: str, numbered_snippet: str, field_type: str | None = None, field_context: dict | str | None = None,
) -> InlineMarkersResult | None:
    system = (
        "You locate a form field input area from OCR. OCR may omit underlines, boxes, circles, "
        "or some text. Return only JSON with one field, candidate_idx. Select one listed blank "
        "candidate. If exactly one candidate is listed, return candidate_idx=0. For a Number or "
        "Text field, a writable area can be inside the matched label span, especially immediately "
        "before a unit token. For a Number field, prefer an inside-label candidate before a unit "
        "token over a virtual gap before the label. For a Text field whose matched label starts at token 0, prefer a "
        "positive-width leading candidate when one exists; trailing punctuation or unit text is "
        "context, not the input area. Do not choose a control gap before the label for Number "
        "fields. For a Boolean field, choose a positive-width control gap immediately before the "
        "label when present. Return null only when no candidate belongs to the field."
    )
    user = (
        f"Field label: {label}\n"
        + (f"Field type: {field_type}\n" if field_type else "")
        + (f"Field metadata: {_field_context_text(field_context)}\n" if field_context else "")
        + f"Candidate context:\n{numbered_snippet}"
    )
    return _call_llm(system, user, InlineMarkersResult)


def _find_inline_gap_candidates(line: dict, label_start_idx: int | None, label_end_idx: int) -> list[dict]:
    token_boxes = line["token_boxes"]
    if not token_boxes:
        return []

    gaps = []

    def add_candidate(before_idx, after_idx, gap_start, gap_end):
        if gap_end <= gap_start:
            return
        relation = "unknown"
        if label_start_idx is not None and after_idx is not None and after_idx <= label_start_idx:
            relation = "before-label"
        elif before_idx is not None and before_idx >= label_end_idx:
            relation = "after-label"
        elif label_start_idx is not None:
            relation = "inside-label"
        gaps.append({"before_idx": before_idx, "after_idx": after_idx, "x0": gap_start, "x1": gap_end, "relation": relation, "source": "ocr"})

    line_start = line["box"][0]
    first_start = token_boxes[0][0]
    line_height = max(box[3] - box[1] for box in token_boxes)
    leading_gap = first_start - line_start
    if leading_gap < line_height * 0.5 and line["tokens"][0].strip() not in CHECKBOX_GLYPHS:
        add_candidate(None, 0, first_start - line_height, first_start)
    else:
        add_candidate(None, 0, line_start, first_start)

    for index in range(1, len(token_boxes)):
        gap_start = token_boxes[index - 1][2]
        gap_end = token_boxes[index][0]
        add_candidate(index - 1, index, gap_start, gap_end)

    last_end = token_boxes[-1][2]
    line_end = line["box"][2]
    add_candidate(len(token_boxes) - 1, None, last_end, line_end)

    return gaps


def _describe_inline_gaps(gaps: list[dict]) -> str:
    def endpoint(token_idx, boundary):
        if token_idx is None:
            return boundary
        return f"[{token_idx}]"

    return "\n".join(
        f"candidate {index}: {endpoint(candidate.get('before_idx'), 'line-start')} -> "
        f"{endpoint(candidate.get('after_idx'), 'line-end')} "
        f"relation={candidate['relation']} source={candidate.get('source', 'ocr')} "
        f"width={candidate['x1'] - candidate['x0']:g}"
        for index, candidate in enumerate(gaps)
    )


def _filter_inline_gap_candidates(line: dict, field_type: str | None, gaps: list[dict]) -> list[dict]:
    def touches_control(candidate: dict) -> bool:
        for token_idx in (candidate["before_idx"], candidate["after_idx"]):
            if token_idx is not None and line["tokens"][token_idx].strip() in CHECKBOX_GLYPHS:
                return True
        return False

    if field_type == "Boolean":
        before_label = [candidate for candidate in gaps if candidate["relation"] == "before-label"]
        if before_label:
            return before_label

    if field_type == "Number":
        inside_label = [candidate for candidate in gaps if candidate["relation"] == "inside-label"]
        if inside_label:
            return inside_label

    without_controls = [candidate for candidate in gaps if not touches_control(candidate)]
    return without_controls or gaps


def _infer_label_start_idx(label: str, line: dict, end_idx: int) -> int | None:
    target_n = normalize(label)
    if not target_n or not line["tokens"] or not (0 <= end_idx < len(line["tokens"])):
        return None

    scored = []
    for start_idx in range(end_idx + 1):
        actual_n = normalize("".join(line["tokens"][start_idx : end_idx + 1]))
        if actual_n:
            scored.append((_match_score(actual_n, target_n), -start_idx, start_idx))

    if not scored:
        return None
    return max(scored)[2]


def _llm_find_date_markers(label: str, numbered_snippet: str) -> DateMarkersResult | None:
    system = (
        "You resolve date markers from OCR text. Find the token indexes for 年, 月, and 日. "
        "The context also lists input candidates generated only from OCR token geometry. "
        "For each marker, return the candidate index immediately before that marker when one "
        "belongs to the date input; otherwise return null for that input field. If the context "
        "does not contain a 年/月/日 date layout, return null marker indexes."
    )
    user = f"Field label: {label}\nDate context:\n{numbered_snippet}"
    return _call_llm(system, user, DateMarkersResult)


def _token_char_boxes(token: str, box: list[int]) -> list[tuple[str, int, int]]:
    char_width = (box[2] - box[0]) / max(1, len(token))
    return [
        (char, round(box[0] + char_width * offset), round(box[0] + char_width * (offset + 1)))
        for offset, char in enumerate(token)
    ]


def _median(values: list[int]) -> int:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return round((ordered[middle - 1] + ordered[middle]) / 2)


def _estimate_date_glyph_width(references: list[tuple[dict, int]]) -> int:
    heights = [max(1, box[3] - box[1]) for line, token_idx in references for box in [line["token_boxes"][token_idx]]]
    narrow_widths = [
        box[2] - box[0]
        for line, token_idx in references
        for box in [line["token_boxes"][token_idx]]
        if box[2] - box[0] <= max(heights) * 1.5
    ]
    widths = narrow_widths or [line["token_boxes"][token_idx][2] - line["token_boxes"][token_idx][0] for line, token_idx in references]
    return max(1, _median(widths))


def _infer_date_content_left(lines: list[dict], target_line: dict, marker_start: int, glyph_width: int) -> int:
    nearby_starts = [target_line["box"][0]]
    nearby_starts.extend(
        candidate["box"][0]
        for candidate in lines
        if candidate is not target_line
        and abs(candidate["box"][1] - target_line["box"][1]) <= 160
        and marker_start - glyph_width * 6 <= candidate["box"][0] < marker_start
    )
    if not nearby_starts:
        return max(target_line["box"][0], marker_start - glyph_width)

    clusters = []
    for value in sorted(nearby_starts):
        if not clusters or value - clusters[-1][-1] > glyph_width:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    best_cluster = max(clusters, key=lambda cluster: (len(cluster), -abs(_median(cluster) - marker_start)))
    return _median(best_cluster)


def _find_date_input_candidates(lines: list[dict], references: list[tuple[dict, int]]) -> list[dict]:
    candidates = []
    glyph_width = _estimate_date_glyph_width(references)
    previous_marker_end = None
    for reference_idx, (line, token_idx) in enumerate(references):
        token = line["tokens"][token_idx]
        token_box = line["token_boxes"][token_idx]
        token_width = token_box[2] - token_box[0]
        for char, char_start, _ in _token_char_boxes(token, token_box):
            if char not in DATE_CHAR_MARKERS:
                continue

            marker_start = char_start
            if token_width > glyph_width * 1.5:
                marker_start = token_box[2] - glyph_width

            if token_idx > 0:
                input_start = line["token_boxes"][token_idx - 1][2]
            elif previous_marker_end is not None:
                line_start = line["box"][0]
                if marker_start - line_start <= glyph_width * 1.5:
                    input_start = previous_marker_end + max(1, glyph_width // 2)
                else:
                    input_start = max(line_start, previous_marker_end + max(1, glyph_width // 2))
            else:
                input_start = _infer_date_content_left(lines, line, marker_start, glyph_width)
            if input_start >= marker_start:
                input_start = marker_start - glyph_width
            candidates.append({"marker": char, "reference_idx": reference_idx, "x0": input_start, "x1": marker_start})
            previous_marker_end = marker_start + glyph_width
    return candidates


def _count_slashes_after_label(line: dict, end_idx: int) -> int:
    return sum(token.count("/") for token in line["tokens"][end_idx:])


def _has_date_char_markers(line: dict, end_idx: int) -> bool:
    return any(marker in token for token in line["tokens"][end_idx:] for marker in DATE_CHAR_MARKERS)


def _has_date_layout_marker(line: dict, start_idx: int = 0) -> bool:
    return any(
        "/" in token or any(marker in token for marker in DATE_CHAR_MARKERS)
        for token in line["tokens"][start_idx:]
    )


def _find_date_layout_block(line: dict, end_idx: int, lines: list[dict]) -> tuple[dict, int] | None:
    """如果 label 所在行本身没有日期版式（斜杠或年/月/日），
    在同一版面区域内往右找一个 y 有重叠、且包含日期版式的独立 OCR 块，
    返回 (那个块, 起始 token 下标)，供后续坐标计算直接复用，而不是只返回 bool。"""
    label_end = _find_colon_boundary(line, end_idx)
    if label_end is None:
        label_end = line["token_boxes"][end_idx][2] if 0 <= end_idx < len(line["token_boxes"]) else line["box"][0]

    anchor_y0, anchor_y1 = line["box"][1], line["box"][3]
    best = None
    for candidate in lines:
        if candidate is line or not candidate["tokens"] or candidate["box"][0] < label_end:
            continue
        overlap = min(anchor_y1, candidate["box"][3]) - max(anchor_y0, candidate["box"][1])
        if overlap <= 0:
            continue
        if _has_date_layout_marker(candidate):
            distance = candidate["box"][0] - label_end  # 多个候选块时，取离 label 最近的那个
            if best is None or distance < best[1]:
                best = (candidate, distance)
    if best is None:
        return None
    return best[0], -1   # -1 表示"这个块没有 label，不需要跳过任何 token"


def _has_date_layout_in_adjacent_blocks(line: dict, end_idx: int, lines: list[dict]) -> bool:
    label_end = _find_colon_boundary(line, end_idx)
    if label_end is None:
        label_end = line["token_boxes"][end_idx][2]

    anchor_y0, anchor_y1 = line["box"][1], line["box"][3]
    for candidate in lines:
        if candidate is line or not candidate["tokens"] or candidate["box"][0] < label_end:
            continue

        overlap = min(anchor_y1, candidate["box"][3]) - max(anchor_y0, candidate["box"][1])
        if overlap <= 0:
            continue

        if _has_date_layout_marker(candidate):
            return True
    return False


def _compute_slash_date_positions(line: dict, end_idx: int) -> dict[str, dict[str, int]] | None:
    slash_ranges = []
    composite_token_box = None
    scan_start = max(0, end_idx)   # 加这一行：-1 时从 0 开始，避免负数索引取到最后一个 token
    for token_idx in range(scan_start, len(line["tokens"])):
        token = line["tokens"][token_idx]
        token_box = line["token_boxes"][token_idx]
        has_colon = any(char in {":", "："} for char in token)
        if has_colon and "/" in token:
            composite_token_box = token_box
        for char, char_start, char_end in _token_char_boxes(token, token_box):
            if char == "/":
                if has_colon and token.count("/") == 1:
                    separator_width = max(
                        1,
                        round((token_box[3] - token_box[1]) * COMPOSITE_DATE_SEPARATOR_WIDTH_RATIO),
                    )
                    slash_ranges.append((token_box[2] - separator_width, token_box[2]))
                else:
                    slash_ranges.append((char_start, char_end))

    slash_count = len(slash_ranges)
    if slash_count not in (1, 2):
        return None

    if composite_token_box is not None:
        line_height = max(1, composite_token_box[3] - composite_token_box[1])
        input_start = composite_token_box[0] + max(
            1, round(line_height * COMPOSITE_DATE_INPUT_OFFSET_RATIO)
        )
    else:
        label_end = _find_colon_boundary(line, end_idx)
        if label_end is None:
            label_end = line["token_boxes"][end_idx][2] if 0 <= end_idx < len(line["token_boxes"]) else line["box"][0]
        input_start = label_end + TEXT_FIELD_GAP
    first_slash_start = slash_ranges[0][0]
    if input_start >= first_slash_start:
        input_start = max(line["box"][0], first_slash_start - max(1, line["box"][3] - line["box"][1]))

    boundaries = [input_start]
    for slash_start, slash_end in slash_ranges:
        boundaries.extend((slash_start, slash_end))
    boundaries.append(line["box"][2])

    if composite_token_box is not None:
        separator_gap = max(
            1,
            round((composite_token_box[3] - composite_token_box[1]) * COMPOSITE_DATE_SEPARATOR_GAP_RATIO),
        )
        boundaries[2] += separator_gap

    part_names = ("month", "day") if slash_count == 1 else ("year", "month", "day")
    positions = {}
    y0, y1 = line["box"][1], line["box"][3]
    for part_idx, part_name in enumerate(part_names):
        x0 = boundaries[part_idx * 2]
        x1 = boundaries[part_idx * 2 + 1]
        if x1 <= x0:
            x1 = x0 + max(1, y1 - y0)
        positions[part_name] = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
    return positions


def _build_date_context(lines: list[dict], line: dict, end_idx: int) -> tuple[str, list[tuple[dict, int]], list[dict]]:
    line_position = next((position for position, candidate in enumerate(lines) if candidate is line), None)
    if line_position is None:
        return "", [], []

    references = []
    context_lines = []
    for offset, context_line in enumerate(
        lines[line_position : line_position + max(1, MAX_DATE_CONTEXT_LINES)]
    ):
        token_start = end_idx + 1 if offset == 0 else 0
        if offset > 0 and not _has_date_layout_marker(context_line):
            break
        context_lines.append((context_line, token_start))

    for offset, context_line in enumerate(context_lines):
        context_line, token_start = context_line
        for token_idx in range(token_start, len(context_line["tokens"])):
            references.append((context_line, token_idx))

    numbered = "\n".join(
        f"[{index}] line={context_line['line_idx']} token={token_idx}: {context_line['tokens'][token_idx]}"
        for index, (context_line, token_idx) in enumerate(references)
    )
    input_candidates = _find_date_input_candidates(lines, references)
    if input_candidates:
        numbered += "\nInput candidates (index starts at 0):\n" + "\n".join(
            f"candidate {index}: marker={candidate['marker']} token_ref=[{candidate['reference_idx']}] x={candidate['x0']}..{candidate['x1']}"
            for index, candidate in enumerate(input_candidates)
        )
    return numbered, references, input_candidates


# ------------------------- 间距异常检测（gap_anomaly）-------------------------


def _typical_char_gap(line: dict) -> float:
    """用同一行里、非 checkbox 相邻 token 之间的间距中位数，估算这行印刷文字"正常"该有多宽的字间距。"""
    token_boxes, tokens = line["token_boxes"], line["tokens"]
    gaps = [
        token_boxes[i][0] - token_boxes[i - 1][2]
        for i in range(1, len(token_boxes))
        if tokens[i - 1].strip() not in CHECKBOX_GLYPHS and tokens[i].strip() not in CHECKBOX_GLYPHS
    ]
    gaps = [g for g in gaps if g >= 0]
    if not gaps:
        return (line["box"][3] - line["box"][1]) * 0.15
    return _median(gaps)


def _typical_char_gap_stats(line: dict) -> dict:
    """统计本行"正常"字间距的中心值和离散度。
    用中位数 + 中位绝对偏差(MAD)，比单纯"中位数的固定倍数"更稳——
    因为它衡量的是"离群程度"，不会因为本行字间距普遍很紧（中位数很小）
    就让阈值一起被压得很低。"""
    token_boxes, tokens = line["token_boxes"], line["tokens"]
    gaps = [
        token_boxes[i][0] - token_boxes[i - 1][2]
        for i in range(1, len(token_boxes))
        if tokens[i - 1].strip() not in CHECKBOX_GLYPHS and tokens[i].strip() not in CHECKBOX_GLYPHS
    ]
    gaps = [g for g in gaps if g >= 0]
    line_height = max(1, line["box"][3] - line["box"][1])

    if len(gaps) < GAP_ANOMALY_MIN_SAMPLES:
        return {"median": None, "mad": None, "line_height": line_height, "sample_size": len(gaps)}

    median = _median(gaps)
    mad = _median([abs(g - median) for g in gaps])
    return {"median": median, "mad": mad, "line_height": line_height, "sample_size": len(gaps)}


def _find_anomalous_gap_after_label(
    line: dict, end_idx: int,
    search_window: int = GAP_ANOMALY_SEARCH_WINDOW,
    boundary_idx: int | None = None,
) -> dict | None:
    """在 label 匹配结束位置之后一小段范围内，找明显宽于本行正常字间距的 gap
    （例如"每天換藥___次"里"藥"和"次"之间的留白）。"""
    stats = _typical_char_gap_stats(line)
    token_boxes, tokens = line["token_boxes"], line["tokens"]
    limit = min(len(token_boxes), end_idx + 1 + search_window)
    if boundary_idx is not None:
        limit = min(limit, boundary_idx)

    best = None
    for i in range(end_idx + 1, limit):
        if tokens[i - 1].strip() in CHECKBOX_GLYPHS or tokens[i].strip() in CHECKBOX_GLYPHS:
            continue
        gap = token_boxes[i][0] - token_boxes[i - 1][2]
        if gap < stats["line_height"] * GAP_ANOMALY_FLOOR_RATIO:
            continue  # 无论统计结果如何，太窄的间距直接排除，不够格当"填空"

        if stats["median"] is not None:
            mad = stats["mad"] or 1  # 避免除0（本行字间距完全一致时 MAD=0）
            z_score = 0.6745 * (gap - stats["median"]) / mad
            is_anomalous = z_score > GAP_ANOMALY_MAD_K
            strength = z_score
        else:
            # 样本太少统计不可靠，只能靠绝对下限判断，把握程度打个折扣
            is_anomalous = gap >= stats["line_height"] * GAP_ANOMALY_FALLBACK_RATIO
            strength = gap / stats["line_height"]

        if is_anomalous and (best is None or gap > best["width"]):
            best = {
                "x0": token_boxes[i - 1][2], "x1": token_boxes[i][0], "width": gap,
                "strength": strength, "sample_size": stats["sample_size"],
            }
    return best


def _find_post_label_separator(line: dict, end_idx: int) -> int | None:
    next_idx = end_idx + 1
    if next_idx >= len(line["tokens"]):
        return None
    token = line["tokens"][next_idx].strip()
    if token and token[0] in {",", "，", "、", ";", "；"}:
        return next_idx
    return None


def _find_nearby_checkbox_reference(
    target_line: dict, target_x: int, lines: list[dict],
) -> dict | None:
    target_height = max(1, target_line["box"][3] - target_line["box"][1])
    max_text_distance = max(
        CONTROL_REFERENCE_MIN_TEXT_DISTANCE,
        target_height * CONTROL_REFERENCE_MAX_TEXT_DISTANCE_RATIO,
    )
    references = []
    for line in lines:
        if abs(line["box"][1] - target_line["box"][1]) > 160:
            continue
        for token_idx, token in enumerate(line["tokens"]):
            if token.strip() not in CHECKBOX_GLYPHS or token_idx + 1 >= len(line["tokens"]):
                continue
            control_box = line["token_boxes"][token_idx]
            text_box = line["token_boxes"][token_idx + 1]
            text_distance = abs(text_box[0] - target_x)
            if text_distance <= max_text_distance:
                references.append(
                    (text_distance, abs(line["box"][1] - target_line["box"][1]), control_box, text_box)
                )

    if not references:
        return None
    _, _, control_box, text_box = min(references, key=lambda item: (item[0], item[1]))
    return {"control_box": control_box, "text_box": text_box}


def _has_left_inline_input_signal(
    label: str, line: dict, end_idx: int, lines: list[dict] | None,
) -> bool:
    separator_idx = _find_post_label_separator(line, end_idx)
    if separator_idx is None or separator_idx + 1 >= len(line["tokens"]):
        return False

    separator_box = line["token_boxes"][separator_idx]
    next_box = line["token_boxes"][separator_idx + 1]
    line_height = max(1, line["box"][3] - line["box"][1])
    gap_after_separator = next_box[0] - separator_box[2]
    if gap_after_separator < line_height * 1.25:
        return False

    label_start_idx = _infer_label_start_idx(label, line, end_idx)
    if label_start_idx is None:
        return False
    if lines is None:
        return True
    label_x = line["token_boxes"][label_start_idx][0]
    return _find_nearby_checkbox_reference(line, label_x, lines) is not None


def _find_glyph_immediately_before_label(line: dict, label_start_idx: int | None) -> dict | None:
    if label_start_idx is None or label_start_idx <= 0:
        return None
    prev_idx = label_start_idx - 1
    if line["tokens"][prev_idx].strip() not in CHECKBOX_GLYPHS:
        return None
    box = line["token_boxes"][prev_idx]
    return {"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]}


# ------------------------- 标签定位 -------------------------


def find_label_position(label: str, lines: list[dict], field_context: dict | str | None = None):
    """在所有 OCR 行里找一段 token 序列跟 label 最匹配的位置。
    返回 (line, end_token_idx, method, score)；method 标明是纯模糊匹配、
    y 范围约束后的精确匹配、还是升级调用了 LLM。分数够高直接采信，分数模糊时才调用 LLM 裁决。"""
    candidates = _find_top_candidates((label,), lines, top_k=MAX_CANDIDATES)
    if not candidates:
        return None

    used_bounded = False
    if isinstance(field_context, dict):
        previous_y = field_context.get("previous_anchor_y")
        next_y = field_context.get("next_anchor_y")
        if previous_y is not None and next_y is not None and previous_y < next_y:
            target_n = normalize(label)
            expanded_candidates = _find_top_candidates((label,), lines, top_k=len(lines))
            bounded_exact = [
                candidate
                for candidate in expanded_candidates
                if previous_y < candidate[1]["box"][1] < next_y
                and normalize("".join(candidate[1]["tokens"][candidate[2] : candidate[3] + 1])) == target_n
            ]
            if bounded_exact:
                candidates = bounded_exact
                used_bounded = True

    top_score = candidates[0][0]
    if top_score < LABEL_MATCH_MIN_CONSIDER:
        return None

    ambiguous = len(candidates) > 1 and (top_score - candidates[1][0]) < LABEL_AMBIGUOUS_GAP
    if top_score >= LABEL_MATCH_CONFIDENCE and not ambiguous:
        _, line, _, end = candidates[0]
        method = "bounded_exact" if used_bounded else "fuzzy_exact"
        return line, end, method, top_score

    chosen = _llm_resolve_label(label, candidates, field_context)
    if chosen is None:
        return None
    score, line, _, end = candidates[chosen]
    return line, end, "llm_resolved", score


def _find_option_control_idx(line: dict, start_idx: int) -> int | None:
    if line["tokens"][start_idx].strip() in CHECKBOX_GLYPHS:
        return start_idx
    if start_idx - 1 >= 0 and line["tokens"][start_idx - 1].strip() in CHECKBOX_GLYPHS:
        return start_idx - 1
    return None


def _extract_option_box(line: dict, start_idx: int) -> tuple[dict, str]:
    """返回 (坐标框, method)。method 区分"真的找到了 checkbox 符号"还是"符号缺失，靠估算兜底"。"""
    first_box = line["token_boxes"][start_idx]

    checkbox_idx = _find_option_control_idx(line, start_idx)

    if checkbox_idx is not None:
        cb_box = line["token_boxes"][checkbox_idx]
        return {"x": cb_box[0], "y": cb_box[1], "width": cb_box[2] - cb_box[0], "height": cb_box[3] - cb_box[1]}, "glyph_found"

    char_h = first_box[3] - first_box[1]
    return {"x": first_box[0] - char_h, "y": first_box[1], "width": char_h, "height": char_h}, "glyph_missing_estimate"


def _infer_option_control_size(reference_candidates: list) -> tuple[int, int] | None:
    control_boxes = []
    seen_controls = set()
    for candidate in reference_candidates:
        candidate_line = candidate[1]
        candidate_start_idx = candidate[2]
        control_idx = _find_option_control_idx(candidate_line, candidate_start_idx)
        if control_idx is None:
            continue
        identity = (candidate_line["line_idx"], control_idx)
        if identity in seen_controls:
            continue
        seen_controls.add(identity)
        control_boxes.append(candidate_line["token_boxes"][control_idx])

    if not control_boxes:
        return None
    return (
        _median([box[2] - box[0] for box in control_boxes]),
        _median([box[3] - box[1] for box in control_boxes]),
    )


def _align_missing_option_box(
    line: dict, start_idx: int, reference_candidates: list,
) -> dict | None:
    first_box = line["token_boxes"][start_idx]
    target_height = max(1, first_box[3] - first_box[1])
    inferred_size = _infer_option_control_size(reference_candidates)
    width, height = inferred_size or (target_height, target_height)
    max_text_distance = max(
        CONTROL_REFERENCE_MIN_TEXT_DISTANCE,
        target_height * CONTROL_REFERENCE_MAX_TEXT_DISTANCE_RATIO,
    )
    references = []
    for candidate in reference_candidates:
        candidate_line = candidate[1]
        candidate_start_idx = candidate[2]
        control_idx = _find_option_control_idx(candidate_line, candidate_start_idx)
        if control_idx is None or candidate_line["tokens"][control_idx].strip() not in CHECKBOX_GLYPHS:
            continue
        if candidate_line is line and candidate_start_idx == start_idx:
            continue
        text_idx = candidate_start_idx
        if candidate_line["tokens"][text_idx].strip() in CHECKBOX_GLYPHS:
            text_idx += 1
        if text_idx >= len(candidate_line["token_boxes"]):
            continue
        text_distance = abs(candidate_line["token_boxes"][text_idx][0] - first_box[0])
        if text_distance <= max_text_distance:
            references.append(
                (text_distance, abs(candidate_line["box"][1] - line["box"][1]), candidate_line["token_boxes"][control_idx])
            )

    if not references:
        return None
    text_distance, _, control_box = min(references, key=lambda item: (item[0], item[1]))
    if text_distance < target_height * CONTROL_ALIGNMENT_MIN_TEXT_DISTANCE_RATIO:
        return None
    return {
        "x": control_box[0],
        "y": first_box[1],
        "width": width,
        "height": height,
    }


def find_option_position(
    option_text: str, lines: list[dict],
    field_label: str | None = None, preferred_y: float | None = None,
    field_context: dict | str | None = None,
):
    """独立工具函数：搜索单个选项文字并返回坐标（不产出 confidence 元信息，
    主流程走的是下面的 _resolve_option_positions）。"""
    candidates = _find_option_candidates(option_text, lines, top_k=MAX_CANDIDATES, preferred_y=preferred_y)
    if not candidates:
        return None

    top_score = candidates[0][0]
    if top_score < OPTION_MATCH_MIN_CONSIDER:
        return None

    if field_context is None and field_label is not None:
        field_context = {"label": field_label}

    chosen_idx = 0
    candidate_text = normalize("".join(candidates[0][1]["tokens"][candidates[0][2] : candidates[0][3] + 1]))
    exact_match = candidate_text == normalize(option_text)
    ambiguous = len(candidates) > 1 and (candidates[0][0] - candidates[1][0]) < OPTION_AMBIGUOUS_GAP
    if top_score < OPTION_MATCH_CONFIDENCE or ambiguous or not exact_match:
        from_group = _llm_resolve_option_group([option_text], candidates, field_context, lines)
        if from_group and from_group[0] is not None:
            chosen_idx = from_group[0]

    _, line, start_idx, _ = candidates[chosen_idx]
    box, _method = _extract_option_box(line, start_idx)
    return box


def _infer_option_section_y_range(
    per_option_candidates: list[list],
) -> tuple[float, float] | None:
    """根据同一字段中唯一的高置信选项，推断选项所在的纵向区段。

    重复选项本身不能用于确定区段；例如多个“其他”候选都可能是满分，
    但其它选项通常能提供当前字段的共同 y 范围。
    """
    supporting_lines = []
    for candidates in per_option_candidates:
        strong_candidates = [
            candidate
            for candidate in candidates
            if candidate[0] >= OPTION_MATCH_CONFIDENCE
        ]
        if len(strong_candidates) == 1:
            supporting_lines.append(strong_candidates[0][1])

    if len(supporting_lines) < 2:
        return None

    y_values = [line["box"][1] for line in supporting_lines]
    line_height = max(
        max(1, line["box"][3] - line["box"][1])
        for line in supporting_lines
    )
    section_end = max(line["box"][3] for line in supporting_lines)
    gap_tolerance = max(
        OPTION_SECTION_MIN_GAP_TOLERANCE,
        round(line_height * OPTION_SECTION_GAP_TOLERANCE_RATIO),
    )
    return min(y_values) - line_height, section_end + gap_tolerance


def _collect_option_candidates(
    options: list[str],
    lines: list[dict],
    preferred_y: float | None,
) -> tuple[list[list], list]:
    initial_candidates = [
        _find_option_candidates(option, lines, top_k=MAX_CANDIDATES, preferred_y=preferred_y)
        for option in options
    ]
    section_y_range = _infer_option_section_y_range(initial_candidates)

    per_option = []
    pooled = []
    by_identity = {}
    for option, initial in zip(options, initial_candidates):
        candidates = initial
        if section_y_range is not None:
            y_start, y_end = section_y_range
            expanded = _find_option_candidates(
                option, lines, top_k=len(lines), preferred_y=preferred_y,
            )
            scoped = [
                candidate
                for candidate in expanded
                if y_start <= candidate[1]["box"][1] <= y_end
                and candidate[0] >= OPTION_MATCH_MIN_CONSIDER
            ]
            if scoped:
                candidates = scoped

        relevant = [candidate for candidate in candidates if candidate[0] >= OPTION_MATCH_MIN_CONSIDER] or candidates[:1]
        per_option.append(relevant)
        for candidate in relevant:
            identity = (candidate[1]["line_idx"], candidate[2], candidate[3])
            existing_idx = by_identity.get(identity)
            if existing_idx is None:
                by_identity[identity] = len(pooled)
                pooled.append(candidate)
            elif candidate[0] > pooled[existing_idx][0]:
                pooled[existing_idx] = candidate
    return per_option, pooled


def _resolve_option_positions(
    options: list[str], lines: list[dict], field_context: dict | str | None, preferred_y: float | None,
) -> dict[str, PositionResult]:
    per_option, pooled = _collect_option_candidates(
        options, lines, preferred_y,
    )
    if not pooled:
        _log("未在 OCR 中找到任何候选（所有选项均无法定位）", indent=2)
        return {option: PositionResult(None, Confidence.UNRESOLVED, "no_candidate", needs_review=True) for option in options}

    exact_flags = []
    for option, candidates in zip(options, per_option):
        if not candidates:
            exact_flags.append(False)
            continue
        exact = (
            len(candidates) == 1
            and candidates[0][0] >= OPTION_MATCH_CONFIDENCE
            and normalize("".join(candidates[0][1]["tokens"][candidates[0][2] : candidates[0][3] + 1])) == normalize(option)
        )
        exact_flags.append(exact)

    needs_group_resolution = not all(exact_flags)
    assignments = None
    if needs_group_resolution:
        ambiguous_count = sum(not f for f in exact_flags)
        _log(f"选项候选存在歧义/低分（{ambiguous_count}/{len(options)} 项），升级调用 LLM 整组裁决", indent=2)
        assignments = _llm_resolve_option_group(options, pooled, field_context, lines)
    else:
        _log("所有选项模糊匹配均命中且无歧义，跳过 LLM 调用", indent=2)

    result: dict[str, PositionResult] = {}
    used_candidates = set()
    for option_idx, option in enumerate(options):
        candidate = None
        method = "fuzzy_exact" if exact_flags[option_idx] else None
        assignment = assignments[option_idx] if assignments is not None else None
        allowed = {(item[1]["line_idx"], item[2], item[3]) for item in per_option[option_idx]}

        if assignment is not None and 0 <= assignment < len(pooled):
            assigned_candidate = pooled[assignment]
            identity = (assigned_candidate[1]["line_idx"], assigned_candidate[2], assigned_candidate[3])
            if identity in allowed and identity not in used_candidates:
                candidate = assigned_candidate
                method = "llm_resolved"

        if candidate is None:
            for fallback in per_option[option_idx]:
                identity = (fallback[1]["line_idx"], fallback[2], fallback[3])
                if identity not in used_candidates:
                    candidate = fallback
                    if method is None:
                        method = "fuzzy_fallback"
                    break

        if candidate is None:
            _log(f"选项 {option!r} -> 未找到可用候选", indent=2)
            result[option] = PositionResult(None, Confidence.UNRESOLVED, "no_candidate", needs_review=True)
            continue

        used_candidates.add((candidate[1]["line_idx"], candidate[2], candidate[3]))
        box, glyph_method = _extract_option_box(candidate[1], candidate[2])
        if glyph_method == "glyph_missing_estimate":
            aligned_box = _align_missing_option_box(candidate[1], candidate[2], pooled)
            if aligned_box is not None:
                box, glyph_method = aligned_box, "glyph_missing_aligned"
            else:
                inferred_size = _infer_option_control_size(pooled)
                if inferred_size is not None:
                    inferred_width, inferred_height = inferred_size
                    box["x"] += max(0, (box["width"] - inferred_width) // 2)
                    box["width"], box["height"] = inferred_width, inferred_height
                    glyph_method = "glyph_missing_size_inferred"
        score = candidate[0]

        if glyph_method in {
            "glyph_missing_estimate",
            "glyph_missing_aligned",
            "glyph_missing_size_inferred",
        }:
            confidence, needs_review, final_method = Confidence.LOW, True, glyph_method
        elif method == "fuzzy_exact":
            confidence, needs_review, final_method = Confidence.HIGH, False, "fuzzy_exact"
        elif method == "llm_resolved":
            confidence, needs_review, final_method = Confidence.MEDIUM, False, "llm_resolved"
        else:
            confidence, needs_review, final_method = Confidence.MEDIUM, True, method or "fuzzy_fallback"

        _log(
            f"选项 {option!r} -> line={candidate[1]['line_idx']} score={score:.3f} "
            f"method={final_method} conf={confidence.value} pos={box}",
            indent=2,
        )
        _record_stat(ValueRegionType.GLYPH_OPTION, confidence)
        result[option] = PositionResult(box, confidence, final_method, needs_review)

    return result


# ------------------------- 非选择类栏位定位 -------------------------


def compute_text_field_position(line: dict, end_idx: int) -> dict:
    x = _find_colon_boundary(line, end_idx)
    if x is None:
        x = line["token_boxes"][end_idx][2]
    x += TEXT_FIELD_GAP
    y0, y1 = line["box"][1], line["box"][3]
    return {"x": x, "y": y0, "height": y1 - y0}


def compute_default_value_overwrite_position(field: dict, line: dict, end_idx: int) -> dict | None:
    remainder = _line_remainder_after_colon(line, end_idx)
    if remainder is None or not remainder["text"].strip():
        return None

    result = _llm_detect_default_value(field["label"], remainder["text"])
    if result is None or not result.has_default_value:
        return None

    y0, y1 = line["box"][1], line["box"][3]
    return {"x": remainder["x0"], "y": y0, "width": remainder["x1"] - remainder["x0"], "height": y1 - y0}


def compute_inline_field_position(field: dict, line: dict, end_idx: int, field_context: dict | str | None = None) -> dict | None:
    tokens = line["tokens"]
    if not tokens:
        return None

    numbered = "OCR tokens: " + " ".join(f"[{index}]{token}" for index, token in enumerate(tokens))
    label_start_idx = _infer_label_start_idx(field["label"], line, end_idx)
    gap_candidates = _find_inline_gap_candidates(line, label_start_idx, end_idx)
    gap_candidates = _filter_inline_gap_candidates(line, field.get("type"), gap_candidates)
    if field.get("type") == "Text" and label_start_idx == 0:
        leading_candidates = [candidate for candidate in gap_candidates if candidate["relation"] == "before-label"]
        if leading_candidates:
            gap_candidates = leading_candidates
    gap_description = _describe_inline_gaps(gap_candidates)
    if gap_description:
        numbered += f"\nBlank candidates (choose one):\n{gap_description}"
    if label_start_idx is not None:
        numbered += f"\nApproximate matched label span: [{label_start_idx}]-[{end_idx}]"

    result = _llm_find_inline_markers(field["label"], numbered, field.get("type"), field_context)
    if result is None or result.candidate_idx is None:
        return None
    if not (0 <= result.candidate_idx < len(gap_candidates)):
        return None
    x = gap_candidates[result.candidate_idx]["x0"]

    y0, y1 = line["box"][1], line["box"][3]
    return {"x": x, "y": y0, "height": y1 - y0}


def compute_left_inline_field_position(field: dict, line: dict, end_idx: int, lines: list[dict]) -> dict | None:
    label_start_idx = _infer_label_start_idx(field["label"], line, end_idx)
    if label_start_idx is None:
        return None

    label_x = line["token_boxes"][label_start_idx][0]
    line_height = max(1, line["box"][3] - line["box"][1])
    reference = _find_nearby_checkbox_reference(line, label_x, lines)
    if reference is not None:
        input_x = reference["control_box"][0] + max(1, round(line_height * LEFT_INLINE_INPUT_OFFSET_RATIO))
    else:
        separator_idx = _find_post_label_separator(line, end_idx)
        right_gap = 0
        if separator_idx is not None and separator_idx + 1 < len(line["token_boxes"]):
            right_gap = line["token_boxes"][separator_idx + 1][0] - line["token_boxes"][separator_idx][2]
        input_width = max(round(line_height * 2.25), right_gap)
        input_x = label_x - input_width

    if input_x >= label_x:
        return None
    y0, y1 = line["box"][1], line["box"][3]
    return {"x": input_x, "y": y0, "height": y1 - y0}


def compute_date_field_positions(field: dict, line: dict, end_idx: int, lines: list[dict]) -> dict[str, dict[str, int]] | None:
    numbered, references, input_candidates = _build_date_context(lines, line, end_idx)
    if not references:
        return None

    result = _llm_find_date_markers(field["label"], numbered)
    if result is None:
        return None

    positions = {}
    for part_name, marker_idx in (("year", result.year_marker_idx), ("month", result.month_marker_idx), ("day", result.day_marker_idx)):
        if marker_idx is None or not (0 <= marker_idx < len(references)):
            continue
        expected_marker = {"year": "年", "month": "月", "day": "日"}[part_name]
        input_idx = getattr(result, f"{part_name}_input_idx")
        candidate = None
        if input_idx is not None and 0 <= input_idx < len(input_candidates):
            possible = input_candidates[input_idx]
            if possible["marker"] == expected_marker and possible["reference_idx"] == marker_idx:
                candidate = possible
        if candidate is None:
            candidate = next(
                (possible for possible in input_candidates if possible["marker"] == expected_marker and possible["reference_idx"] == marker_idx),
                None,
            )
        if candidate is None:
            continue
        positions[part_name] = {
            "x": candidate["x0"], "y": line["box"][1],
            "width": candidate["x1"] - candidate["x0"], "height": line["box"][3] - line["box"][1],
        }

    return positions or None


# ------------------------- 类型判定 + 处理器 -------------------------


def classify_value_region(
    field: dict, line: dict, end_idx: int,
    lines: list[dict] | None = None,
    field_context: dict | None = None,
) -> list[ValueRegionType]:
    """纯几何/结构判断，不调用 LLM，确保"归到哪一类"是确定性的、可复现的；
    不确定性只留给"类型内部具体选哪个候选"这一层。"""
    field_type = field.get("type")

    if field_type in ("SingleChoice", "MultiChoice"):
        return [ValueRegionType.GLYPH_OPTION]

    if field_type == "Date":
        if _count_slashes_after_label(line, end_idx) in (1, 2):
            return [ValueRegionType.GLYPH_DATE_PART]
        if _has_date_char_markers(line, end_idx):
            return [ValueRegionType.GLYPH_DATE_PART]
        if lines is not None:
            found = _find_date_layout_block(line, end_idx, lines)
            if found is not None:
                if field_context is not None:
                    field_context["dateTargetLine"] = found[0]
                    field_context["dateTargetStartIdx"] = found[1]
                return [ValueRegionType.GLYPH_DATE_PART]
        if _find_colon_boundary(line, end_idx) is not None:
            return [ValueRegionType.COLON_ANCHORED]
        return [ValueRegionType.GAP_ANOMALY]

    detected: list[ValueRegionType] = []

    if field_type == "Boolean":
        label_start_idx = _infer_label_start_idx(field["label"], line, end_idx)
        if _find_glyph_immediately_before_label(line, label_start_idx) is not None:
            detected.append(ValueRegionType.GLYPH_BEFORE_LABEL)
        else:
            detected.append(ValueRegionType.GLYPH_MISSING_ESTIMATE)
    elif (
        field_type in ("Text", "Number")
        and lines is not None
        and _has_left_inline_input_signal(field["label"], line, end_idx, lines)
    ):
        detected.append(ValueRegionType.INLINE_BEFORE_LABEL)
    elif _find_colon_boundary(line, end_idx) is not None:
        detected.append(ValueRegionType.COLON_ANCHORED)
    elif _find_anomalous_gap_after_label(line, end_idx) is not None:
        detected.append(ValueRegionType.GAP_ANOMALY)
    else:
        detected.append(ValueRegionType.GLYPH_MISSING_ESTIMATE)

    if _line_remainder_after_colon(line, end_idx):
        detected.append(ValueRegionType.OVERWRITE_REGION)

    return detected


def _handle_colon_anchored(field, line, end_idx, lines, field_context) -> PositionResult:
    position = compute_text_field_position(line, end_idx)
    return PositionResult(position, Confidence.HIGH, "colon_anchored")


def _handle_inline_before_label(field, line, end_idx, lines, field_context) -> PositionResult:
    position = compute_left_inline_field_position(field, line, end_idx, lines)
    if position is None:
        return PositionResult(None, Confidence.UNRESOLVED, "inline_before_label_unresolved", needs_review=True)
    return PositionResult(position, Confidence.LOW, "inline_before_label", needs_review=True)


def _handle_glyph_before_label(field, line, end_idx, lines, field_context) -> PositionResult:
    label_start_idx = _infer_label_start_idx(field["label"], line, end_idx)
    box = _find_glyph_immediately_before_label(line, label_start_idx)
    if box is None:
        return PositionResult(None, Confidence.UNRESOLVED, "glyph_before_label_missing", needs_review=True)
    return PositionResult(box, Confidence.HIGH, "glyph_before_label_found")


def _handle_gap_anomaly(field, line, end_idx, lines, field_context) -> PositionResult:
    gap = _find_anomalous_gap_after_label(line, end_idx)
    if gap is not None:
        y0, y1 = line["box"][1], line["box"][3]
        position = {"x": gap["x0"], "y": y0, "height": y1 - y0}
        # 样本量太少、或离群程度只是刚好过线（没有明显余量），都不给 HIGH，
        # 避免"看起来很确定，实际上是踩线判断"的结果被当成不需要复核。
        reliable = (
            gap["sample_size"] >= GAP_ANOMALY_MIN_SAMPLES
            and gap["strength"] >= GAP_ANOMALY_MAD_K * GAP_ANOMALY_STRONG_MARGIN
        )
        confidence = Confidence.HIGH if reliable else Confidence.MEDIUM
        return PositionResult(
            position, confidence, "gap_anomaly", needs_review=not reliable,
            extra={
                "gapWidth": round(gap["width"], 1),
                "strength": round(gap["strength"], 2),
                "sampleSize": gap["sample_size"],
            },
        )
    position = compute_inline_field_position(field, line, end_idx, field_context)
    if position is not None:
        return PositionResult(position, Confidence.MEDIUM, "inline_llm_fallback")
    return PositionResult(None, Confidence.UNRESOLVED, "gap_anomaly_unresolved", needs_review=True)


def _handle_glyph_date_part(field, line, end_idx, lines, field_context) -> PositionResult:
    target_line, target_end_idx = line, end_idx
    cross_block = False
    if field_context and "dateTargetLine" in field_context:
        target_line = field_context["dateTargetLine"]
        target_end_idx = field_context["dateTargetStartIdx"]
        cross_block = True

    date_positions = _compute_slash_date_positions(target_line, target_end_idx)
    method, confidence = "slash_deterministic", Confidence.HIGH
    if date_positions is None:
        date_positions = compute_date_field_positions(field, target_line, target_end_idx, lines)
        method, confidence = "llm_date_markers", Confidence.MEDIUM

    if not date_positions:
        position = compute_text_field_position(line, end_idx)  # 兜底仍用原始 label 行
        return PositionResult(position, Confidence.LOW, "date_fallback_text", needs_review=True)

    first = next(iter(date_positions.values()))
    position = {k: first[k] for k in ("x", "y", "height")}

    if cross_block:
        # "这两个块属于同一个字段"本身是几何推断出来的，不是 100% 确定，
        # 哪怕坐标计算这一步走的是确定性的 slash 解析，整体置信度也要降一级并标记复核。
        confidence = Confidence.MEDIUM if confidence == Confidence.HIGH else Confidence.LOW
        return PositionResult(
            position, confidence, f"{method}_cross_block", needs_review=True,
            extra={"positions": date_positions, "crossBlockLineIdx": target_line["line_idx"]},
        )

    return PositionResult(position, confidence, method, extra={"positions": date_positions})


def _handle_glyph_missing_estimate(field, line, end_idx, lines, field_context) -> PositionResult:
    position = compute_inline_field_position(field, line, end_idx, field_context)
    if position is not None:
        return PositionResult(position, Confidence.LOW, "inline_llm_fallback", needs_review=True)
    return PositionResult(None, Confidence.UNRESOLVED, "not_found", needs_review=True)


def _handle_overwrite_region(field, line, end_idx, lines, field_context) -> PositionResult:
    position = compute_default_value_overwrite_position(field, line, end_idx)
    if position is None:
        return PositionResult(None, Confidence.MEDIUM, "llm_default_value_rejected")
    return PositionResult(position, Confidence.MEDIUM, "llm_default_value_detected")


TYPE_HANDLERS = {
    ValueRegionType.COLON_ANCHORED: _handle_colon_anchored,
    ValueRegionType.INLINE_BEFORE_LABEL: _handle_inline_before_label,
    ValueRegionType.GLYPH_BEFORE_LABEL: _handle_glyph_before_label,
    ValueRegionType.GAP_ANOMALY: _handle_gap_anomaly,
    ValueRegionType.GLYPH_DATE_PART: _handle_glyph_date_part,
    ValueRegionType.GLYPH_MISSING_ESTIMATE: _handle_glyph_missing_estimate,
    ValueRegionType.OVERWRITE_REGION: _handle_overwrite_region,
}


# ------------------------- 字段上下文 & 锚点 -------------------------


def _build_field_context(
    fields: list[dict], field_index: int,
    match: tuple | None = None, anchor_hints: list[float | None] | None = None,
) -> dict:
    field = fields[field_index]
    context = {
        "key": field.get("key"), "label": field.get("label"), "type": field.get("type"),
        "options": field.get("options") or [], "order": field.get("order", field_index + 1),
        "field_index": field_index,
        "previous_fields": [
            {"key": p.get("key"), "label": p.get("label"), "type": p.get("type"), "order": p.get("order")}
            for p in fields[max(0, field_index - 2) : field_index]
        ],
        "next_fields": [
            {"key": n.get("key"), "label": n.get("label"), "type": n.get("type"), "order": n.get("order")}
            for n in fields[field_index + 1 : field_index + 3]
        ],
    }
    if match is not None:
        context["anchor_line_idx"] = match[0]["line_idx"]
        context["anchor_y"] = match[0]["box"][1]
        context["anchor_text"] = match[0]["text"]
    if anchor_hints is not None:
        previous_hints = [hint for hint in anchor_hints[:field_index] if hint is not None]
        next_hints = [hint for hint in anchor_hints[field_index + 1 :] if hint is not None]
        if previous_hints:
            context["previous_anchor_y"] = previous_hints[-1]
        if next_hints:
            context["next_anchor_y"] = next_hints[0]
    return context


def compute_field_anchors(fields: list[dict], lines: list[dict]):
    """尝试用每个栏位自己的 label 在 OCR 里找到对应行，记录该行的 y0 作为"锚点"，
    同时把匹配结果（line, end_idx, method, score）缓存下来，后续所有该字段的定位都复用这一次匹配。"""
    anchors = []
    matches = []
    anchor_hints = []
    for field in fields:
        candidates = _find_top_candidates((field["label"],), lines, top_k=1)
        anchor_hints.append(candidates[0][1]["box"][1] if candidates else None)

    for field_index, field in enumerate(fields):
        context = _build_field_context(fields, field_index, anchor_hints=anchor_hints)
        found = find_label_position(field["label"], lines, field_context=context)
        if found is None:
            anchors.append(None)
            matches.append(None)
            continue
        line, end_idx, method, score = found
        anchors.append(line["box"][1])
        matches.append((line, end_idx, method, score))
    return anchors, matches


# ------------------------- 主流程 -------------------------


def extract_field_positions(fields_data: dict, ocr_data: dict) -> list[dict]:
    lines = flatten_ocr(ocr_data)
    fields = fields_data["fields"]

    print("========== 开始解析栏位坐标 ==========")
    anchors, matches = compute_field_anchors(fields, lines)

    results = []
    for field_index, (field, anchor, match) in enumerate(zip(fields, anchors, matches)):
        field_type = field["type"]
        entry = {"key": field["key"], "label": field["label"], "type": field_type}
        field_context = _build_field_context(fields, field_index, match)

        _log_header(field)

        if match is None:
            _log(f"anchor: 未找到（无法匹配到 label={field['label']!r}）")
            if field_type in ("SingleChoice", "MultiChoice"):
                options = field.get("options") or []
                entry["options"] = {opt: None for opt in options}
                entry["optionsResolution"] = {
                    opt: PositionResult(None, Confidence.UNRESOLVED, "no_anchor", needs_review=True).to_dict()
                    for opt in options
                }
            else:
                entry["position"] = None
                entry["resolution"] = {
                    "anchor_missing": PositionResult(None, Confidence.UNRESOLVED, "no_anchor", needs_review=True).to_dict()
                }
            results.append(entry)
            continue

        line, end_idx, anchor_method, anchor_score = match
        _log(
            f"anchor: method={anchor_method} score={anchor_score:.3f} "
            f"line={line['line_idx']} y={line['box'][1]} text={line['text']!r}"
        )

        if field_type in ("SingleChoice", "MultiChoice"):
            options = field.get("options") or []
            _log(f"region_type: [{ValueRegionType.GLYPH_OPTION.value}]")
            option_results = _resolve_option_positions(options, lines, field_context, anchor)
            entry["options"] = {opt: res.position for opt, res in option_results.items()}
            entry["optionsResolution"] = {opt: res.to_dict() for opt, res in option_results.items()}
            results.append(entry)
            continue

        region_types = classify_value_region(field, line, end_idx, lines, field_context)
        _log(f"region_type: {[t.value for t in region_types]}")

        resolution: dict[str, dict] = {}
        primary_position = None
        primary_positions = None
        for region_type in region_types:
            outcome = TYPE_HANDLERS[region_type](field, line, end_idx, lines, field_context)
            _record_stat(region_type, outcome.confidence)
            resolution[region_type.value] = outcome.to_dict()
            _log(
                f"{region_type.value} -> method={outcome.method} conf={outcome.confidence.value} "
                f"needs_review={outcome.needs_review} value={outcome.position}",
                indent=2,
            )
            if region_type == ValueRegionType.OVERWRITE_REGION:
                entry["overwritePosition"] = outcome.position
            else:
                primary_position = outcome.position
                if outcome.extra and "positions" in outcome.extra:
                    primary_positions = outcome.extra["positions"]

        entry["position"] = primary_position
        if primary_positions is not None:
            entry["positions"] = primary_positions
        entry["resolution"] = resolution
        _log(f"=> FINAL position={primary_position}")
        results.append(entry)

    return results


def main():
    fields_data = json.loads(FIELD_PATH.read_text(encoding="utf-8"))
    ocr_data = json.loads(OCR_RESULT_PATH.read_text(encoding="utf-8"))

    positions = extract_field_positions(fields_data, ocr_data)
    print_summary()

    out_path = FIELD_DIR / "llm_position.json"
    out_path.write_text(json.dumps(positions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out_path}，共 {len(positions)} 个栏位")


if __name__ == "__main__":
    main()
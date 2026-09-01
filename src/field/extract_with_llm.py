"""
根据「表单栏位提取结果」(fields.json) 和「OCR 结果（含字符级坐标）」(ocr_result.json)，
计算每个栏位在原图上应该显示的坐标。

设计说明（相较旧版的主要变化）：
- 移除了所有针对具体表单硬编码的特例表：
    FIELD_LABEL_ALIASES / OPTION_ALIASES / DATE_FIELD_LAYOUTS /
    INLINE_FIELD_LAYOUTS / DEFAULT_VALUE_FIELDS / FIELD_SEARCH_BOUNDARIES
- 这些特例改为：先用现有的模糊匹配（difflib）做第一道筛选，
    只有在分数不够高、或多个候选分数打平存在歧义时，才升级调用本地 LLM 做裁决。
- 每次 LLM 调用只传入少量候选文本（不含坐标数组），并做 token 预算校验，
    避免超出本地模型 8192 的上下文上限。
- 同一栏位的标签匹配结果会被复用（只匹配一次），避免同一字段被多次调用模型。

使用前需要根据你实际的本地 vLLM / Gemma4 部署方式，调整下面
"LLM 客户端配置" 部分的 base_url / model，以及 `_call_llm` 里
`with_structured_output` 的调用方式（如果你们的 Gemma4 tool parser
在结构化输出上有线程安全问题，可以在这里换成你们已有的封装）。
"""

import difflib
import json
import os
import opencc
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Type, TypeVar

from src.utils.path import OUT_DIR
load_dotenv()


CHECKBOX_GLYPHS = {"□", "○", "◯", "☐"}

FIELD_DIR = OUT_DIR / "field"
FIELD_PATH = FIELD_DIR / "field_116.json"
OCR_RESULT_PATH = OUT_DIR / "ocr" / "ocr_result.json"

TEXT_FIELD_GAP = 10  # 通用常量，标题冒号右边留白，不属于"特例"

s2t = opencc.OpenCC("s2t.json")


# ------------------------- LLM 客户端配置 -------------------------

_LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
_LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4")
_LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
_LLM_MAX_CONTEXT_TOKENS = int(os.environ.get("LLM_MAX_CONTEXT_TOKENS", "8192"))
_LLM_OUTPUT_RESERVE_TOKENS = 512  # 给模型输出 + chat template 预留的余量

# 每次匹配最多带几个候选给模型看，候选越多单次 prompt 越长，
# 如果频繁触发预算报错，可以调低这个值。
MAX_CANDIDATES = int(os.environ.get("LLM_MAX_CANDIDATES", "5"))
MAX_DATE_CONTEXT_LINES = int(os.environ.get("LLM_DATE_CONTEXT_LINES", "6"))

# 模糊匹配分数阈值：>= CONFIDENCE 直接采信，不调用模型；
# < MIN_CONSIDER 视为根本不是同一个东西，也不调用模型（省 token）；
# 落在中间区间，或候选之间分数打平（有歧义），才升级给模型裁决。
LABEL_MATCH_CONFIDENCE = 0.85
LABEL_MATCH_MIN_CONSIDER = 0.35

OPTION_MATCH_CONFIDENCE = 0.85
OPTION_MATCH_MIN_CONSIDER = 0.4
OPTION_AMBIGUOUS_GAP = 0.08  # 与次优候选的分差小于此值，视为存在歧义
LABEL_AMBIGUOUS_GAP = 0.08
MAX_OPTION_CONTEXT_TOKENS = 8

_llm_client = ChatOpenAI(
    base_url=_LLM_BASE_URL,
    api_key=_LLM_API_KEY, # type: ignore
    model=_LLM_MODEL,
    temperature=0,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        },
        "reasoning_effort": "none",
        "skip_reasoning": True,
        "skip_special_tokens": True
    }
)


class LabelMatchResult(BaseModel):
    matched_idx: int | None = Field(
        description="所选候选行在候选列表中的下标（从 0 开始）；如果都不匹配则为 null"
    )


class OptionMatchResult(BaseModel):
    matched_idx: int | None = Field(
        description="真正属于该栏位的候选下标（从 0 开始）；如果都不属于则为 null"
    )


class OptionGroupMatchResult(BaseModel):
    matched_indices: list[int | None] = Field(
        description="按目标选项顺序返回每个选项对应的候选下标；没有匹配则为 null"
    )


class DefaultValueResult(BaseModel):
    has_default_value: bool = Field(description="冒号后的文字是否是需要清除的预填默认答案")


class InlineMarkersResult(BaseModel):
    candidate_idx: int | None = Field(
        description="所选候选空白区的编号；如果没有可填写区则为 null"
    )


class DateMarkersResult(BaseModel):
    year_marker_idx: int | None = Field(description="“年”字所在 token 编号，找不到为 null")
    month_marker_idx: int | None = Field(description="“月”字所在 token 编号，找不到为 null")
    day_marker_idx: int | None = Field(description="“日”字所在 token 编号，找不到为 null")
    year_input_idx: int | None = Field(
        default=None, description="年对应的输入区候选编号，没有则为 null"
    )
    month_input_idx: int | None = Field(
        default=None, description="月对应的输入区候选编号，没有则为 null"
    )
    day_input_idx: int | None = Field(
        default=None, description="日对应的输入区候选编号，没有则为 null"
    )


def _estimate_tokens(text: str) -> int:
    # 粗略估算：中英文混排场景下按约 1.2 字符/ token 估计，偏保守。
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
    _check_budget(system_with_format, user_prompt)
    structured_llm = _llm_client.model_copy(
            update={"disable_streaming": True}
        ).with_structured_output(
            schema,
            method="json_mode",
            include_raw=False,
        )
    try:
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
        print(f"[llm] 调用失败，本次判定按未匹配处理：{exc}")
        return None


# ------------------------- 通用文本处理 -------------------------


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


def _line_remainder_after_colon(line: dict, start_idx: int) -> dict | None:
    """找到 start_idx 之后第一个冒号，返回冒号右边剩余文字的文本和横向范围。
    用来判断"标题冒号后面印着的文字"是不是需要清除的默认值，
    不再依赖预先声明"哪个栏位有默认值、默认值是什么"。
    """
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
        return {
            "text": "".join(remainder_tokens),
            "x0": x0,
            "x1": remainder_boxes[-1][2],
        }
    return None


def _find_top_candidates(
    targets: tuple[str, ...],
    lines: list[dict],
    top_k: int = MAX_CANDIDATES,
    preferred_y: float | None = None,
):
    """在所有 OCR 行里找出与 target 最匹配的若干候选（按分数从高到低），
    每一行最多贡献一个候选。"""
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
                        scored.append(
                            (_match_score(acc_n, target_n), line, start, len(tokens) - 1)
                        )

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
    actual_without_controls = "".join(
        char for char in actual_n if char not in CHECKBOX_GLYPHS
    )
    prefix = actual_without_controls[: len(target)]
    return max(
        fuzzy_ratio(actual_without_controls, target),
        fuzzy_ratio(prefix, target),
    )


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
                        + (
                            0.2
                            if start > 0
                            and tokens[start - 1].strip() in CHECKBOX_GLYPHS
                            else 0
                        )
                        + (
                            0.2
                            if end == segment_end - 1
                            and len(candidate_text) >= len(target_n)
                            else 0
                        ),
                        -abs(len(candidate_text) - len(target_n)),
                        end == segment_end - 1 and len(candidate_text) >= len(target_n),
                        -end,
                    )
                    if best_for_start is None or candidate_key > best_for_start[0]:
                        best_for_start = (candidate_key, candidate)

                if best_for_start is not None:
                    if (
                        segment_candidate is None
                        or best_for_start[0] > segment_candidate[0]
                    ):
                        segment_candidate = best_for_start
            if segment_candidate is not None:
                scored.append(segment_candidate[1])

    def sort_key(item):
        score, line, start, end = item
        boundary_score = score
        if start > 0 and line["tokens"][start - 1].strip() in CHECKBOX_GLYPHS:
            boundary_score += 0.2
        if any(
            token.strip() in CHECKBOX_GLYPHS
            for token in line["tokens"][start + 1 : end + 1]
        ):
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
    """把 OCR 结果整理成便于查找的结构：每一行的整体文本/坐标框，
    以及行内逐字符（token）的文本和坐标。"""
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
                "box": ocr_data.get("rec_boxes", prunedResult.get("rec_boxes", []))[i],  # [x0, y0, x1, y1]
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
        "key",
        "label",
        "type",
        "order",
        "field_index",
        "anchor_line_idx",
        "anchor_y",
        "previous_anchor_y",
        "next_anchor_y",
    ):
        value = field_context.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    for relation in ("previous_fields", "next_fields"):
        fields = field_context.get(relation) or []
        if fields:
            parts.append(
                f"{relation}="
                + ", ".join(
                    f"{item.get('order', '?')}:{item.get('key', '')}/{item.get('label', '')}"
                    for item in fields
                )
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


def _llm_resolve_option(
    option_text: str,
    candidates: list,
    field_context: dict | str | None,
    lines: list[dict],
):
    by_line_idx = {line["line_idx"]: line for line in lines}
    parts = []
    for i, (_, line, start_idx, end_idx) in enumerate(candidates):
        before = by_line_idx.get(line["line_idx"] - 1)
        after = by_line_idx.get(line["line_idx"] + 1)
        preceding_token = line["tokens"][start_idx - 1].strip() if start_idx > 0 else None
        following_token = (
            line["tokens"][end_idx + 1].strip()
            if end_idx + 1 < len(line["tokens"])
            else None
        )
        context = " | ".join(
            part
            for part in (
                f"before={before['text']}" if before else "",
                f"selected={line['text']}",
                f"after={after['text']}" if after else "",
            )
            if part
        )
        parts.append(
            f"{i}. line={line['line_idx']} y={line['box'][1]} "
            f"tokens={_line_token_text(line)} selected_range=[{start_idx}]-[{end_idx}] "
            f"control_before={preceding_token in CHECKBOX_GLYPHS} "
            f"control_after={following_token in CHECKBOX_GLYPHS} "
            f"context={context}"
        )
    lines_desc = "\n".join(parts)

    system = (
        "You resolve a form option occurrence from OCR. The same option may appear in several "
        "sections. Use the field metadata, anchor row, candidate line number, selected token "
        "range, control-boundary flags, and neighboring row text to choose the occurrence "
        "belonging to the field. If OCR shows only part of an option before the next control "
        "glyph, it may still be the requested option. "
        "Return null only when none belongs to the field."
    )
    user = (
        f"Field metadata: {_field_context_text(field_context)}\n"
        f"Target option: {option_text}\n"
        f"Candidates (index starts at 0):\n{lines_desc}"
    )
    result = _call_llm(system, user, OptionMatchResult)
    if result is None or result.matched_idx is None:
        return None
    if 0 <= result.matched_idx < len(candidates):
        return result.matched_idx
    return None


def _llm_resolve_option_group(
    options: list[str],
    candidates: list,
    field_context: dict | str | None,
    lines: list[dict],
) -> list[int | None] | None:
    by_line_idx = {line["line_idx"]: line for line in lines}
    candidate_descriptions = []
    for candidate_idx, (score, line, start_idx, end_idx) in enumerate(candidates):
        preceding_token = line["tokens"][start_idx - 1].strip() if start_idx > 0 else None
        following_token = (
            line["tokens"][end_idx + 1].strip()
            if end_idx + 1 < len(line["tokens"])
            else None
        )
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

    option_descriptions = "\n".join(
        f"{index}. {option}" for index, option in enumerate(options)
    )
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
    return [
        index if index is None or 0 <= index < len(candidates) else None
        for index in result.matched_indices
    ]


def _llm_detect_default_value(label: str, remainder_text: str) -> DefaultValueResult | None:
    system = (
        "你在协助解析医院表单。栏位标题冒号后面如果印着一段文字，"
        "可能是表单预先印好、需要在填写时清除的默认答案，也可能只是说明文字。"
        "请判断这段文字是否属于需要被覆盖清除的默认答案。"
    )
    user = f"栏位标题：{label}\n冒号后文字：{remainder_text}"
    return _call_llm(system, user, DefaultValueResult)


def _llm_find_inline_markers(
    label: str,
    numbered_snippet: str,
    field_type: str | None = None,
    field_context: dict | str | None = None,
) -> InlineMarkersResult | None:
    system = (
        "You locate a form field input area from OCR. OCR may omit underlines, boxes, circles, "
        "or some text. Return only JSON with one field, candidate_idx. Select one listed blank "
        "candidate. If exactly one candidate is listed, return candidate_idx=0. For a Number or "
        "Text field, a writable area can be inside the matched label span, especially immediately "
        "before a unit token. For a Text field whose matched label starts at token 0, prefer a "
        "positive-width leading candidate when one exists; trailing punctuation or unit text is "
        "context, not the input area. Do not choose a control gap before the label for Number "
        "fields. For a Boolean field, "
        "choose a positive-width control gap immediately before the label when present. Return "
        "null only when no candidate belongs to the field."
    )
    user = (
        f"Field label: {label}\n"
        + (f"Field type: {field_type}\n" if field_type else "")
        + (f"Field metadata: {_field_context_text(field_context)}\n" if field_context else "")
        + f"Candidate context:\n{numbered_snippet}"
    )
    return _call_llm(system, user, InlineMarkersResult)


def _find_inline_gap_candidates(
    line: dict, label_start_idx: int | None, label_end_idx: int
) -> list[dict]:
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
        gaps.append(
            {
                "before_idx": before_idx,
                "after_idx": after_idx,
                "x0": gap_start,
                "x1": gap_end,
                "relation": relation,
                "source": "ocr",
            }
        )

    line_start = line["box"][0]
    first_start = token_boxes[0][0]
    line_height = max(box[3] - box[1] for box in token_boxes)
    leading_gap = first_start - line_start
    if (
        leading_gap < line_height * 0.5
        and line["tokens"][0].strip() not in CHECKBOX_GLYPHS
    ):
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
        f"candidate {index}: "
        f"{endpoint(candidate.get('before_idx'), 'line-start')} -> "
        f"{endpoint(candidate.get('after_idx'), 'line-end')} "
        f"relation={candidate['relation']} source={candidate.get('source', 'ocr')} "
        f"width={candidate['x1'] - candidate['x0']:g}"
        for index, candidate in enumerate(gaps)
    )


def _filter_inline_gap_candidates(
    line: dict, field_type: str | None, gaps: list[dict]
) -> list[dict]:
    def touches_control(candidate: dict) -> bool:
        for token_idx in (candidate["before_idx"], candidate["after_idx"]):
            if token_idx is not None and line["tokens"][token_idx].strip() in CHECKBOX_GLYPHS:
                return True
        return False

    if field_type == "Boolean":
        before_label = [candidate for candidate in gaps if candidate["relation"] == "before-label"]
        if before_label:
            return before_label

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
        (
            char,
            round(box[0] + char_width * offset),
            round(box[0] + char_width * (offset + 1)),
        )
        for offset, char in enumerate(token)
    ]


def _median(values: list[int]) -> int:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return round((ordered[middle - 1] + ordered[middle]) / 2)


def _estimate_date_glyph_width(references: list[tuple[dict, int]]) -> int:
    heights = [
        max(1, box[3] - box[1])
        for line, token_idx in references
        for box in [line["token_boxes"][token_idx]]
    ]
    narrow_widths = [
        box[2] - box[0]
        for line, token_idx in references
        for box in [line["token_boxes"][token_idx]]
        if box[2] - box[0] <= max(heights) * 1.5
    ]
    widths = narrow_widths or [
        line["token_boxes"][token_idx][2] - line["token_boxes"][token_idx][0]
        for line, token_idx in references
    ]
    return max(1, _median(widths))


def _infer_date_content_left(
    lines: list[dict], target_line: dict, marker_start: int, glyph_width: int
) -> int:
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


def _find_date_input_candidates(
    lines: list[dict], references: list[tuple[dict, int]]
) -> list[dict]:
    candidates = []
    glyph_width = _estimate_date_glyph_width(references)
    previous_marker_end = None
    for reference_idx, (line, token_idx) in enumerate(references):
        token = line["tokens"][token_idx]
        token_box = line["token_boxes"][token_idx]
        token_width = token_box[2] - token_box[0]
        for char, char_start, _ in _token_char_boxes(token, token_box):
            if char not in {"年", "月", "日"}:
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
                    input_start = max(
                        line_start, previous_marker_end + max(1, glyph_width // 2)
                    )
            else:
                input_start = _infer_date_content_left(
                    lines, line, marker_start, glyph_width
                )
            if input_start >= marker_start:
                input_start = marker_start - glyph_width
            candidates.append(
                {
                    "marker": char,
                    "reference_idx": reference_idx,
                    "x0": input_start,
                    "x1": marker_start,
                }
            )
            previous_marker_end = marker_start + glyph_width
    return candidates


def _count_slashes_after_label(line: dict, end_idx: int) -> int:
    return sum(token.count("/") for token in line["tokens"][end_idx:])


def _compute_slash_date_positions(
    line: dict, end_idx: int
) -> dict[str, dict[str, int]] | None:
    slash_ranges = []
    for token_idx in range(end_idx, len(line["tokens"])):
        for char, char_start, char_end in _token_char_boxes(
            line["tokens"][token_idx], line["token_boxes"][token_idx]
        ):
            if char == "/":
                slash_ranges.append((char_start, char_end))

    slash_count = len(slash_ranges)
    if slash_count not in (1, 2):
        return None

    label_end = _find_colon_boundary(line, end_idx)
    if label_end is None:
        label_end = (
            line["token_boxes"][end_idx][2]
            if 0 <= end_idx < len(line["token_boxes"])
            else line["box"][0]
        )
    input_start = label_end + TEXT_FIELD_GAP
    first_slash_start = slash_ranges[0][0]
    if input_start >= first_slash_start:
        input_start = max(line["box"][0], first_slash_start - max(1, line["box"][3] - line["box"][1]))

    boundaries = [input_start]
    for slash_start, slash_end in slash_ranges:
        boundaries.extend((slash_start, slash_end))
    boundaries.append(line["box"][2])

    part_names = ("month", "day") if slash_count == 1 else ("year", "month", "day")
    positions = {}
    y0, y1 = line["box"][1], line["box"][3]
    for part_idx, part_name in enumerate(part_names):
        x0 = boundaries[part_idx * 2]
        x1 = boundaries[part_idx * 2 + 1]
        if x1 <= x0:
            x1 = x0 + max(1, y1 - y0)
        positions[part_name] = {
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y1 - y0,
        }
    return positions


def _build_date_context(
    lines: list[dict], line: dict, end_idx: int
) -> tuple[str, list[tuple[dict, int]], list[dict]]:
    line_position = next(
        (position for position, candidate in enumerate(lines) if candidate is line),
        None,
    )
    if line_position is None:
        return "", [], []

    references = []
    context_lines = lines[
        line_position : line_position + max(1, MAX_DATE_CONTEXT_LINES)
    ]
    for offset, context_line in enumerate(context_lines):
        token_start = end_idx + 1 if offset == 0 else 0
        for token_idx in range(token_start, len(context_line["tokens"])):
            references.append((context_line, token_idx))

    numbered = "\n".join(
        f"[{index}] line={context_line['line_idx']} token={token_idx}: "
        f"{context_line['tokens'][token_idx]}"
        for index, (context_line, token_idx) in enumerate(references)
    )
    input_candidates = _find_date_input_candidates(lines, references)
    if input_candidates:
        numbered += "\nInput candidates (index starts at 0):\n" + "\n".join(
            f"candidate {index}: marker={candidate['marker']} "
            f"token_ref=[{candidate['reference_idx']}] "
            f"x={candidate['x0']}..{candidate['x1']}"
            for index, candidate in enumerate(input_candidates)
        )
    return numbered, references, input_candidates


# ------------------------- 标签 / 选项 定位 -------------------------


def find_label_position(
    label: str,
    lines: list[dict],
    field_context: dict | str | None = None,
):
    """在所有 OCR 行里找一段 token 序列跟 label 最匹配的位置，返回 (line, end_token_idx)。
    分数够高直接采信；分数模糊时才调用 LLM 裁决。"""
    candidates = _find_top_candidates((label,), lines, top_k=MAX_CANDIDATES)
    if not candidates:
        return None

    if isinstance(field_context, dict):
        previous_y = field_context.get("previous_anchor_y")
        next_y = field_context.get("next_anchor_y")
        if previous_y is not None and next_y is not None and previous_y < next_y:
            target_n = normalize(label)
            expanded_candidates = _find_top_candidates(
                (label,), lines, top_k=len(lines)
            )
            bounded_exact = [
                candidate
                for candidate in expanded_candidates
                if previous_y < candidate[1]["box"][1] < next_y
                and normalize(
                    "".join(candidate[1]["tokens"][candidate[2] : candidate[3] + 1])
                )
                == target_n
            ]
            if bounded_exact:
                candidates = bounded_exact

    top_score = candidates[0][0]
    if top_score < LABEL_MATCH_MIN_CONSIDER:
        return None

    ambiguous = (
        len(candidates) > 1
        and (top_score - candidates[1][0]) < LABEL_AMBIGUOUS_GAP
    )
    if top_score >= LABEL_MATCH_CONFIDENCE and not ambiguous:
        _, line, _, end = candidates[0]
        return line, end

    chosen = _llm_resolve_label(label, candidates, field_context)
    if chosen is None:
        return None
    _, line, _, end = candidates[chosen]
    return line, end


def find_option_position(
    option_text: str,
    lines: list[dict],
    field_label: str | None = None,
    preferred_y: float | None = None,
    field_context: dict | str | None = None,
):
    """搜索选项文字，返回它前面 checkbox 符号的坐标；
    找不到符号时，按选项首字的框往左推一个估算框。
    同一选项文字在多处出现、或分数不够高时，交给 LLM 结合所属栏位裁决。"""
    candidates = _find_option_candidates(
        option_text,
        lines,
        top_k=MAX_CANDIDATES,
        preferred_y=preferred_y,
    )
    if not candidates:
        return None

    top_score = candidates[0][0]
    if top_score < OPTION_MATCH_MIN_CONSIDER:
        return None

    if field_context is None and field_label is not None:
        field_context = {"label": field_label}

    chosen_idx = 0
    candidate_text = normalize(
        "".join(candidates[0][1]["tokens"][candidates[0][2] : candidates[0][3] + 1])
    )
    exact_match = candidate_text == normalize(option_text)
    ambiguous = (
        len(candidates) > 1 and (candidates[0][0] - candidates[1][0]) < OPTION_AMBIGUOUS_GAP
    )
    if top_score < OPTION_MATCH_CONFIDENCE or ambiguous or not exact_match:
        resolved = _llm_resolve_option(option_text, candidates, field_context, lines)
        if resolved is not None:
            chosen_idx = resolved

    _, line, start_idx, _ = candidates[chosen_idx]
    return _extract_option_box(line, start_idx)


def _extract_option_box(line: dict, start_idx: int) -> dict:
    first_box = line["token_boxes"][start_idx]

    checkbox_idx = None
    if line["tokens"][start_idx].strip() in CHECKBOX_GLYPHS:
        checkbox_idx = start_idx
    elif start_idx - 1 >= 0 and line["tokens"][start_idx - 1].strip() in CHECKBOX_GLYPHS:
        checkbox_idx = start_idx - 1

    if checkbox_idx is not None:
        cb_box = line["token_boxes"][checkbox_idx]
        return {
            "x": cb_box[0],
            "y": cb_box[1],
            "width": cb_box[2] - cb_box[0],
            "height": cb_box[3] - cb_box[1],
        }

    # 没有符号：按第一个字的框高度估算一个等大方框，放在它左边
    char_h = first_box[3] - first_box[1]
    return {
        "x": first_box[0] - char_h,
        "y": first_box[1],
        "width": char_h,
        "height": char_h,
    }


def _collect_option_candidates(
    options: list[str], lines: list[dict], preferred_y: float | None
) -> tuple[list[list], list]:
    per_option = []
    pooled = []
    by_identity = {}
    for option in options:
        candidates = _find_option_candidates(
            option,
            lines,
            top_k=MAX_CANDIDATES,
            preferred_y=preferred_y,
        )
        relevant = [
            candidate
            for candidate in candidates
            if candidate[0] >= OPTION_MATCH_MIN_CONSIDER
        ] or candidates[:1]
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
    options: list[str],
    lines: list[dict],
    field_context: dict | str | None,
    preferred_y: float | None,
) -> dict[str, dict | None]:
    per_option, pooled = _collect_option_candidates(options, lines, preferred_y)
    if not pooled:
        return {option: None for option in options}

    needs_group_resolution = any(
        len(candidates) != 1
        or candidates[0][0] < OPTION_MATCH_CONFIDENCE
        or normalize(
            "".join(
                candidates[0][1]["tokens"][candidates[0][2] : candidates[0][3] + 1]
            )
        )
        != normalize(option)
        for option, candidates in zip(options, per_option)
        if candidates
    )
    assignments = None
    if needs_group_resolution:
        assignments = _llm_resolve_option_group(options, pooled, field_context, lines)

    result = {}
    used_candidates = set()
    for option_idx, option in enumerate(options):
        candidate = None
        assignment = assignments[option_idx] if assignments is not None else None
        allowed = {
            (item[1]["line_idx"], item[2], item[3])
            for item in per_option[option_idx]
        }
        if assignment is not None and 0 <= assignment < len(pooled):
            assigned_candidate = pooled[assignment]
            identity = (
                assigned_candidate[1]["line_idx"],
                assigned_candidate[2],
                assigned_candidate[3],
            )
            if identity in allowed and identity not in used_candidates:
                candidate = assigned_candidate
        if candidate is None:
            for fallback in per_option[option_idx]:
                identity = (fallback[1]["line_idx"], fallback[2], fallback[3])
                if identity not in used_candidates:
                    candidate = fallback
                    break
        if candidate is not None:
            used_candidates.add((candidate[1]["line_idx"], candidate[2], candidate[3]))
        result[option] = (
            _extract_option_box(candidate[1], candidate[2]) if candidate else None
        )
    return result


# ------------------------- 非选择类栏位定位 -------------------------


def compute_text_field_position(line: dict, end_idx: int) -> dict:
    x = _find_colon_boundary(line, end_idx)
    if x is None:
        x = line["token_boxes"][end_idx][2]
    x += TEXT_FIELD_GAP
    y0, y1 = line["box"][1], line["box"][3]
    return {"x": x, "y": y0, "height": y1 - y0}


def compute_default_value_overwrite_position(
    field: dict, line: dict, end_idx: int
) -> dict | None:
    remainder = _line_remainder_after_colon(line, end_idx)
    if remainder is None or not remainder["text"].strip():
        return None

    result = _llm_detect_default_value(field["label"], remainder["text"])
    if result is None or not result.has_default_value:
        return None

    y0, y1 = line["box"][1], line["box"][3]
    return {
        "x": remainder["x0"],
        "y": y0,
        "width": remainder["x1"] - remainder["x0"],
        "height": y1 - y0,
    }


def compute_inline_field_position(
    field: dict,
    line: dict,
    end_idx: int,
    field_context: dict | str | None = None,
) -> dict | None:
    tokens = line["tokens"]
    if not tokens:
        return None

    numbered = "OCR tokens: " + " ".join(
        f"[{index}]{token}" for index, token in enumerate(tokens)
    )
    label_start_idx = _infer_label_start_idx(field["label"], line, end_idx)
    gap_candidates = _find_inline_gap_candidates(line, label_start_idx, end_idx)
    gap_candidates = _filter_inline_gap_candidates(line, field.get("type"), gap_candidates)
    if field.get("type") == "Text" and label_start_idx == 0:
        leading_candidates = [
            candidate
            for candidate in gap_candidates
            if candidate["relation"] == "before-label"
        ]
        if leading_candidates:
            gap_candidates = leading_candidates
    gap_description = _describe_inline_gaps(gap_candidates)
    if gap_description:
        numbered += f"\nBlank candidates (choose one):\n{gap_description}"
    if label_start_idx is not None:
        numbered += f"\nApproximate matched label span: [{label_start_idx}]-[{end_idx}]"

    result = _llm_find_inline_markers(
        field["label"], numbered, field.get("type"), field_context
    )
    if result is None or result.candidate_idx is None:
        return None
    if not (0 <= result.candidate_idx < len(gap_candidates)):
        return None
    x = gap_candidates[result.candidate_idx]["x0"]

    y0, y1 = line["box"][1], line["box"][3]
    return {"x": x, "y": y0, "height": y1 - y0}


def compute_date_field_positions(
    field: dict, line: dict, end_idx: int, lines: list[dict]
) -> dict[str, dict[str, int]] | None:
    numbered, references, input_candidates = _build_date_context(lines, line, end_idx)
    if not references:
        return None

    result = _llm_find_date_markers(field["label"], numbered)
    if result is None:
        return None

    positions = {}
    for part_name, marker_idx in (
        ("year", result.year_marker_idx),
        ("month", result.month_marker_idx),
        ("day", result.day_marker_idx),
    ):
        if marker_idx is None:
            continue
        if not (0 <= marker_idx < len(references)):
            continue
        expected_marker = {"year": "年", "month": "月", "day": "日"}[part_name]
        input_idx = getattr(result, f"{part_name}_input_idx")
        candidate = None
        if input_idx is not None and 0 <= input_idx < len(input_candidates):
            possible = input_candidates[input_idx]
            if (
                possible["marker"] == expected_marker
                and possible["reference_idx"] == marker_idx
            ):
                candidate = possible
        if candidate is None:
            candidate = next(
                (
                    possible
                    for possible in input_candidates
                    if possible["marker"] == expected_marker
                    and possible["reference_idx"] == marker_idx
                ),
                None,
            )
        if candidate is None:
            continue
        positions[part_name] = {
            "x": candidate["x0"],
            "y": line["box"][1],
            "width": candidate["x1"] - candidate["x0"],
            "height": line["box"][3] - line["box"][1],
        }

    return positions or None


def _build_field_context(
    fields: list[dict],
    field_index: int,
    match: tuple[dict, int] | None = None,
    anchor_hints: list[float | None] | None = None,
) -> dict:
    field = fields[field_index]
    context = {
        "key": field.get("key"),
        "label": field.get("label"),
        "type": field.get("type"),
        "options": field.get("options") or [],
        "order": field.get("order", field_index + 1),
        "field_index": field_index,
        "previous_fields": [
            {
                "key": previous.get("key"),
                "label": previous.get("label"),
                "type": previous.get("type"),
                "order": previous.get("order"),
            }
            for previous in fields[max(0, field_index - 2) : field_index]
        ],
        "next_fields": [
            {
                "key": following.get("key"),
                "label": following.get("label"),
                "type": following.get("type"),
                "order": following.get("order"),
            }
            for following in fields[field_index + 1 : field_index + 3]
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
    """尝试用每个栏位自己的 label 在 OCR 里找到对应行，
    记录该行的 y0 作为"锚点"，同时把匹配结果（line, end_idx）缓存下来，
    后续文本/日期/嵌入填空/默认值的定位都复用这一次匹配，避免重复调用模型。"""
    anchors = []
    matches = []
    anchor_hints = []
    for field in fields:
        candidates = _find_top_candidates((field["label"],), lines, top_k=1)
        anchor_hints.append(candidates[0][1]["box"][1] if candidates else None)

    for field_index, field in enumerate(fields):
        context = _build_field_context(fields, field_index, anchor_hints=anchor_hints)
        found = find_label_position(field["label"], lines, field_context=context)
        anchors.append(found[0]["box"][1] if found else None)
        matches.append(found)
    return anchors, matches


def extract_field_positions(fields_data: dict, ocr_data: dict) -> list[dict]:
    lines = flatten_ocr(ocr_data)
    fields = fields_data["fields"]

    anchors, matches = compute_field_anchors(fields, lines)

    results = []
    for field_index, (field, anchor, match) in enumerate(zip(fields, anchors, matches)):
        field_type = field["type"]
        label = field["label"]
        entry = {"key": field["key"], "label": label, "type": field_type}
        field_context = _build_field_context(fields, field_index, match)

        if field_type in ("SingleChoice", "MultiChoice"):
            options = field.get("options") or []
            entry["options"] = _resolve_option_positions(
                options,
                lines,
                field_context,
                anchor,
            )
            results.append(entry)
            continue

        if match is None:
            entry["position"] = None  # 找不到时记 None，方便发现问题
            results.append(entry)
            continue

        line, end_idx = match
        if field_type == "Date":
            date_positions = _compute_slash_date_positions(line, end_idx)
            if date_positions is None:
                date_positions = compute_date_field_positions(field, line, end_idx, lines)
            if date_positions:
                first_position = next(iter(date_positions.values()))
                entry["position"] = {key: first_position[key] for key in ("x", "y", "height")}
                entry["positions"] = date_positions
            else:
                entry["position"] = compute_text_field_position(line, end_idx)
        elif field_type == "Boolean":
            entry["position"] = compute_inline_field_position(
                field, line, end_idx, field_context
            )

            overwrite_position = compute_default_value_overwrite_position(field, line, end_idx)
            if overwrite_position is not None:
                entry["overwritePosition"] = overwrite_position
        else:
            has_colon = _find_colon_boundary(line, end_idx) is not None
            entry["position"] = (
                compute_text_field_position(line, end_idx)
                if has_colon
                else compute_inline_field_position(field, line, end_idx, field_context)
            )

            overwrite_position = compute_default_value_overwrite_position(field, line, end_idx)
            if overwrite_position is not None:
                entry["overwritePosition"] = overwrite_position

        results.append(entry)

    return results


def main():
    fields_data = json.loads(FIELD_PATH.read_text(encoding="utf-8"))
    ocr_data = json.loads(OCR_RESULT_PATH.read_text(encoding="utf-8"))

    positions = extract_field_positions(fields_data, ocr_data)

    out_path = FIELD_DIR / "llm_field_position.json"
    out_path.write_text(
        json.dumps(positions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"写入 {out_path}，共 {len(positions)} 个栏位")


if __name__ == "__main__":
    main()
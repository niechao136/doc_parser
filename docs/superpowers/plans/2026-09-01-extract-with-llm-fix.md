# Extract With LLM Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the four null field positions, support all observed date layouts and label-adjacent blanks, and remove the structured-output serializer warning without field-specific hardcoded rules.

**Architecture:** Keep fuzzy matching as candidate generation, but preserve short OCR candidates and send uncertain placement decisions to the local LLM. Represent date markers and inline blank boundaries as indexes into compact OCR token contexts; calculate final coordinates only from the selected token boundaries. Use JSON-mode structured output so the LangChain response does not carry Pydantic `parsed` objects through the OpenAI response serializer.

**Tech Stack:** Python 3.11+, LangChain OpenAI, Pydantic, unittest, OpenCC.

## Global Constraints

- Do not add field-key-specific aliases, layouts, search boundaries, or coordinates.
- Send uncertain matching and placement decisions to the local LLM.
- Preserve the existing output shape (`position`, optional `positions`, and `options`).
- Keep OCR coordinates and deterministic geometry calculations outside the LLM response.
- Validate with focused regression tests, the existing test suite, and a real extractor run when the local model is available.

---

### Task 1: Add Regression Tests

**Files:**
- Create: `tests/test_extract_with_llm.py`

**Interfaces:**
- Tests `src.field.extract_with_llm.find_label_position`, `compute_inline_field_position`, `compute_date_field_positions`, `_call_llm`, and `extract_field_positions` behavior with deterministic fake LLM responses.

- [ ] **Step 1: Write failing tests**

Add tests for these behaviors:

```python
# A target that is one OCR character longer must remain a candidate for LLM review.
# An inline value may occupy the gap immediately before a suffix token.
# A Boolean control may occupy the gap before the label.
# Date markers may be found on the two OCR lines following the label.
# Date fields without markers fall back to normal colon-based positions.
# Structured calls use JSON mode and do not request raw response serialization.
```

- [ ] **Step 2: Run focused tests to verify the failure**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_extract_with_llm -v`

Expected: FAIL because the current candidate search drops the short OCR label, inline placement rejects a null previous token, date context is limited to one line, and structured-output transport does not select JSON mode.

### Task 2: Implement Generic Matching and Placement

**Files:**
- Modify: `src/field/extract_with_llm.py`

**Interfaces:**
- Preserve public extraction entry points and JSON output keys.
- Extend date placement to accept the flattened OCR lines needed for adjacent-line marker resolution.
- Keep `InlineMarkersResult` boundary indexes nullable so line-start and line-end gaps are representable.

- [ ] **Step 1: Preserve incomplete OCR candidates**

When a token sequence ends before reaching the normalized target length, score and retain its complete sequence as a candidate. Keep the existing top-k, deduplication, and LLM threshold behavior.

- [ ] **Step 2: Make uncertain inline placement boundary-aware**

Send the matched line's token sequence and field type to the LLM. Interpret `(previous_token, next_token)` as a gap; allow either side to be null for a line boundary. Use the selected gap's deterministic x coordinates and the line's y/height.

- [ ] **Step 3: Resolve date markers across a compact adjacent-line context**

Flatten the matched line's remaining tokens plus a bounded number of following OCR lines into one numbered context. Map each model-selected marker index back to its source line/token before calculating positions. If no markers are returned, use the existing colon/text fallback so `日期：/` and ordinary colon dates remain valid.

- [ ] **Step 4: Remove the serializer warning at the transport boundary**

Use `with_structured_output(schema, method="json_mode", include_raw=False)` and add concise JSON-only format instructions derived from the schema. Continue validating the parsed result against the requested Pydantic schema.

- [ ] **Step 5: Run the focused regression tests**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_extract_with_llm -v`

Expected: PASS with no new warnings.

### Task 3: Validate Existing Behavior and Real Output

**Files:**
- Modify: `output/field/llm_field_position.json` only through the extractor run if generated output changes.

- [ ] **Step 1: Run the existing tests**

Run: `.venv\\Scripts\\python.exe -m unittest discover -s tests -v`

Expected: Existing tests pass; unrelated legacy extractor fixtures remain unchanged.

- [ ] **Step 2: Run the LLM extractor**

Run: `.venv\\Scripts\\python.exe -m src.field.extract_with_llm`

Expected: The command exits successfully, emits no Pydantic serializer warning, and the four previously null entries have non-null positions.

- [ ] **Step 3: Check the generated result and syntax diagnostics**

Run: `.venv\\Scripts\\python.exe -m py_compile src/field/extract_with_llm.py`

Inspect the four field keys and all Date entries in `output/field/llm_field_position.json`; verify no field-specific rule table was introduced.

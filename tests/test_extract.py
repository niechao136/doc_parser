import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.field import extract


class ExtractTests(unittest.TestCase):
    @staticmethod
    def make_line(line_idx, tokens, token_boxes, box=None):
        if box is None:
            box = [token_boxes[0][0], token_boxes[0][1], token_boxes[-1][2], token_boxes[-1][3]]
        return {
            "line_idx": line_idx,
            "text": "".join(tokens),
            "box": box,
            "tokens": tokens,
            "token_boxes": token_boxes,
        }

    def test_date_markers_in_adjacent_ocr_blocks_are_resolved(self):
        lines = [
            self.make_line(
                0,
                ["出", "院", "日", "期"],
                [[46, 76, 63, 94], [61, 76, 78, 94], [76, 76, 93, 94], [94, 76, 109, 94]],
            ),
            self.make_line(
                1,
                ["年", "月"],
                [[210, 75, 247, 93], [244, 75, 280, 93]],
            ),
            self.make_line(2, ["日"], [[314, 76, 332, 93]]),
            self.make_line(3, ["出", "院", "方", "式"], [[43, 103, 63, 124], [61, 103, 77, 124], [78, 103, 94, 124], [92, 103, 109, 124]]),
        ]
        field = {"key": "date", "label": "出院日期", "type": "Date"}
        marker_result = extract.DateMarkersResult(
            year_marker_idx=0,
            month_marker_idx=1,
            day_marker_idx=2,
        )

        with patch.object(extract, "_llm_find_date_markers", return_value=marker_result):
            result = extract.extract_field_positions(
                {"fields": [field]},
                {
                    "rec_texts": [line["text"] for line in lines],
                    "rec_boxes": [line["box"] for line in lines],
                    "text_word": [line["tokens"] for line in lines],
                    "text_word_boxes": [line["token_boxes"] for line in lines],
                },
            )[0]

        self.assertEqual(
            extract.classify_value_region(field, lines[0], 3, lines),
            [extract.ValueRegionType.GLYPH_DATE_PART],
        )
        _, references, _ = extract._build_date_context(lines, lines[0], 3)
        self.assertEqual(
            [(line["line_idx"], token_idx) for line, token_idx in references],
            [(1, 0), (1, 1), (2, 0)],
        )
        self.assertEqual(set(result["positions"]), {"year", "month", "day"})
        self.assertEqual(result["position"]["y"], 76)

    def test_number_fields_prefer_input_gap_inside_label(self):
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        lines = extract.flatten_ocr(ocr_data)
        expected_x = {"dressing_frequency": 237, "drain_tube_count": 488}

        for field_key, x in expected_x.items():
            with self.subTest(field_key=field_key):
                field = next(item for item in fields_data["fields"] if item["key"] == field_key)
                match = extract.find_label_position(field["label"], lines)
                if match is None:
                    self.fail(f"label not found: {field['label']}")
                line, end_idx, *_ = match
                with patch.object(
                    extract,
                    "_llm_find_inline_markers",
                    return_value=SimpleNamespace(candidate_idx=0),
                ):
                    position = extract.compute_inline_field_position(field, line, end_idx)

                if position is None:
                    self.fail(f"position not found: {field_key}")
                self.assertEqual(position["x"], x)

    def test_repeated_option_is_scoped_to_the_field_section(self):
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        fields = fields_data["fields"]
        field_index = next(index for index, field in enumerate(fields) if field["key"] == "health_care_needs")
        field = fields[field_index]
        lines = extract.flatten_ocr(ocr_data)
        anchors, matches = extract.compute_field_anchors(fields, lines)
        context = extract._build_field_context(fields, field_index, matches[field_index], anchor_hints=anchors)
        per_option, _ = extract._collect_option_candidates(field["options"], lines, anchors[field_index])
        other_option = field["options"][4]

        self.assertEqual(
            {(candidate[1]["line_idx"], candidate[1]["box"][1]) for candidate in per_option[4]},
            {(57, 814)},
        )
        with patch.object(extract, "_llm_resolve_option_group", return_value=None):
            result = extract.extract_field_positions({"fields": [field]}, ocr_data)[0]

        self.assertEqual(result["options"][other_option]["y"], 814)
        self.assertEqual(result["options"][field["options"][0]]["x"], 156)

    def test_option_on_last_nearby_ocr_block_keeps_its_checkbox(self):
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        field = next(
            item
            for item in fields_data["fields"]
            if item["key"] == "catheter_care_instruction"
        )
        lines = extract.flatten_ocr(ocr_data)
        anchors, _ = extract.compute_field_anchors(fields_data["fields"], lines)

        with patch.object(extract, "_llm_resolve_option_group", return_value=None):
            result = extract.extract_field_positions({"fields": [field]}, ocr_data)[0]

        self.assertEqual(
            result["options"][field["options"][-1]],
            {"x": 412, "y": 640, "width": 3, "height": 17},
        )

    def test_composite_slash_date_starts_at_the_first_input_line(self):
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        field = next(item for item in fields_data["fields"] if item["key"] == "record_date")
        result = extract.extract_field_positions({"fields": [field]}, ocr_data)[0]

        self.assertEqual(result["positions"]["month"]["x"], 381)
        self.assertEqual(result["positions"]["day"]["x"], 422)

    def test_colon_date_starts_after_the_colon(self):
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        field = next(
            item
            for item in fields_data["fields"]
            if item["key"] == "catheter_next_change_date"
        )
        result = extract.extract_field_positions({"fields": [field]}, ocr_data)[0]

        self.assertEqual(result["position"], {"x": 311, "y": 578, "height": 17})
        self.assertEqual(
            list(result["resolution"]),
            [extract.ValueRegionType.COLON_ANCHORED.value],
        )

    def test_stoma_text_uses_input_before_label_when_followed_by_comma(self):
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        field = next(item for item in fields_data["fields"] if item["key"] == "stoma_care_instruction")
        lines = extract.flatten_ocr(ocr_data)
        line = lines[45]

        self.assertEqual(
            extract.classify_value_region(field, line, 2, lines),
            [extract.ValueRegionType.INLINE_BEFORE_LABEL],
        )
        self.assertEqual(
            extract.compute_left_inline_field_position(field, line, 2, lines),
            {"x": 421, "y": 618, "height": 20},
        )

    def test_missing_option_checkbox_aligns_to_known_checkbox_column(self):
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        field = next(item for item in fields_data["fields"] if item["key"] == "catheter_care_instruction")
        lines = extract.flatten_ocr(ocr_data)
        fields = fields_data["fields"]
        field_index = next(index for index, item in enumerate(fields) if item["key"] == field["key"])
        anchors, _ = extract.compute_field_anchors(fields, lines)
        per_option, pooled = extract._collect_option_candidates(field["options"], lines, anchors[field_index])
        option_index = field["options"].index("造瘻口")
        candidate = next(item for item in per_option[option_index] if item[1]["line_idx"] == 45)

        self.assertEqual(
            extract._align_missing_option_box(candidate[1], candidate[2], pooled),
            {"x": 412, "y": 618, "width": 3, "height": 17},
        )

    def test_missing_option_size_uses_same_field_checkbox_size_without_column_alignment(self):
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        field = next(item for item in fields_data["fields"] if item["key"] == "catheter_care_instruction")

        with patch.object(extract, "_llm_resolve_option_group", return_value=None):
            result = extract.extract_field_positions({"fields": [field]}, ocr_data)[0]

        position = result["options"][field["options"][0]]
        resolution = result["optionsResolution"][field["options"][0]]
        self.assertEqual(position, {"x": 157, "y": 555, "width": 3, "height": 17})
        self.assertEqual(resolution["method"], "glyph_missing_size_inferred")

    def test_prefixed_options_are_positioned_from_their_suffixes(self):
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields_data = json.loads(
            (extract.FIELD_PATH.parent / "field_118.json").read_text(encoding="utf-8")
        )
        field = next(item for item in fields_data["fields"] if item["key"] == "discharge_living_status")

        with patch.object(extract, "_llm_resolve_option_group", side_effect=AssertionError("unexpected LLM call")):
            result = extract.extract_field_positions({"fields": [field]}, ocr_data)[0]

        self.assertEqual(
            {option: result["options"][option]["x"] for option in field["options"]},
            {
                "回家：獨居": 214,
                "回家：家人同住": 261,
                "回家：朋友同住": 340,
                "回家：其他": 420,
            },
        )

    def test_116_same_row_label_is_not_replaced_by_partial_match(self):
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields = fields_data["fields"]
        lines = extract.flatten_ocr(ocr_data)
        anchors, matches = extract.compute_field_anchors(fields, lines)

        special_index = next(index for index, field in enumerate(fields) if field["key"] == "special_medication_name")
        signature_index = next(index for index, field in enumerate(fields) if field["key"] == "patient_or_guardian_signature")

        self.assertEqual(anchors[special_index], 753)
        self.assertEqual(matches[special_index][0], lines[54])
        self.assertEqual(anchors[signature_index], 924)
        self.assertEqual(matches[signature_index][0], lines[62])

    def test_116_signature_position_uses_its_colon_after_neighboring_fields(self):
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields = fields_data["fields"]
        lines = extract.flatten_ocr(ocr_data)
        _, matches = extract.compute_field_anchors(fields, lines)
        signature_index = next(index for index, field in enumerate(fields) if field["key"] == "patient_or_guardian_signature")
        line, end_idx, *_ = matches[signature_index]

        self.assertEqual(
            extract.compute_text_field_position(line, end_idx),
            {"x": 160, "y": 924, "height": 17},
        )

    def test_medical_record_anchor_ignores_low_confidence_prior_labels(self):
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields = fields_data["fields"][:8]
        lines = extract.flatten_ocr(ocr_data)

        with patch.object(extract, "_llm_resolve_label", return_value=None):
            anchors, matches = extract.compute_field_anchors(fields, lines)

        medical_index = next(index for index, field in enumerate(fields) if field["key"] == "medical_record_number")
        self.assertEqual(anchors[medical_index], 34)
        self.assertEqual(matches[medical_index][0], lines[5])
        self.assertEqual(
            extract.compute_text_field_position(*matches[medical_index][:2]),
            {"x": 567, "y": 34, "height": 18},
        )

    def test_partial_label_uses_matching_row_inside_field_window(self):
        lines = [
            self.make_line(
                0,
                ["輔", "具", "使", "用", "指", "導"],
                [[163, 311, 180, 328], [180, 311, 197, 328], [197, 311, 214, 328], [214, 311, 231, 328], [231, 311, 248, 328], [248, 311, 265, 328]],
            ),
            self.make_line(
                1,
                ["其", "它", "："],
                [[165, 331, 183, 349], [180, 331, 198, 349], [203, 331, 206, 349]],
            ),
            self.make_line(
                2,
                ["飲", "食"],
                [[4, 355, 27, 378], [24, 355, 47, 378]],
            ),
        ]
        field = {"key": "activity_ability_other", "label": "活動能力其他", "type": "Text"}

        result = extract.find_label_position(
            field["label"],
            lines,
            field_context={"previous_anchor_y": 311, "next_anchor_y": 355},
        )

        self.assertIsNotNone(result)
        self.assertEqual(result[0], lines[1])
        self.assertEqual(result[1:], (1, "bounded_partial", 0.5))
        self.assertEqual(
            extract.compute_text_field_position(result[0], result[1]),
            {"x": 216, "y": 331, "height": 18},
        )

    def test_anchor_hints_follow_field_order_for_partial_labels(self):
        lines = [
            self.make_line(
                0,
                ["活", "動", "能", "力"],
                [[7, 246, 27, 267], [28, 246, 48, 267], [45, 246, 65, 267], [63, 246, 81, 267]],
            ),
            self.make_line(
                1,
                ["輔", "具", "使", "用", "指", "導"],
                [[163, 311, 180, 328], [180, 311, 197, 328], [197, 311, 214, 328], [214, 311, 231, 328], [231, 311, 248, 328], [248, 311, 265, 328]],
            ),
            self.make_line(
                2,
                ["其", "它", "："],
                [[165, 331, 183, 349], [180, 331, 198, 349], [203, 331, 206, 349]],
            ),
            self.make_line(
                3,
                ["飲", "食", "指", "導"],
                [[4, 355, 27, 378], [24, 355, 47, 378], [48, 355, 65, 378], [64, 355, 81, 378]],
            ),
        ]
        fields = [
            {"key": "previous", "label": "輔具使用指導", "type": "Text"},
            {"key": "activity_ability_other", "label": "活動能力其他", "type": "Text"},
            {"key": "next", "label": "飲食指導", "type": "Text"},
        ]

        anchors, matches = extract.compute_field_anchors(fields, lines)

        self.assertEqual(anchors, [311, 331, 355])
        self.assertEqual(matches[1][0], lines[2])
        self.assertEqual(matches[1][2], "bounded_partial")

    def test_full_transfer_text_starts_after_checkbox_label(self):
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        field = next(item for item in fields_data["fields"] if item["key"] == "transfer_hospital")
        lines = extract.flatten_ocr(ocr_data)
        candidate = extract._find_top_candidates((field["label"],), lines, top_k=1)[0]
        line, end_idx = candidate[1], candidate[3]

        self.assertEqual(
            extract.classify_value_region(field, line, end_idx, lines),
            [extract.ValueRegionType.CHECKBOX_LABEL_TEXT],
        )
        result = extract.TYPE_HANDLERS[extract.ValueRegionType.CHECKBOX_LABEL_TEXT](
            field, line, end_idx, lines, None
        )
        self.assertEqual(result.position, {"x": 366, "y": 103, "height": 19})

    def test_full_other_activity_label_uses_its_own_other_row(self):
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        fields = fields_data["fields"]
        lines = extract.flatten_ocr(ocr_data)
        field_index = next(
            index for index, field in enumerate(fields)
            if field["key"] == "other_activity_ability_instruction"
        )
        anchor_hints = [
            extract._find_confident_label_hint(field["label"], lines)
            for field in fields
        ]
        context = extract._build_field_context(fields, field_index, anchor_hints=anchor_hints)
        result = extract.find_label_position(
            fields[field_index]["label"], lines, field_context=context
        )

        self.assertIsNotNone(result)
        self.assertEqual(result[0]["line_idx"], 25)
        self.assertEqual(
            extract.compute_text_field_position(result[0], result[1]),
            {"x": 216, "y": 331, "height": 18},
        )

    def test_full_tube_number_fields_use_structural_input_gap(self):
        fields_data = json.loads(extract.FIELD_PATH.read_text(encoding="utf-8"))
        ocr_data = json.loads(extract.OCR_RESULT_PATH.read_text(encoding="utf-8"))
        lines = extract.flatten_ocr(ocr_data)
        expected = {
            "nasogastric_tube_number": (41, 477, 577),
            "urinary_catheter_number": (43, 477, 599),
        }

        for key, (line_idx, expected_x, expected_y) in expected.items():
            with self.subTest(key=key):
                field = next(item for item in fields_data["fields"] if item["key"] == key)
                context = {"previous_anchor_y": 578, "next_anchor_y": 663}
                match = extract.find_label_position(field["label"], lines, field_context=context)

                self.assertIsNotNone(match)
                self.assertEqual(match[0]["line_idx"], line_idx)
                region_type = extract.classify_value_region(field, match[0], match[1], lines)
                self.assertEqual(region_type, [extract.ValueRegionType.STRUCTURAL_INPUT_GAP])
                result = extract.TYPE_HANDLERS[extract.ValueRegionType.STRUCTURAL_INPUT_GAP](
                    field, match[0], match[1], lines, context
                )
                self.assertEqual(result.position, {"x": expected_x, "y": expected_y, "height": 17})


if __name__ == "__main__":
    unittest.main()
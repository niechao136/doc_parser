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


if __name__ == "__main__":
    unittest.main()
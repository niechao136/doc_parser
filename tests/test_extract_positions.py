import json
import unittest

from src.field.extract_positions import (
    FIELD_PATH,
    OCR_RESULT_PATH,
    compute_text_field_position,
    extract_field_positions,
)


class ExtractFieldPositionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fields_data = json.loads(FIELD_PATH.read_text(encoding="utf-8"))
        ocr_data = json.loads(OCR_RESULT_PATH.read_text(encoding="utf-8"))
        cls.positions = {
            entry["fieldKey"]: entry
            for entry in extract_field_positions(fields_data, ocr_data)
        }

    def test_unprinted_detail_fields_use_their_instruction_line(self):
        self.assertEqual(
            self.positions["wound_care_details"]["position"],
            {"x": 384, "y": 467, "height": 17},
        )
        self.assertEqual(
            self.positions["tube_specific_info"]["position"],
            {"x": 360, "y": 555, "height": 17},
        )

    def test_text_field_position_leaves_gap_after_colon(self):
        expected = {
            "filler_name": {"x": 400, "y": 11, "height": 21},
            "patient_name": {"x": 534, "y": 12, "height": 19},
            "ward_number": {"x": 567, "y": 34, "height": 18},
            "bed_number": {"x": 534, "y": 52, "height": 22},
            "patient_age": {"x": 646, "y": 53, "height": 19},
            "discharge_time": {"x": 385, "y": 53, "height": 19},
            "wound_location": {"x": 485, "y": 488, "height": 17},
            "special_medication_name": {"x": 453, "y": 753, "height": 17},
            "patient_or_guardian_signature": {"x": 160, "y": 924, "height": 17},
            "relationship": {"x": 439, "y": 921, "height": 21},
        }

        for field_key, position in expected.items():
            with self.subTest(field_key=field_key):
                self.assertEqual(self.positions[field_key]["position"], position)

    def test_inline_number_fields_start_at_their_middle_underline(self):
        self.assertEqual(
            self.positions["dressing_frequency"]["position"],
            {"x": 237, "y": 488, "height": 17},
        )
        self.assertEqual(
            self.positions["drain_tube_count"]["position"],
            {"x": 488, "y": 507, "height": 20},
        )

    def test_relationship_position_starts_at_default_value(self):
        self.assertEqual(
            self.positions["relationship"]["position"],
            {"x": 439, "y": 921, "height": 21},
        )
        self.assertEqual(
            self.positions["relationship"]["overwritePosition"],
            {"x": 441, "y": 921, "width": 39, "height": 21},
        )
        self.assertNotIn("overwritePosition", self.positions["filler_name"])

    def test_default_value_overwrite_position_has_a_bounded_width(self):
        relationship = self.positions["relationship"]

        self.assertEqual(
            relationship["overwritePosition"]["x"]
            + relationship["overwritePosition"]["width"],
            480,
        )

    def test_text_field_position_handles_colon_inside_one_ocr_token(self):
        lines = [
            {
                "line_idx": 0,
                "tokens": ["姓名：张三"],
                "token_boxes": [[10, 20, 60, 40]],
                "box": [10, 20, 60, 40],
            }
        ]

        self.assertEqual(
            compute_text_field_position("姓名", lines),
            {"x": 50, "y": 20, "height": 20},
        )

    def test_discharge_date_exposes_three_input_positions(self):
        discharge_date = self.positions["discharge_date"]

        self.assertEqual(
            discharge_date["position"],
            {"x": 160, "y": 76, "height": 18},
        )
        self.assertEqual(
            discharge_date["positions"],
            {
                "year": {"x": 160, "y": 76, "width": 64, "height": 18},
                "month": {"x": 240, "y": 76, "width": 30, "height": 18},
                "day": {"x": 287, "y": 76, "width": 39, "height": 18},
            },
        )

    def test_ocr_aliases_resolve_missing_options(self):
        expected = {
            ("primary_caregiver", "外傭"): {"x": 567, "y": 167, "width": 3, "height": 17},
            ("dietary_requirements", "流質"): {"x": 284, "y": 369, "width": 2, "height": 14},
            ("dietary_requirements", "低鈉"): {"x": 443, "y": 369, "width": 3, "height": 14},
        }

        for (field_key, option), position in expected.items():
            with self.subTest(field_key=field_key, option=option):
                self.assertEqual(
                    self.positions[field_key]["options"][option], position
                )

    def test_options_are_found_across_their_full_section(self):
        expected = {
            ("wound_care_instructions", "每天換藥"): {"x": 146, "y": 488, "width": 17, "height": 17},
            ("wound_care_instructions", "換藥方式"): {"x": 173, "y": 509, "width": 3, "height": 17},
            ("wound_care_instructions", "有引流管"): {"x": 410, "y": 507, "width": 3, "height": 20},
            ("wound_care_instructions", "其他"): {"x": 147, "y": 529, "width": 21, "height": 21},
            ("health_care_needs", "緊急就醫狀況"): {"x": 149, "y": 774, "width": 16, "height": 16},
            ("health_care_needs", "給予預約回診單"): {"x": 159, "y": 836, "width": 3, "height": 20},
            ("health_care_needs", "其他"): {"x": 144, "y": 814, "width": 21, "height": 21},
            ("tube_care", "其他"): {"x": 147, "y": 638, "width": 21, "height": 21},
            ("institution_type", "其他"): {"x": 474, "y": 196, "width": 3, "height": 17},
            ("primary_caregiver", "朋友"): {"x": 521, "y": 167, "width": 3, "height": 17},
            ("primary_caregiver", "機構"): {"x": 149, "y": 196, "width": 17, "height": 17},
        }

        for (field_key, option), position in expected.items():
            with self.subTest(field_key=field_key, option=option):
                self.assertEqual(
                    self.positions[field_key]["options"][option], position
                )


if __name__ == "__main__":
    unittest.main()
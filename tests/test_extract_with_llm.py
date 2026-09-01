import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.field import extract_with_llm as extractor


class ExtractWithLlmTests(unittest.TestCase):
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

    @staticmethod
    def make_ocr(lines):
        return {
            "rec_texts": [line["text"] for line in lines],
            "rec_boxes": [line["box"] for line in lines],
            "text_word": [line["tokens"] for line in lines],
            "text_word_boxes": [line["token_boxes"] for line in lines],
        }

    def test_short_ocr_label_remains_available_for_matching(self):
        line = self.make_line(
            0,
            ["每", "天", "换", "藥", "次"],
            [
                [163, 488, 188, 505],
                [178, 488, 202, 505],
                [195, 488, 219, 505],
                [212, 488, 237, 505],
                [258, 488, 277, 505],
            ],
        )

        result = extractor.find_label_position("每天換藥次數", [line])

        self.assertIsNotNone(result)
        self.assertEqual(result[0], line)
        self.assertEqual(result[1], 4)

    @patch.object(
        extractor,
        "_llm_find_inline_markers",
        return_value=SimpleNamespace(
            candidate_idx=0,
            blank_start_idx=None,
            blank_end_idx=0,
        ),
    )
    def test_inline_position_can_start_before_the_label(
        self, mocked_find_inline_markers
    ):
        line = self.make_line(
            0,
            ["造", "癀", "口"],
            [
                [466, 618, 484, 638],
                [483, 618, 501, 638],
                [499, 618, 518, 638],
            ],
            box=[446, 618, 518, 638],
        )
        field = {"label": "造瘻口", "type": "Text"}

        result = extractor.compute_inline_field_position(field, line, 2)

        self.assertEqual(result, {"x": 446, "y": 618, "height": 20})
        mocked_find_inline_markers.assert_called_once()

    @patch.object(
        extractor,
        "_llm_find_inline_markers",
        return_value=SimpleNamespace(
            candidate_idx=0,
            blank_start_idx=3,
            blank_end_idx=4,
        ),
    )
    def test_inline_prompt_lists_positive_width_blank_candidates(
        self, mocked_find_inline_markers
    ):
        line = self.make_line(
            0,
            ["每", "天", "换", "藥", "次"],
            [
                [163, 488, 188, 505],
                [178, 488, 202, 505],
                [195, 488, 219, 505],
                [212, 488, 237, 505],
                [258, 488, 277, 505],
            ],
        )
        field = {"label": "每天換藥次數", "type": "Number"}

        extractor.compute_inline_field_position(field, line, 4)

        numbered_context = mocked_find_inline_markers.call_args.args[1]
        self.assertIn("candidate 1: [3] -> [4]", numbered_context)
        self.assertIn("width=21", numbered_context)

    @patch.object(
        extractor,
        "_llm_find_inline_markers",
        return_value=SimpleNamespace(
            candidate_idx=0,
            blank_start_idx=4,
            blank_end_idx=5,
        ),
    )
    def test_inline_prompt_identifies_the_approximate_label_range(
        self, mocked_find_inline_markers
    ):
        line = self.make_line(
            0,
            ["□", "有", "引", "流", "管", "條"],
            [
                [410, 507, 413, 527],
                [416, 507, 441, 527],
                [432, 507, 458, 527],
                [449, 507, 475, 527],
                [463, 507, 488, 527],
                [514, 507, 532, 527],
            ],
        )
        field = {"label": "引流管條數", "type": "Number"}

        extractor.compute_inline_field_position(field, line, 5)

        numbered_context = mocked_find_inline_markers.call_args.args[1]
        self.assertIn("Approximate matched label span: [2]-[5]", numbered_context)
        self.assertEqual(mocked_find_inline_markers.call_args.args[2], "Number")
        self.assertNotIn("relation=before-label", numbered_context)

    def test_inline_llm_prompt_requires_a_choice_when_one_candidate_exists(self):
        with patch.object(
            extractor,
            "_call_llm",
            return_value=extractor.InlineMarkersResult(candidate_idx=0),
        ) as mocked_call_llm:
            extractor._llm_find_inline_markers(
                "每天換藥次數",
                "candidate 0: [3] -> [4] relation=inside-label width=21",
                "Number",
            )

        system_prompt, user_prompt, schema = mocked_call_llm.call_args.args
        self.assertIn("If exactly one candidate is listed, return candidate_idx=0", system_prompt)
        self.assertIn("Number or Text", system_prompt)
        self.assertIn("candidate 0", user_prompt)
        self.assertIs(schema, extractor.InlineMarkersResult)

    def test_date_markers_can_be_resolved_on_following_ocr_lines(self):
        lines = [
            self.make_line(
                0,
                ["出", "院", "日", "期"],
                [
                    [46, 76, 63, 94],
                    [61, 76, 78, 94],
                    [76, 76, 93, 94],
                    [94, 76, 109, 94],
                ],
            ),
            self.make_line(
                1,
                ["年", "月"],
                [[210, 75, 247, 93], [244, 75, 280, 93]],
            ),
            self.make_line(2, ["日"], [[314, 76, 332, 93]]),
        ]
        fields_data = {
            "fields": [{"key": "date", "label": "出院日期", "type": "Date"}]
        }
        marker_result = extractor.DateMarkersResult(
            year_marker_idx=0,
            month_marker_idx=1,
            day_marker_idx=2,
        )

        with patch.object(
            extractor, "_llm_find_date_markers", return_value=marker_result
        ):
            result = extractor.extract_field_positions(fields_data, self.make_ocr(lines))

        entry = result[0]
        self.assertEqual(set(entry["positions"]), {"year", "month", "day"})
        self.assertEqual(entry["position"]["y"], 76)
        self.assertTrue(all(position["height"] == 18 for position in entry["positions"].values()))

    def test_date_slash_layout_falls_back_to_colon_position(self):
        line = self.make_line(
            0,
            ["日", "期", "：/"],
            [[334, 34, 352, 52], [349, 34, 367, 52], [372, 34, 420, 52]],
        )
        fields_data = {
            "fields": [{"key": "date", "label": "日期", "type": "Date"}]
        }

        with patch.object(
            extractor,
            "_llm_find_date_markers",
            return_value=extractor.DateMarkersResult(
                year_marker_idx=None,
                month_marker_idx=None,
                day_marker_idx=None,
            ),
        ):
            result = extractor.extract_field_positions(fields_data, self.make_ocr([line]))

        self.assertEqual(set(result[0]["positions"]), {"month", "day"})

    def test_normal_colon_date_layout_falls_back_to_text_position(self):
        line = self.make_line(
            0,
            ["管", "路", "下", "次", "更", "换", "日", "期", "："],
            [
                [167, 578, 183, 595],
                [181, 578, 198, 595],
                [198, 578, 214, 595],
                [215, 578, 231, 595],
                [229, 578, 245, 595],
                [246, 578, 262, 595],
                [260, 578, 276, 595],
                [277, 578, 293, 595],
                [298, 578, 301, 595],
            ],
        )
        fields_data = {
            "fields": [
                {
                    "key": "date",
                    "label": "管路下次更换日期",
                    "type": "Date",
                }
            ]
        }

        with patch.object(
            extractor,
            "_llm_find_date_markers",
            return_value=extractor.DateMarkersResult(
                year_marker_idx=None,
                month_marker_idx=None,
                day_marker_idx=None,
            ),
        ):
            result = extractor.extract_field_positions(fields_data, self.make_ocr([line]))

        self.assertEqual(result[0]["position"], {"x": 311, "y": 578, "height": 17})
        self.assertNotIn("positions", result[0])

    def test_structured_llm_call_uses_json_mode_without_raw_response(self):
        class StubStructuredLlm:
            messages = None

            def invoke(self, messages):
                self.messages = messages
                return extractor.LabelMatchResult(matched_idx=0)

        class StubClient:
            def model_copy(self, update):
                return self

            def with_structured_output(self, schema, **kwargs):
                self.schema = schema
                self.kwargs = kwargs
                return StubStructuredLlm()

        client = StubClient()
        with patch.object(extractor, "_llm_client", client):
            result = extractor._call_llm("system", "user", extractor.LabelMatchResult)

        self.assertEqual(result, extractor.LabelMatchResult(matched_idx=0))
        self.assertEqual(client.kwargs["method"], "json_mode")
        self.assertFalse(client.kwargs["include_raw"])

    def test_empty_ocr_line_does_not_break_candidate_search(self):
        line = {
            "line_idx": 0,
            "text": "",
            "box": [10, 20, 10, 20],
            "tokens": [],
            "token_boxes": [],
        }

        self.assertEqual(extractor._find_top_candidates(("標題",), [line]), [])

    def test_structured_llm_prompt_requests_instance_fields_not_schema_document(self):
        class StubStructuredLlm:
            messages = None

            def invoke(self, messages):
                self.messages = messages
                return extractor.LabelMatchResult(matched_idx=0)

        class StubClient:
            def model_copy(self, update):
                return self

            def with_structured_output(self, schema, **kwargs):
                self.structured = StubStructuredLlm()
                return self.structured

        client = StubClient()
        with patch.object(extractor, "_llm_client", client):
            extractor._call_llm("system", "user", extractor.LabelMatchResult)

        system_prompt = client.structured.messages[0]["content"]
        self.assertIn("matched_idx", system_prompt)
        self.assertNotIn('"properties"', system_prompt)
        self.assertNotIn("返回格式示例", system_prompt)

    def test_duplicate_label_candidates_are_resolved_with_field_context(self):
        lines = [
            self.make_line(0, ["其", "他", "："], [[10, 10, 20, 30], [20, 10, 30, 30], [30, 10, 32, 30]]),
            self.make_line(1, ["其", "他", "："], [[10, 50, 20, 70], [20, 50, 30, 70], [30, 50, 32, 70]]),
        ]

        with patch.object(extractor, "_llm_resolve_label", return_value=1) as resolve:
            result = extractor.find_label_position(
                "其他",
                lines,
                field_context={
                    "key": "other_instructions",
                    "type": "Text",
                    "order": 33,
                },
            )

        self.assertEqual(result[0], lines[1])
        self.assertEqual(resolve.call_args.args[0], "其他")
        self.assertEqual(resolve.call_args.args[2]["key"], "other_instructions")

    def test_option_matching_ignores_intervening_option_control_text(self):
        line = self.make_line(
            0,
            ["就", "○", "有", "○", "無", "氣", "囊"],
            [
                [529, 640, 546, 657],
                [553, 640, 556, 657],
                [560, 640, 578, 657],
                [585, 640, 588, 657],
                [595, 640, 612, 657],
                [609, 640, 627, 657],
                [627, 640, 640, 657],
            ],
        )

        def choose_first_option(option_text, candidates, field_context, lines):
            return next(index for index, candidate in enumerate(candidates) if candidate[2] == 2)

        with patch.object(extractor, "_llm_resolve_option", side_effect=choose_first_option):
            result = extractor.find_option_position(
                "有氣囊",
                [line],
                field_context={
                    "key": "colostomy_tube_type",
                    "label": "氯切管路",
                    "type": "SingleChoice",
                },
            )

        self.assertEqual(result, {"x": 553, "y": 640, "width": 3, "height": 17})

    def test_option_candidates_prefer_a_control_boundary_for_ocr_typos(self):
        line = self.make_line(
            0,
            ["□", "朋", "友", "□", "外", "備"],
            [
                [500, 167, 503, 184],
                [507, 167, 527, 184],
                [521, 167, 541, 184],
                [567, 167, 570, 184],
                [576, 167, 595, 184],
                [593, 167, 608, 184],
            ],
        )

        candidates = extractor._find_option_candidates("外傭", [line])

        self.assertEqual(candidates[0][2:], (4, 5))

    def test_boolean_field_with_trailing_colon_uses_control_area_before_label(self):
        line = self.make_line(
            0,
            ["特", "殊", "藥", "物", "指", "導", "，", "藥", "名", "："],
            [
                [294, 753, 311, 770],
                [308, 753, 325, 770],
                [325, 753, 342, 770],
                [340, 753, 356, 770],
                [357, 753, 373, 770],
                [371, 753, 387, 770],
                [395, 753, 398, 770],
                [405, 753, 421, 770],
                [419, 753, 436, 770],
                [440, 753, 443, 770],
            ],
            box=[276, 753, 449, 770],
        )
        field = {"label": "特殊藥物指導", "type": "Boolean"}

        with patch.object(
            extractor,
            "_llm_find_inline_markers",
            return_value=SimpleNamespace(candidate_idx=0),
        ):
            result = extractor.extract_field_positions(
                {"fields": [{"key": "special", **field}]}, self.make_ocr([line])
            )

        self.assertEqual(result[0]["position"], {"x": 276, "y": 753, "height": 17})

    def test_text_field_can_use_virtual_area_before_label(self):
        line = self.make_line(
            0,
            ["造", "癀", "口", "，", "號"],
            [
                [466, 618, 484, 638],
                [483, 618, 501, 638],
                [499, 618, 518, 638],
                [520, 618, 524, 638],
                [563, 618, 581, 638],
            ],
            box=[464, 618, 581, 638],
        )
        field = {"label": "造瘻口", "type": "Text"}

        with patch.object(
            extractor,
            "_llm_find_inline_markers",
            return_value=SimpleNamespace(candidate_idx=0),
        ):
            result = extractor.compute_inline_field_position(field, line, 2)

        self.assertEqual(result, {"x": 446, "y": 618, "height": 20})

    def test_date_markers_use_input_area_before_marker_tokens(self):
        lines = [
            self.make_line(
                0,
                ["出", "院", "日", "期"],
                [[46, 76, 63, 94], [61, 76, 78, 94], [76, 76, 93, 94], [94, 76, 109, 94]],
            ),
            self.make_line(
                1,
                ["年", "月"],
                [[140, 75, 160, 93], [180, 75, 200, 93]],
                box=[100, 75, 200, 93],
            ),
            self.make_line(2, ["日"], [[260, 76, 280, 94]], box=[220, 76, 300, 94]),
        ]
        field = {"key": "date", "label": "出院日期", "type": "Date"}

        with patch.object(
            extractor,
            "_llm_find_date_markers",
            return_value=extractor.DateMarkersResult(
                year_marker_idx=0,
                month_marker_idx=1,
                day_marker_idx=2,
                year_input_idx=0,
                month_input_idx=1,
                day_input_idx=2,
            ),
        ):
            result = extractor.compute_date_field_positions(field, lines[0], 3, lines)

        self.assertEqual(
            result,
            {
                "year": {"x": 100, "y": 76, "width": 40, "height": 18},
                "month": {"x": 160, "y": 76, "width": 20, "height": 18},
                "day": {"x": 220, "y": 76, "width": 40, "height": 18},
            },
        )

    def test_slash_count_one_produces_month_and_day_positions(self):
        line = self.make_line(
            0,
            ["日", "期", "：/"],
            [[334, 34, 352, 52], [349, 34, 367, 52], [372, 34, 420, 52]],
            box=[333, 34, 425, 52],
        )

        with patch.object(
            extractor,
            "_llm_find_date_markers",
            return_value=extractor.DateMarkersResult(
                year_marker_idx=None,
                month_marker_idx=None,
                day_marker_idx=None,
            ),
        ):
            result = extractor.extract_field_positions(
                {"fields": [{"key": "date", "label": "日期", "type": "Date"}]},
                self.make_ocr([line]),
            )[0]

        self.assertEqual(set(result["positions"]), {"month", "day"})

    def test_slash_count_two_produces_year_month_and_day_positions(self):
        line = self.make_line(
            0,
            ["日", "期", "：//"],
            [[334, 34, 352, 52], [349, 34, 367, 52], [372, 34, 444, 52]],
            box=[333, 34, 449, 52],
        )

        with patch.object(
            extractor,
            "_llm_find_date_markers",
            return_value=extractor.DateMarkersResult(
                year_marker_idx=None,
                month_marker_idx=None,
                day_marker_idx=None,
            ),
        ):
            result = extractor.extract_field_positions(
                {"fields": [{"key": "date", "label": "日期", "type": "Date"}]},
                self.make_ocr([line]),
            )[0]

        self.assertEqual(set(result["positions"]), {"year", "month", "day"})

    def test_date_without_slash_uses_normal_text_position(self):
        line = self.make_line(
            0,
            ["日", "期", "："],
            [[334, 34, 352, 52], [349, 34, 367, 52], [372, 34, 375, 52]],
        )

        with patch.object(
            extractor,
            "_llm_find_date_markers",
            return_value=extractor.DateMarkersResult(
                year_marker_idx=None,
                month_marker_idx=None,
                day_marker_idx=None,
            ),
        ):
            result = extractor.extract_field_positions(
                {"fields": [{"key": "date", "label": "日期", "type": "Date"}]},
                self.make_ocr([line]),
            )[0]

        self.assertNotIn("positions", result)
        self.assertEqual(result["position"], {"x": 385, "y": 34, "height": 18})

    def test_duplicate_other_label_uses_field_order_window(self):
        lines = [
            self.make_line(0, ["其", "他", "："], [[10, 10, 20, 30], [20, 10, 30, 30], [30, 10, 32, 30]]),
            self.make_line(1, ["其", "他", "："], [[10, 50, 20, 70], [20, 50, 30, 70], [30, 50, 32, 70]]),
            self.make_line(2, ["其", "他", "："], [[10, 90, 20, 110], [20, 90, 30, 110], [30, 90, 32, 110]]),
        ]

        result = extractor.find_label_position(
            "其他",
            lines,
            field_context={
                "key": "other_instructions",
                "type": "Text",
                "order": 33,
                "previous_anchor_y": 30,
                "next_anchor_y": 90,
            },
        )

        self.assertIsNotNone(result)
        self.assertEqual(result[0], lines[1])

    def test_option_group_rejects_reused_or_foreign_assignments(self):
        line = self.make_line(
            0,
            ["□", "甲", "□", "乙"],
            [
                [10, 10, 13, 30],
                [16, 10, 26, 30],
                [30, 10, 33, 30],
                [36, 10, 46, 30],
            ],
        )
        options = ["甲", "乙"]
        context = {"key": "choices", "label": "選項", "type": "MultiChoice"}

        original_collect = extractor._collect_option_candidates
        with patch.object(
            extractor,
            "_collect_option_candidates",
            return_value=original_collect(options, [line], 10),
        ), patch.object(
            extractor,
            "_llm_resolve_option_group",
            return_value=[0, 0],
        ):
            result = extractor._resolve_option_positions(options, [line], context, 10)

        self.assertEqual(result["甲"]["x"], 10)
        self.assertEqual(result["乙"]["x"], 30)

    def test_date_context_stops_after_contiguous_date_marker_lines(self):
        lines = [
            self.make_line(
                0,
                ["出", "院", "日", "期"],
                [[46, 76, 63, 94], [61, 76, 78, 94], [76, 76, 93, 94], [94, 76, 109, 94]],
            ),
            self.make_line(1, ["年", "月"], [[210, 75, 247, 93], [244, 75, 280, 93]]),
            self.make_line(2, ["日"], [[314, 76, 332, 93]]),
            self.make_line(
                3,
                ["出", "院", "方", "式"],
                [[47, 103, 63, 124], [61, 103, 77, 124], [78, 103, 94, 124], [92, 103, 108, 124]],
            ),
        ]

        _, references, _ = extractor._build_date_context(lines, lines[0], 3)

        self.assertEqual(
            [(line["line_idx"], token_idx) for line, token_idx in references],
            [(1, 0), (1, 1), (2, 0)],
        )

    def test_discharge_date_geometry_ignores_later_form_rows(self):
        lines = [
            self.make_line(
                0,
                ["出", "院", "日", "期"],
                [[46, 76, 63, 94], [61, 76, 78, 94], [76, 76, 93, 94], [94, 76, 109, 94]],
                box=[44, 76, 109, 94],
            ),
            self.make_line(
                1,
                ["年", "月"],
                [[210, 75, 247, 93], [244, 75, 280, 93]],
                box=[210, 75, 280, 93],
            ),
            self.make_line(2, ["日"], [[314, 76, 332, 93]], box=[310, 76, 332, 93]),
            self.make_line(3, ["回", "家"], [[165, 139, 185, 153], [182, 139, 201, 153]]),
            self.make_line(4, ["其", "他", "："], [[168, 814, 185, 835], [182, 814, 203, 835], [206, 814, 210, 835]]),
        ]
        field = {"key": "date", "label": "出院日期", "type": "Date"}
        marker_result = extractor.DateMarkersResult(
            year_marker_idx=0,
            month_marker_idx=1,
            day_marker_idx=2,
        )

        with patch.object(extractor, "_llm_find_date_markers", return_value=marker_result):
            result = extractor.compute_date_field_positions(field, lines[0], 3, lines)

        self.assertEqual(
            result,
            {
                "year": {"x": 160, "y": 76, "width": 64, "height": 18},
                "month": {"x": 240, "y": 76, "width": 30, "height": 18},
                "day": {"x": 287, "y": 76, "width": 39, "height": 18},
            },
        )

    def test_option_candidates_are_scoped_to_generic_field_section(self):
        lines = [
            self.make_line(0, ["□", "鼻", "胃", "管", "○", "一", "般"], [[412, 577, 415, 594], [418, 577, 437, 594], [435, 577, 454, 594], [449, 577, 468, 594], [545, 577, 548, 594], [551, 577, 570, 594], [569, 577, 587, 594]]),
            self.make_line(1, ["□", "導", "尿", "管", "○", "一", "般"], [[412, 599, 415, 616], [417, 599, 438, 616], [434, 599, 455, 616], [448, 599, 470, 616], [545, 599, 548, 616], [550, 599, 572, 616], [567, 599, 589, 616]]),
        ]

        _, pooled = extractor._collect_option_candidates(
            ["一般"],
            lines,
            preferred_y=599,
            field_context={"anchor_y": 599, "next_anchor_y": 618},
        )

        self.assertTrue(pooled)
        self.assertEqual({candidate[1]["line_idx"] for candidate in pooled}, {1})


if __name__ == "__main__":
    unittest.main()

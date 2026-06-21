import unittest

from app.demo_hybrid import resolve_dual_model_screening
from app.pipeline import build_screening_response_fields
from app.schemas import QualityReport, ScreeningResult


class DualModelConsistencyTests(unittest.TestCase):
    def test_non_referable_and_early_severity_remain_non_referable(self) -> None:
        result = resolve_dual_model_screening(False, 1)
        self.assertEqual(result["result"], "non_referable")
        self.assertEqual(result["label"], "Non-referable DR")

    def test_referable_and_referable_severity_remain_referable(self) -> None:
        result = resolve_dual_model_screening(True, 3)
        self.assertEqual(result["result"], "referable")
        self.assertEqual(result["label"], "Referable DR")

    def test_moderate_or_worse_severity_escalates_non_referable_screening(self) -> None:
        result = resolve_dual_model_screening(False, 2)
        self.assertEqual(result["result"], "referable_review")
        self.assertEqual(result["status"], "severity_escalation")
        self.assertTrue(result["referable"])

    def test_referable_screening_with_early_severity_requires_review(self) -> None:
        result = resolve_dual_model_screening(True, 0)
        self.assertEqual(result["result"], "referable_review")
        self.assertEqual(result["status"], "screening_severity_disagreement")
        self.assertEqual(
            result["recommendation"],
            "The screening and severity outputs require clinical confirmation.",
        )

    def test_low_quality_result_does_not_report_false_zero_probability(self) -> None:
        result = self.screening_result(model_type="dual_model_screening_hybrid_severity")
        fields = build_screening_response_fields(result, self.quality(False))
        self.assertEqual(fields["screening_result"], "uncertain")
        self.assertIsNone(fields["referable_probability"])
        self.assertIsNone(fields["screening_confidence"])

    def test_unavailable_models_do_not_fall_back_to_non_referable(self) -> None:
        result = self.screening_result(model_type="dual_model_unavailable")
        fields = build_screening_response_fields(result, self.quality(True))
        self.assertEqual(fields["screening_result"], "uncertain")
        self.assertEqual(fields["screening_label"], "Analysis models unavailable")
        self.assertIsNone(fields["referable_probability"])

    @staticmethod
    def quality(is_acceptable: bool) -> QualityReport:
        return QualityReport(
            is_acceptable=is_acceptable,
            blur_score=100.0,
            brightness_mean=100.0,
            contrast_std=30.0,
            fundus_area_ratio=0.8,
        )

    @staticmethod
    def screening_result(model_type: str) -> ScreeningResult:
        return ScreeningResult(
            classification="Unavailable",
            referable=False,
            dr_probability=0.0,
            stage=None,
            stage_label="Medical severity assessment unavailable",
            explanation="Required artifacts were not loaded.",
            recommendation="Check model artifacts.",
            reason="Test fixture",
            disclaimer="Screening support only.",
            model_type=model_type,
            probabilities={"Non-Referable": 1.0, "Referable": 0.0},
        )


if __name__ == "__main__":
    unittest.main()

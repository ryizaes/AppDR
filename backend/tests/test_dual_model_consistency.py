import unittest

from app.demo_hybrid import resolve_dual_model_screening


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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from rnsa_surrogate.roi_refiner import (
    VESSEL_CONTEXT_ORACLE,
    VESSEL_CONTEXT_STAGE1,
    validate_vessel_context_records,
)
from rnsa_surrogate.submission_contract import (
    input_interface,
    resolve_inference_amp,
    validate_task1_locations,
    validate_task2_array,
)


class SubmissionContractTests(unittest.TestCase):
    def test_only_one_supported_image_socket_is_accepted(self) -> None:
        self.assertEqual(
            input_interface([{"socket": {"slug": "head-ct-angiography"}}])[0],
            "ct",
        )
        with self.assertRaises(ValueError):
            input_interface([])
        with self.assertRaises(ValueError):
            input_interface([{"socket": {"slug": "vessel-mask"}}])

    def test_task1_labels_are_unique_in_range_integers(self) -> None:
        self.assertEqual(validate_task1_locations([1, np.int64(52)]), [1, 52])
        for invalid in ([0], [53], [1, 1], [True], [1.0]):
            with self.assertRaises((TypeError, ValueError)):
                validate_task1_locations(invalid)

    def test_task2_is_uint8_3d_and_geometry_matched(self) -> None:
        valid = np.zeros((3, 4, 5), dtype=np.uint8)
        self.assertIs(validate_task2_array(valid, (3, 4, 5)), valid)
        for invalid in (
            np.zeros((3, 4), dtype=np.uint8),
            np.zeros((3, 4, 5), dtype=np.int16),
            np.full((3, 4, 5), 53, dtype=np.uint8),
        ):
            with self.assertRaises((TypeError, ValueError)):
                validate_task2_array(invalid, (3, 4, 5))
        with self.assertRaises(ValueError):
            validate_task2_array(valid, (5, 4, 3))

    def test_amp_falls_back_on_t4(self) -> None:
        self.assertEqual(resolve_inference_amp("bf16", "cuda", False), "fp16")
        self.assertEqual(resolve_inference_amp("bf16", "cuda", True), "bf16")
        self.assertEqual(resolve_inference_amp("fp16", "cpu", False), "none")

    def test_stage1_vessel_provenance_is_required(self) -> None:
        stage1 = [{"candidate_id": "a", "vessel_context": True,
                   "vessel_context_source": VESSEL_CONTEXT_STAGE1}]
        self.assertEqual(
            validate_vessel_context_records(stage1, True),
            VESSEL_CONTEXT_STAGE1,
        )
        with self.assertRaises(ValueError):
            validate_vessel_context_records([{"candidate_id": "a"}], True)
        self.assertEqual(
            validate_vessel_context_records(
                [{"candidate_id": "a"}], True, allow_oracle=True
            ),
            VESSEL_CONTEXT_ORACLE,
        )


if __name__ == "__main__":
    unittest.main()

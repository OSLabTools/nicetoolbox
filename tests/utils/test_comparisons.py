import numpy as np
from fixtures import BaseTestCase

from nicetoolbox.utils import comparisons as cp


class TestComparisons(BaseTestCase):
    def test_compare_dict_keys(self):
        rtn1 = cp.compare_dict_keys({"a": 1, "b": 2}, {"a": 3, "b": 4})
        self.assertTrue(rtn1 == (True, {"a", "b"}))

        rtn2 = cp.compare_dict_keys({"a": []}, {"b": 4})
        self.assertTrue(rtn2 == (False, set()))

        rtn3 = cp.compare_dict_keys({"a": 1, "b": []}, {"b": {}})
        self.assertTrue(rtn3 == (False, {"b"}))

    def test_compare_numerical_nparrays(self):
        rtn1 = cp.compare_numerical_nparrays(np.ones((2, 3)), np.ones((2, 2)))
        self.assertFalse(rtn1)

        rtn2 = cp.compare_numerical_nparrays(np.array([1, 2, 3]), np.array([1, 2, 4]))
        self.assertFalse(rtn2)

        rtn3 = cp.compare_numerical_nparrays(np.array([1, 2, 3]), np.array([1, 2, 3]))
        self.assertTrue(rtn3)

        rtn4 = cp.compare_numerical_nparrays(
            np.array([[None, 2], [3, None]]), np.array([[None, 2], [3, None]])
        )
        self.assertTrue(rtn4)

    def test_compare_dicts_of_collections_of_strings(self):
        self.assertTrue(cp.compare_dicts_of_collections_of_strings({}, {}))

        rtn1 = cp.compare_dicts_of_collections_of_strings({"a": "b"}, {"a": "b"})
        self.assertTrue(rtn1)

        rtn2 = cp.compare_dicts_of_collections_of_strings({"a": "b"}, {"a": "c"})
        self.assertFalse(rtn2)

        rtn3 = cp.compare_dicts_of_collections_of_strings({"a": "b"}, {"b": "b"})
        self.assertFalse(rtn3)

        self.assertFalse(
            cp.compare_dicts_of_collections_of_strings({"a": "b"}, {"a": "b", "b": "c"})
        )

        self.assertTrue(
            cp.compare_dicts_of_collections_of_strings(
                {"a": ["x", "y"]}, {"a": ["x", "y"]}
            )
        )

    def test_compare_dicts_of_general_nparrays(self):
        self.assertTrue(
            cp.compare_dicts_of_general_nparrays(
                {("a", "b"): np.ones((2, 3))}, {("a", "b"): np.ones((2, 3))}
            )
        )

        self.assertFalse(
            cp.compare_dicts_of_general_nparrays(
                {"a": np.array([1, 2, 3])}, {("a",): np.array([1, 2, 3])}
            )
        )

        self.assertTrue(
            cp.compare_dicts_of_general_nparrays(
                {"a": np.array([1, "y", {3, 4, None}])},
                {"a": np.array([1, "y", {3, 4, None}])},
            )
        )

        self.assertFalse(
            cp.compare_dicts_of_general_nparrays(
                {"a": np.array({"x": 4})},
                {"a": np.array({"x": 4}), "b": np.array({"y": 5})},
            )
        )

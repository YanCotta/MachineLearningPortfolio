import unittest
import runpy


class TestClassificationScripts(unittest.TestCase):
    def test_run_scripts(self):
        scripts = [
            "src/logistic_regression.py",
            "src/kernel_svm.py",
            "src/naive_bayes.py",
            "src/random_forest_classification.py",
            "src/k_nearest_neighbors.py",
            "src/decision_tree_classification.py"
        ]
        for script in scripts:
            with self.subTest(script=script):
                try:
                    runpy.run_path(script, run_name="__main__")
                except Exception as e:
                    self.fail(f"Script {script} encountered an error: {e}")


if __name__ == "__main__":
    unittest.main()
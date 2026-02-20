import unittest
from mini_claude_codev4 import detect_tech_stack

class TestTeacherGuidelines(unittest.TestCase):
    
    def test_pytorch_detection(self):
        guidelines = detect_tech_stack("train a neural network model", [])
        self.assertIn("PYTORCH CRITICAL RULES", guidelines)
        self.assertIn("Device Safety", guidelines)
        self.assertNotIn("NUMPY/DATA RULES", guidelines)

    def test_numpy_detection(self):
        guidelines = detect_tech_stack("matrix multiplication with numpy", [])
        self.assertIn("NUMPY/DATA RULES", guidelines)
        self.assertIn("Check for `NaN`", guidelines)
        self.assertNotIn("PYTORCH CRITICAL RULES", guidelines)

    def test_general_python_detection(self):
        guidelines = detect_tech_stack("something", ["task.py"])
        self.assertIn("GENERAL PYTHON RULES", guidelines)
        self.assertIn("Reproducibility", guidelines)

    def test_api_safety_detection(self):
        # Triggers: train, evaluate, fit, predict
        guidelines = detect_tech_stack("train model", [])
        self.assertIn("API SAFETY RULES", guidelines)
        self.assertIn("Keyword Arguments", guidelines)
        # Should also trigger pytorch since "train" is in pytorch triggers too
        self.assertIn("PYTORCH CRITICAL RULES", guidelines)

    def test_no_detection(self):
        guidelines = detect_tech_stack("write a poem", [])
        self.assertEqual(guidelines, "")

if __name__ == '__main__':
    unittest.main()

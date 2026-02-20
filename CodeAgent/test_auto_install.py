import unittest
from unittest.mock import patch
from mini_claude_codev4 import _handle_missing_modules

class TestAutoInstall(unittest.TestCase):
    
    @patch('mini_claude_codev4.run_shell')
    def test_detect_sklearn(self, mock_run_shell):
        error_out = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'sklearn'"
        
        # Mock success installation
        mock_run_shell.return_value = (0, "Successfully installed scikit-learn")
        
        log = _handle_missing_modules(error_out)
        
        self.assertIn("Auto-Install: pip install --no-input scikit-learn", log)
        mock_run_shell.assert_called_with("pip install --no-input scikit-learn")

    @patch('mini_claude_codev4.run_shell')
    def test_detect_pil(self, mock_run_shell):
        error_out = "ImportError: No module named 'PIL'"
        mock_run_shell.return_value = (0, "Successfully installed Pillow")
        
        log = _handle_missing_modules(error_out)
        
        self.assertIn("Auto-Install: pip install --no-input Pillow", log)
        mock_run_shell.assert_called_with("pip install --no-input Pillow")

    def test_no_module_error(self):
        log = _handle_missing_modules("OtherError: something broke")
        self.assertIsNone(log)

if __name__ == '__main__':
    unittest.main()

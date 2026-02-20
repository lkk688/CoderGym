import unittest
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from mini_claude_codev4 import complete_with_continuation

# Mock OpenAI Structure
@dataclass
class MockMessage:
    content: str
@dataclass
class MockChoice:
    finish_reason: str
    message: MockMessage
@dataclass
class MockResponse:
    choices: List[MockChoice]

class MockClient:
    def __init__(self, responses: List[MockResponse]):
        self.responses = responses
        self.call_count = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        raise Exception("No more mocked responses")

class TestStitching(unittest.TestCase):

    def test_write_file_fence_stripping(self):
        """
        Scenario: Model starts a WRITE_FILE block, gets cut off mid-content (finish_reason='length').
        The continuation starts with a hallucinated ```python fence because the model thinks it needs one.
        We expect complete_with_continuation to detect we are inside WRITE_FILE and strip the fence.
        """
        
        # Chunk 1: Ends abruptly inside content
        chunk1 = MockResponse([MockChoice(
            finish_reason="length",
            message=MockMessage(content='WRITE_FILE: task.py\n<<<CONTENT\ndef hello():\n    print("Hello')
        )])
        
        # Chunk 2: Starts with hallucinated fence then continues code
        chunk2 = MockResponse([MockChoice(
            finish_reason="stop",
            message=MockMessage(content='```python\n")\nCONTENT>>>')
        )])
        
        client = MockClient([chunk1, chunk2])
        
        result = complete_with_continuation(
            client=client,
            model="mocklogger",
            messages=[],
            max_output_tokens=100
        )
        
        expected = 'WRITE_FILE: task.py\n<<<CONTENT\ndef hello():\n    print("Hello")\nCONTENT>>>'
        
        # Debug output if fail
        if result != expected:
            print(f"\nExpected:\n{repr(expected)}\nGot:\n{repr(result)}")
            
        self.assertEqual(result, expected)

    def test_normal_code_fence_stripping(self):
        """
        Scenario: Model is inside a normal ```python block. continued.
        """
        chunk1 = MockResponse([MockChoice(
            finish_reason="length",
            message=MockMessage(content='Here is code:\n```python\nx = 1\n')
        )])
        
        chunk2 = MockResponse([MockChoice(
            finish_reason="stop",
            message=MockMessage(content='```python\ny = 2\n```')
        )])
        
        client = MockClient([chunk1, chunk2])
        result = complete_with_continuation(client, "model", [], 100)
        
        expected = 'Here is code:\n```python\nx = 1\ny = 2\n```'
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()

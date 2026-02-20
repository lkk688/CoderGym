
import unittest
import tempfile
import json
import shutil
from pathlib import Path
from dataclasses import asdict

import sys
import os
sys.path.append(str(Path(__file__).parent.parent))

from CodeAgent.mini_claude_codev4 import (
    Skill, load_skills, score_skill, 
    select_relevant_skills, format_skill_injection
)

class TestSkillSystem(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.skill_dir = self.tmp_dir / "skilldb"
        self.skill_dir.mkdir()
        
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_load_mixed_skills(self):
        # Create legacy file
        legacy = {
            "tag": "legacy_tag", 
            "kind": "failure", 
            "text": "Goal: test legacy\n\nOutput: failed", 
            "pattern": "legacy",
            "evidence": "fail trace"
        }
        (self.skill_dir / "failures.jsonl").write_text(json.dumps(legacy) + "\n")
        
        # Create new file
        new_skill = Skill(
            category="PyTorch",
            pattern="tensor",
            insight="Check shapes",
            evidence="RuntimeError: shape mismatch",
            count=1,
            created_at=""
        )
        (self.skill_dir / "skills.jsonl").write_text(json.dumps(asdict(new_skill)) + "\n")
        
        skills = load_skills(self.skill_dir)
        self.assertEqual(len(skills), 2)
        
        # Check types
        s1 = next(s for s in skills if s.category == "Legacy")
        self.assertIn("Legacy failure", s1.insight)
        
        s2 = next(s for s in skills if s.category == "PyTorch")
        self.assertEqual(s2.insight, "Check shapes")

    def test_scoring(self):
        # High match: pattern match
        s1 = Skill("Cat", "pattern_match", "irrelevant info", "", 1)
        self.assertEqual(score_skill(s1, "use pattern_match here"), 5)
        
        # Medium match: insight keywords
        s2 = Skill("Cat", "none", "Check tensor shapes carefully", "", 1)
        self.assertEqual(score_skill(s2, "my tensor has wrong shape"), 1) # matches 'tensor' only
        # Insight words: check, tensor, shapes, carefully. Query: my, tensor, has, wrong, shape.
        # Overlap: tensor (1). Wait, 'shapes' vs 'shape'?
        # The scorer uses: re.findall(r"[a-zA-Z0-9_]{3,}", (skill.insight or "").lower())
        # insight words: check, tensor, shapes, carefully
        # query: my, tensor, has, wrong, shape
        # Hits: tensor. (shapes != shape). Score = 1.
        
        # Combined
        s3 = Skill("Cat", "pattern", "insight word", "", 1)
        self.assertEqual(score_skill(s3, "pattern with insight word"), 5 + 2) # 7
        
    def test_format_injection(self):
        skills = [
            Skill("PyTorch", "conv2d", "Check kernel size", "", 1),
            Skill("Syntax", "print", "Use flush=True", "", 1),
            Skill("PyTorch", "linear", "Check dimensions", "", 1)
        ]
        
        out = format_skill_injection(skills)
        print(f"\nInjection Output:\n{out}")
        
        self.assertIn("## Teacher Guidelines (From Experience)", out)
        self.assertIn("### PyTorch", out)
        self.assertIn("- [conv2d] Check kernel size", out)
        self.assertIn("- [linear] Check dimensions", out)
        self.assertIn("### Syntax", out)
        
if __name__ == "__main__":
    unittest.main()

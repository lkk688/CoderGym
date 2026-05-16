#!/usr/bin/env python
# Fix non-deterministic hash calls in Task 1

with open('MLtasks/tasks/nlp_lvl2_bigframe_embeddings/task.py', 'r') as f:
    lines = f.readlines()

# Write back with hash fixes
with open('MLtasks/tasks/nlp_lvl2_bigframe_embeddings/task.py', 'w') as f:
    for i, line in enumerate(lines):
        # Replace self._deterministic_hash with simple ord-based hash
        if 'feature_idx = (self._deterministic_hash(trigram,' in line:
            line = '            feature_idx = (sum(ord(c) for c in trigram) * 7919) % self.embedding_dim\n'
        elif 'feature_idx = (self._deterministic_hash(word,' in line:
            line = '            feature_idx = ((sum(ord(c) for c in word) * 7919) % (self.embedding_dim // 2)) + (self.embedding_dim // 2)\n'
        elif 'np.random.seed(self._deterministic_hash(text,' in line:
            line = '        np.random.seed((sum(ord(c) for c in text) * 7919) % (2**31))\n'
        # Remove broken _deterministic_hash method
        elif 'def _deterministic_hash(self' in line:
            # Skip this line and the next 3 lines
            lines[i] = '# REMOVED: _deterministic_hash method\n'
            if i + 3 < len(lines):
                for j in range(1, 4):
                    lines[i + j] = ''
        
        f.write(line)

print("Fixed hash functions in Task 1")

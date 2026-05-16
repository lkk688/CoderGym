import hashlib

# Read the file
with open('MLtasks/tasks/nlp_lvl2_bigframe_embeddings/task.py', 'r') as f:
    content = f.read()

# Replace hash() calls with deterministic hash
if 'import hashlib' not in content:
    content = content.replace('import random', 'import random\nimport hashlib')

# Add deterministic hash method after imports
if '_deterministic_hash' not in content:
    # Find the EmbeddingGenerator class and add the method
    class_start = content.find('class EmbeddingGenerator(nn.Module):')
    init_start = content.find('def __init__(self', class_start)
    indent_pos = init_start
    
    method_code = '''
    def _deterministic_hash(self, text: str, mod: int = 2**31) -> int:
        """Create deterministic hash from text."""
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % mod
    
    '''
    
    # Insert before __init__
    content = content[:init_start] + method_code + content[init_start:]

# Replace hash() calls with deterministic hash
content = content.replace('feature_idx = (hash(trigram) % self.embedding_dim)', 
                         'feature_idx = (self._deterministic_hash(trigram, self.embedding_dim))')
content = content.replace('feature_idx = (hash(word) % (self.embedding_dim // 2)) + (self.embedding_dim // 2)', 
                         'feature_idx = (self._deterministic_hash(word, self.embedding_dim // 2)) + (self.embedding_dim // 2)')
content = content.replace('np.random.seed(hash(text) % (2**31))', 
                         'np.random.seed(self._deterministic_hash(text, 2**31))')

# Write back
with open('MLtasks/tasks/nlp_lvl2_bigframe_embeddings/task.py', 'w') as f:
    f.write(content)

print('Updated to use deterministic hashing')

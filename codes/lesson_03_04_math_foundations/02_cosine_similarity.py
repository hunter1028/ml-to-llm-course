"""课时 03-04：用余弦相似度比较 query 和文档向量。"""

import numpy as np


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


q = np.array([1.0, 0.2, 0.1])
doc_a = np.array([0.9, 0.1, 0.2])
doc_b = np.array([0.1, 0.8, 0.3])

print("q vs doc_a:", cosine(q, doc_a))
print("q vs doc_b:", cosine(q, doc_b))

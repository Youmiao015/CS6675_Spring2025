# import faiss
# index = faiss.read_index("faiss_index.bin")
# print("Total vectors in index:", index.ntotal)


# import faiss
# index = faiss.read_index("faiss_index.bin")
# import numpy as np

# # Encode a simple query
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# q = model.encode(["machine learning"])
# distances, indices = index.search(np.array(q).astype('float32'), 5)
# print("Indices:", indices[0])
# print("Distances:", distances[0])


# import faiss, sqlite3, numpy as np
# from sentence_transformers import SentenceTransformer

# INDEX = "faiss_index.bin"
# DB    = "metadata.db"

# # Load index + embedding model
# index = faiss.read_index(INDEX)
# model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# # Perform a test search
# vec = model.encode(["machine learning"])
# distances, indices = index.search(np.array(vec).astype('float32'), 5)
# print("Returned FAISS indices:", indices[0])
# print("Returned distances:", distances[0])

# # Query SQLite for each index
# conn = sqlite3.connect(DB)
# cur = conn.cursor()
# for idx in indices[0]:
#     cur.execute("SELECT id, title FROM papers WHERE vector_idx = ?", (idx,))
#     print("vector_idx", idx, "→", cur.fetchone())

# conn.close()



import faiss, sqlite3, numpy as np
from sentence_transformers import SentenceTransformer

import os
print("CWD:", os.getcwd())
print("DB path:", os.path.abspath("/home/eau/projects/6675/data/metadata.db"))


index = faiss.read_index("/home/eau/projects/6675/data/faiss_index.bin")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
vec = model.encode(["machine learning"])
distances, indices = index.search(np.array(vec).astype('float32'), 5)

print("FAISS returned:", indices[0])
conn = sqlite3.connect("/home/eau/projects/6675/data/metadata.db")
cur = conn.cursor()

for idx in indices[0]:
    cur.execute("SELECT id, title FROM papers WHERE vector_idx = ?", (int(idx),))
    print("vector_idx", idx, "→", cur.fetchone())

conn.close()



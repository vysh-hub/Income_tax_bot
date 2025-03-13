import faiss
import numpy as np
import json

# ✅ Load Finance Bill JSON (to retrieve original text later)
with open("finance_bill_data.json", "r", encoding="utf-8") as f:
    finance_data = json.load(f)

# ✅ Load OpenAI Embeddings
embeddings = np.load("openai_embeddings.npy")

# ✅ Create FAISS Index
dimension = embeddings.shape[1]  # Must match OpenAI embedding size (1536)
index = faiss.IndexFlatL2(dimension)  # L2 Distance for similarity search
index.add(embeddings)  # Add embeddings to FAISS index

# ✅ Save FAISS Index
faiss.write_index(index, "finance_bill_index.faiss")

print("✅ FAISS index saved successfully!")

import faiss
import numpy as np
import json

# ✅ Load Old FAISS Index
old_index = faiss.read_index("finance_bill_index.faiss")

# ✅ Load Old and New JSON Files
with open("merged_data.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)  # This should contain all text content

# ✅ Load New Embeddings
new_embeddings = np.load("merged_embeddings.npy")

# ✅ Merge Old and New Embeddings into FAISS
dimension = new_embeddings.shape[1]
new_index = faiss.IndexFlatL2(dimension)
new_index.add(new_embeddings)

# ✅ Save Updated FAISS Index
faiss.write_index(new_index, "updated_finance_bill_index.faiss")

# ✅ Save Updated JSON Data (For Retrieval)
with open("updated_finance_bill_data.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=4)

print("✅ FAISS index and JSON data updated successfully!")


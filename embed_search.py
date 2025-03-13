import numpy as np
import openai
import json
# ✅ Load Merged Data
with open("merged_data.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)

# ✅ Extract Texts for Embeddings
texts = [entry["content"] for entry in all_data if "content" in entry and entry["content"].strip()]

# ✅ Function to Generate OpenAI Embeddings
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# ✅ Generate New Embeddings
new_embeddings = np.array([get_embedding(text) for text in texts])

# ✅ Save New Embeddings
np.save("merged_embeddings.npy", new_embeddings)

print("✅ New embeddings generated and saved!")


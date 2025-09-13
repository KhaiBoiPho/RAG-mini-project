from qdrant_client import QdrantClient
import yaml
import json

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)["qdrant"]

client = QdrantClient(host=cfg["host"], port=cfg["port"])

# Suppose you have pre-made embeddings
with open("sample_embeddings.json") as f:
    data = json.load(f)

points = [
    {"id": item["id"], "vector": item["embedding"], "payload": {"text": item["text"]}}
    for item in data
]

client.upsert(collection_name=cfg["collection_name"], points=points)
print("Data seeded successfully")
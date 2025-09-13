from qdrant_client import QdrantClient
import yaml

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)["qdrant"]

client = QdrantClient(host=cfg["host"], port=cfg["port"])

# Create collection if don't have
client.recreate_collection(
    collection_name=cfg["collection_name"],
    vector_size=cfg["vector_size"],
    distance=cfg["distance"]
)

print(f"Collection {cfg['collection_name']} is ready!")
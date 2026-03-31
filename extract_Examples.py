import json
from collections import defaultdict

# CONFIG
MODEL_OUTPUT_PATH = "model_outputs/fcoref_annotated_only.json"  # or fcoref_annotated_only.json
CORPUS_PATH = "data/NatSci_annotated.json"
MIN_CLUSTER_SIZE = 4
TARGET_ROLE = None  # e.g. "Hero", "Villain", or None for all

# LOAD DATA
with open(MODEL_OUTPUT_PATH) as f:
    model_output = json.load(f)

with open(CORPUS_PATH) as f:
    corpus = json.load(f)

corpus_lookup = {a["article_id"]: a for a in corpus}

# COLLECT CANDIDATE CLUSTERS
candidates = []

for article in model_output:
    art_id = article["article_id"]
    for cl in article["clusters"]:
        role = cl["role"]
        cluster = cl["cluster"]

        if TARGET_ROLE and role != TARGET_ROLE:
            continue

        if len(cluster) >= MIN_CLUSTER_SIZE:
            candidates.append({
                "article_id": art_id,
                "role": role,
                "cluster_size": len(cluster),
                "cluster": cluster
            })

# SORT BY CLUSTER SIZE (largest first)
candidates = sorted(candidates, key=lambda x: x["cluster_size"], reverse=True)

# PRINT TOP 10 EXAMPLES
for i, c in enumerate(candidates[:10]):
    print("=" * 80)
    print(f"Example {i+1}")
    print("Article:", c["article_id"])
    print("Role:", c["role"])
    print("Cluster size:", c["cluster_size"])
    print("Cluster mentions:")
    for m in c["cluster"]:
        print("  -", m)
    print()

#extract full article text for deeper inspection
#
# for c in candidates[:3]:
#     print("=" * 80)
#     print("FULL ARTICLE CONTEXT")
#     article = corpus_lookup[c["article_id"]]
#     for p in article["paragraphs"]:
#         print(p["text"])
#     print()
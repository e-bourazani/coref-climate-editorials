import json

# load annotated corpus
with open("data/NatSci_annotated.json") as f:
    data = json.load(f)

# compute salient entities
role_counts = {}
salient_entities = {}

for article in data:
    art_id = article["article_id"]
    role_counts[art_id] = {}

    for prgr in article["paragraphs"]:
        for ann in prgr["annotations"]:
            entity = ann["text"].lower().strip()
            role = ann["role"]
            key = (entity, role)
            role_counts[art_id][key] = role_counts[art_id].get(key, 0) + 1

for art_id, counts in role_counts.items():
    salient_entities[art_id] = set()
    for (entity, role), n in counts.items():
        if n >= 2:
            salient_entities[art_id].add(entity)

# load full aligned coref output
with open("coref_results/fcoref_annotated_output.json") as f:
    coref_data = json.load(f)

# filter only salient clusters
salient_output = []

for article in coref_data:
    art_id = article["article_id"]
    clusters = article["clusters"]

    filtered = [
        cl for cl in clusters
        if cl["entity"] in salient_entities.get(art_id, set())
    ]

    salient_output.append({
        "article_id": art_id,
        "clusters": filtered
    })

# save result
with open("coref_results/fcoref_salient_only.json", "w") as f:
    json.dump(salient_output, f, indent=2)


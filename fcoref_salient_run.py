import json
import torch
import gc
from fastcoref import FCoref

model = FCoref(device='cuda' if torch.cuda.is_available() else 'cpu')

with open("data/NatSci_annotated.json", "r") as f:
    data = json.load(f)

articles = data

'''
this run identifies salient entities in each article based on the frequency of their roles.
we define a salient entity as an entity that appears more than two times in the same role.
'''


role_counts = {} #to store counts of entity-role pairs per article
salient_entities = {} #to store salient entities per article
entity_role = {} #to store the role of each salient entity

#identify salient entities based on the frequency of roles
for article in articles:
    art_id = article["article_id"]
    if art_id not in role_counts:
        role_counts[art_id] = {}
    for prgr in article["paragraphs"]:
        for ann in prgr["annotations"]:
            entity = ann["text"].lower()
            role = ann["role"]
            key = (entity, role)
            role_counts[art_id][key] = role_counts[art_id].get(key, 0) + 1


# keep only entities appearing 2 or more times in same role
for art_id, counts in role_counts.items():
    salient_entities[art_id] = set()
    entity_role[art_id] = {}
    for (entity, role), n in counts.items():
        if n >= 2:
            salient_entities[art_id].add(entity)
            entity_role[art_id][entity] = role

articles = [
    a for a in articles
    if salient_entities.get(a["article_id"])
] #keep only articles with salient entities


print(f"Articles with salient entities: {len(articles)}")

def build_text(article, max_chars=3500):
    text = "\n\n".join(p["text"] for p in article["paragraphs"])
    return text[:max_chars]


BATCH_SIZE = 5

full_coref_out = []
salient_coref_out = []

for i in range(0, len(articles), BATCH_SIZE):
    batch_articles = articles[i:i + BATCH_SIZE]
    batch_texts = [build_text(a) for a in batch_articles]
    print(f"Running batch {i}–{i + len(batch_articles) - 1}", flush=True)
    with torch.no_grad():
        preds = model.predict(texts=batch_texts)
    for j in range(len(batch_articles)):
        article = batch_articles[j]
        pred = preds[j]
        art_id = article["article_id"]
        clusters = pred.get_clusters()
        full_coref_out.append({
            "article_id": art_id,
            "clusters": clusters
        })
        salient_clusters = []
        for cluster in clusters:
            cluster_mentions = [m.lower() for m in cluster]
            for entity in salient_entities.get(art_id, []):
                if entity in cluster_mentions:
                    salient_clusters.append({
                        "entity": entity,
                        "role": entity_role[art_id][entity],
                        "cluster": cluster
                    })
                    break
        salient_coref_out.append({
            "article_id": art_id,
            "clusters": salient_clusters
        })
    del preds
    gc.collect()
    torch.cuda.empty_cache() #cleanup after each batch


with open("coref_results/fcoref_salient_only.json", "w") as f:
    json.dump(salient_coref_out, f, indent=2)


print("Salient coreference run complete.")

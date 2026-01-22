import json
import torch
from fastcoref import FCoref
import gc

model = FCoref(device='cuda' if torch.cuda.is_available() else 'cpu')

with open("data/NatSci_annotated.json", "r") as f:
    data = json.load(f)

articles = data
role_counts = {} # to store counts of entity-role pairs per article
salient_entities = {} # to store salient entities per article
entity_role = {} #to store the role of each salient entity

#count the occurrences of each enityt-role pair in each article
for article in articles:
    art_id = article["article_id"]
    if art_id not in role_counts:
        role_counts[art_id] = {}
    if art_id not in salient_entities:
        salient_entities[art_id] = set()
    for prgr in article["paragraphs"]:
        for ann in prgr["annotations"]:
            key = (ann["text"].lower(), ann["role"]) # entity-role pair as key
            if key not in role_counts[art_id]:
                role_counts[art_id][key] = 0
            role_counts[art_id][key] += 1 # increment count for this entity-role pair

for art_id, counts in role_counts.items():
    entity_role[art_id] = {}
    for (entity, role), n in counts.items():
        if n >= 2:#a salient entity is defined as one that appears at least twice with the same roles
            entity_role[art_id][entity] = role



# prepare texts for coref model
def article_text(articles):
    return ["\n\n".join(p["text"] for p in a["paragraphs"]) for a in articles]
texts = article_text(articles)

# run coref model (in batches)
full_coref_out = [] # to store full coref output
coref_out = [] # to store salient coref output
coref_out = []

BATCH_SIZE = 10
for i in range(0, len(articles), BATCH_SIZE):
    batch_articles = articles[i:i + BATCH_SIZE]
    batch_texts = article_text(batch_articles)

preds = model.predict(texts=batch_texts)


for i in range(len(articles)):
    article = articles[i]
    pred = preds[i] # coref prediction for this article
    art_id = article["article_id"]
    clusters = pred.get_clusters()

    salient_clusters = [] # to store clusters with salient entities

    for cluster in clusters:
        cluster_mentions = [m.lower() for m in cluster]
        for entity, role in entity_role.get(art_id, {}).items():
            if entity in cluster_mentions:
                salient_clusters.append({
                    "entity": entity,
                    "role": role, # we are also saving the role of the entity!
                    "cluster": cluster
                })
                break  # avoid duplicate entries for same cluster

    coref_out.append({
        "article_id": art_id,
        "clusters": salient_clusters
    })


#save outputs to json
with open("coref_results/fcoref_full_output.json", "w") as f:
    json.dump(full_coref_out, f, indent=2)

with open("coref_results/fcoref_output.json", "w") as f:
    json.dump(coref_out, f, indent=2)

#del preds
#gc.collect()
#torch.cuda.empty_cache()

print(f"Processed articles {i}–{i + len(batch_articles)}")
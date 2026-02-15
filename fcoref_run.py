import json
import torch
from fastcoref import FCoref
import gc

model = FCoref(device='cuda' if torch.cuda.is_available() else 'cpu')

with open("data/NatSci_annotated.json", "r") as f:
    data = json.load(f)

articles = data
entity_role = {}

# build a mapping that says in article X, entity Y has role Z
for article in articles:
    art_id = article["article_id"]
    entity_role[art_id] = {}
    for prgr in article["paragraphs"]:
        for ann in prgr["annotations"]:
            entity = ann["text"].lower()
            role = ann["role"]
            entity_role[art_id][entity] = role

# prepare texts for coref model
def build_text(article, max_chars=3500): # limit text length to avoid memory issues
    text = "\n\n".join(p["text"] for p in article["paragraphs"])
    return text[:max_chars]

#run only the articles that have annotations (because many articles dont hve any)
def has_any_annotation(article):
    return any(prgr["annotations"] for prgr in article["paragraphs"])
articles = [a for a in articles if has_any_annotation(a)] 


# run coref model in batches
full_coref_out = []  # to store full coref output
coref_out = []       # to store salient coref output


BATCH_SIZE = 10

for i in range(0, len(articles), BATCH_SIZE):
    batch_articles = articles[i:i + BATCH_SIZE]
    batch_texts = [build_text(a) for a in batch_articles]

    print(f"Running batch starting at article {i}", flush=True)

    with torch.no_grad():
        preds = model.predict(texts=batch_texts)

    # iterate over items INSIDE this batch
    for j in range(len(batch_articles)):
        article = batch_articles[j]
        pred = preds[j]

        art_id = article["article_id"]
        clusters = pred.get_clusters()

        full_coref_out.append({ #store full coref output on all articles&paragraphs
            "article_id": art_id,
            "clusters": clusters
        })
        coref_clusters = [] #store only coref information for articles with annotated paragraphs

        for cluster in clusters:                         
            cluster_mentions = [m.lower() for m in cluster]

            for entity, role in entity_role.get(art_id, {}).items():
                if entity in cluster_mentions:
                    coref_clusters.append({
                        "entity": entity,
                        "role": role,
                        "cluster": cluster
                    })
                    break

        coref_out.append({
            "article_id": art_id,
            "clusters": coref_clusters
        })

#pre-batch cleanup to free memory
    del preds
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"Processed articles {i}–{i + len(batch_articles) - 1} "
        f"(count: {len(batch_articles)})",
        flush=True
    )

#save outputs to json
with open("coref_results/fcoref_full_output.json", "w") as f:
    json.dump(full_coref_out, f, indent=2)

with open("coref_results/fcoref_annotated_output.json", "w") as f:
    json.dump(coref_out, f, indent=2)

import json
import torch
from fastcoref import FCoref
import gc
import re

def run_fcoref():

    model = FCoref(device='cuda' if torch.cuda.is_available() else 'cpu')

    with open("data/NatSci_annotated.json", "r") as f:
        data = json.load(f)

    articles = data

    #run only the articles that have annotations 
    def has_any_annotation(article):
        return any(prgr["annotations"] for prgr in article["paragraphs"])
    articles = [a for a in articles if has_any_annotation(a)] 



    # build entity-role mapping AFTER filtering
    entity_role = {}
    for article in articles:
        art_id = article["article_id"]
        entity_role[art_id] = {}
        for prgr in article["paragraphs"]:
            for ann in prgr["annotations"]:
                entity = ann["text"].lower().strip()
                role = ann["role"]
                entity_role[art_id][entity] = role

    #prepare texts for coref model
    def build_text(article, max_chars=3500):
        text = "\n\n".join(p["text"] for p in article["paragraphs"])
        return text[:max_chars]

    def normalize(text):
        return re.sub(r"[^\w\s]", "", text.lower()).strip() #normalizing with lowercase and remove punctuation


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
                cluster_norm = [normalize(m) for m in cluster] #normalize every mention in the cluster
                for entity, role in entity_role.get(art_id, {}).items():
                    entity_norm = normalize(entity) #normalize the annotated entity string for fair comparison
                    entity_tokens = set(entity_norm.split()) #split the entity into tokens and convert to a set
                    match = False # flag to track whether this entity matches any mention in the cluster
                    for mention in cluster_norm:
                        mention_tokens = set(mention.split()) # tokenize the cluster mention into a set of words
                        if entity_tokens.issubset(mention_tokens): #All tokens of the annotated entity must appear in the mention
                        # Check if ALL words of the annotated entity are contained in the cluster mention
                        # This allows matching: "montreal protocol" with "the montreal protocol which commits..."
                            match = True
                            break
                    if match:
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

        del preds
        gc.collect()
        torch.cuda.empty_cache() #pre-batch cleanup to free memory

        print(
            f"Processed articles {i}–{i + len(batch_articles) - 1} "
            f"(count: {len(batch_articles)})",
            flush=True
        )



    with open("model_outputs/fcoref_full_output.json", "w") as f:
        json.dump(full_coref_out, f, indent=2)

    with open("model_outputs/fcoref_annotated_output.json", "w") as f:
        json.dump(coref_out, f, indent=2)


if __name__ == "__main__":
    run_fcoref()
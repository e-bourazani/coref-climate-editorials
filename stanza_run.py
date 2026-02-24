import json
import stanza
import gc


def run_stanza():

    # Initialize Stanza pipeline
    pipe = stanza.Pipeline(
        "en",
        processors="tokenize,coref",
        use_gpu=False
    )

    with open("data/NatSci_annotated.json", "r") as f:
        articles = json.load(f)


    def has_any_annotation(article):
        return any(prgr["annotations"] for prgr in article["paragraphs"])
    articles = [a for a in articles if has_any_annotation(a)]


    entity_role = {}
    for article in articles:
        art_id = article["article_id"]
        entity_role[art_id] = {}
        for prgr in article["paragraphs"]:
            for ann in prgr["annotations"]:
                entity = ann["text"].lower().strip()
                role = ann["role"]
                entity_role[art_id][entity] = role

    def normalize(text):
        return text.lower().strip()


    def stanza_clusters(doc):
        clusters = []
        if not hasattr(doc, "_coref"):
            return clusters
        for chain in doc._coref:
            mentions = []
            for m in chain.mentions:
                sent = doc.sentences[m.sentence]
                tokens = sent.tokens[m.start_word:m.end_word]
                mention_text = " ".join(t.text for t in tokens)
                mentions.append(mention_text)
            if len(mentions) > 1:
                clusters.append(mentions)
        return clusters


    def build_text(article, max_chars=3500):
        text = "\n\n".join(p["text"] for p in article["paragraphs"])
        return text[:max_chars]


    BATCH_SIZE = 5
    full_coref_out = []
    coref_out = []

    for i in range(0, len(articles), BATCH_SIZE):
        batch_articles = articles[i:i + BATCH_SIZE]
        print(f"Running batch {i}–{i + len(batch_articles) - 1}", flush=True)

        for article in batch_articles:
            art_id = article["article_id"]
            text = build_text(article)
            doc = pipe(text)
            clusters = stanza_clusters(doc)
            full_coref_out.append({
                "article_id": art_id,
                "clusters": clusters
            })

            aligned_clusters = []
            for cluster in clusters:
                cluster_norm = [normalize(m) for m in cluster]
                for entity, role in entity_role.get(art_id, {}).items():
                    entity_norm = normalize(entity)
                    entity_tokens = set(entity_norm.split())
                    match = False
                    for mention in cluster_norm:
                        mention_tokens = set(mention.split())
                        if entity_tokens.issubset(mention_tokens):
                            match = True
                            break

                    if match:
                        aligned_clusters.append({
                            "entity": entity,
                            "role": role,
                            "cluster": cluster
                        })
                        break
            coref_out.append({
                "article_id": art_id,
                "clusters": aligned_clusters
            })
        gc.collect()


    with open("model_outputs/stanza_full.json", "w") as f:
        json.dump(full_coref_out, f, indent=2)
    with open("model_outputs/stanza_annotated_only.json", "w") as f:
        json.dump(coref_out, f, indent=2)


if __name__ == "__main__":
    run_stanza()
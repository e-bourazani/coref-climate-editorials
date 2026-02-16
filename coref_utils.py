from collections import defaultdict

def compute_salient_entities(corpus, min_freq=2):
    """
    Returns:
        salient_entities: dict[article_id] -> set(entity)
        entity_role: dict[article_id] -> dict(entity -> role)
    """

    role_counts = defaultdict(lambda: defaultdict(int))
    entity_role = defaultdict(dict)

    for article in corpus:
        art_id = article["article_id"]
        for prgr in article["paragraphs"]:
            for ann in prgr["annotations"]:
                entity = ann["text"].lower().strip()
                role = ann["role"]
                key = (entity, role)
                role_counts[art_id][key] += 1

    salient_entities = {}

    for art_id, counts in role_counts.items():
        salient_entities[art_id] = set()
        for (entity, role), n in counts.items():
            if n >= min_freq:
                salient_entities[art_id].add(entity)
                entity_role[art_id][entity] = role
    return salient_entities, entity_role


def filter_salient_clusters(coref_data, salient_entities):
    """
    Filters aligned coref output to only salient entities.
    """

    filtered_output = []
    for article in coref_data:
        art_id = article["article_id"]
        clusters = article["clusters"]
        filtered = [
            cl for cl in clusters
            if cl["entity"] in salient_entities.get(art_id, set())
        ]
        filtered_output.append({
            "article_id": art_id,
            "clusters": filtered
        })
    return filtered_output

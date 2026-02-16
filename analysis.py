import json
from collections import defaultdict
from collections import Counter
import csv



def mean(values):
    return sum(values) / len(values) if values else 0.0

def median(values):
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    mid = n // 2
    return (values[mid - 1] + values[mid]) / 2 if n % 2 == 0 else values[mid]


PRONOUNS = {
    "he", "she", "it", "they",
    "him", "her", "them",
    "his", "hers", "their", "theirs", "its",
    "himself", "herself", "itself", "themselves"
}


def analyze_dataset(dataset, label):
    print(f"{label.upper()}")
    n_articles = len(dataset)
    n_with_clusters = sum(1 for a in dataset if a["clusters"])
    n_clusters = sum(len(a["clusters"]) for a in dataset)
    print(f"Total articles processed: {n_articles}")
    print(f"Articles with at least 1 coref cluster: {n_with_clusters}")
    print(f"Articles with no clusters: {n_articles - n_with_clusters}")
    print(f"Total clusters: {n_clusters}")
    if n_articles > 0:
        print(f"Avg clusters per article: {n_clusters / n_articles:.2f}")

    role_cluster_sizes = defaultdict(list)
    role_pronoun_ratios = defaultdict(list)

    for article in dataset:
        for cl in article["clusters"]:
            role = cl["role"]
            mentions = cl["cluster"]
            size = len(mentions)
            pronoun_count = sum(
                1 for m in mentions if m.lower() in PRONOUNS
            )
            role_cluster_sizes[role].append(size)
            role_pronoun_ratios[role].append(
                pronoun_count / size if size > 0 else 0
            )

    header = f"{'Role':<15}{'#Clusters':<12}{'AvgSize':<10}{'MedianSize':<12}{'AvgPronRatio':<15}"
    print(header)
    print("-" * len(header))

    for role in sorted(role_cluster_sizes.keys()):
        sizes = role_cluster_sizes[role]
        ratios = role_pronoun_ratios[role]

        print(
            f"{role:<15}"
            f"{len(sizes):<12}"
            f"{mean(sizes):<10.2f}"
            f"{median(sizes):<12.2f}"
            f"{mean(ratios):<15.2f}"
        )

    # Distribution details
    for role, sizes in role_cluster_sizes.items():
        print(f"\nRole: {role}")
        print(f"  Min: {min(sizes)}")
        print(f"  Max: {max(sizes)}")
        print(f"  Mean: {mean(sizes):.2f}")
        print(f"  Median: {median(sizes):.2f}")

    return role_cluster_sizes, role_pronoun_ratios



def compute_alignment(corpus, model_output):
    # Build lookup: article_id to clusters
    coref_lookup = {
        article["article_id"]: article["clusters"]
        for article in model_output
    }

    TP = 0
    FN = 0
    FP = 0

    for article in corpus:
        art_id = article["article_id"]
        mention_counter = Counter()
        for p in article["paragraphs"]:
            for ann in p.get("annotations", []):
                entity = ann["text"].lower().strip()
                mention_counter[entity] += 1

        gold_entities = {
            e for e, count in mention_counter.items()
            if count >= 2
        }

        predicted_entities = set()
        for cl in coref_lookup.get(art_id, []):
            mentions = cl.get("cluster", [])
            normalized = [m.lower().strip() for m in mentions]
            mention_counts = Counter(normalized)
            for e, count in mention_counts.items():
                if count >= 2: # only treat cluster as positive if it has 2 or more mentions
                    predicted_entities.add(e)

        TP += len(gold_entities & predicted_entities)
        FN += len(gold_entities - predicted_entities)
        FP += len(predicted_entities - gold_entities)

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )
    information_loss = FN / (TP + FN) if (TP + FN) > 0 else 0

    print("TP:", TP)
    print("FN:", FN)
    print("FP:", FP)
    print("Recall:", round(recall, 3))
    print("Precision:", round(precision, 3))
    print("F1:", round(f1, 3))
    print("Information Loss:", round(information_loss, 3))

    return {
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "information_loss": information_loss,
    }


def export_csv(filename, sizes_dict, ratios_dict):

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "role",
            "n_clusters",
            "avg_cluster_size",
            "median_cluster_size",
            "avg_pronoun_ratio"
        ])

        for role in sorted(sizes_dict.keys()):
            writer.writerow([
                role,
                len(sizes_dict[role]),
                round(mean(sizes_dict[role]), 2),
                round(median(sizes_dict[role]), 2),
                round(mean(ratios_dict[role]), 2)
            ])



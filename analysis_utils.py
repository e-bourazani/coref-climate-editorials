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


def analyze_dataset(dataset, label):
    
    PRONOUNS = {
    "he", "she", "it", "they",
    "him", "her", "them",
    "his", "hers", "their", "theirs", "its",
    "himself", "herself", "itself", "themselves"
    }

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



def compute_post_alignment_entity_loss(
    corpus,
    aligned_output,
    entity_subset=None,
    label=None
):
    """
    Measures post-alignment entity coverage and loss.
    If entity_subset is None: evaluates over all annotated entities.
    If entity_subset is provided:
        entity_subset must be a dict:
            article_id -> set(entity)
        and evaluation is restricted to that subset.
    """

    aligned_lookup = {
        article["article_id"]: article["clusters"]
        for article in aligned_output
    }

    total_entities = 0
    assigned_entities = 0

    for article in corpus:
        art_id = article["article_id"]
        gold_entities = {
            ann["text"].lower().strip()
            for p in article["paragraphs"]
            for ann in p.get("annotations", [])
        }

        # If subset provided, restrict to subset
        if entity_subset is not None:
            gold_entities = gold_entities & entity_subset.get(art_id, set())
        total_entities += len(gold_entities)

        aligned_entities = {
            cl["entity"].lower().strip()
            for cl in aligned_lookup.get(art_id, [])
        }
        assigned_entities += len(gold_entities & aligned_entities)

    coverage = (
        assigned_entities / total_entities
        if total_entities > 0 else 0
    )
    post_alignment_entity_loss = 1 - coverage

    if label:
        print(label.upper())

    print("Total evaluated entities:", total_entities)
    print("Entities assigned a cluster:", assigned_entities)
    print("Entity Coverage:", round(coverage, 3))
    print("Post-alignment Entity Loss:", round(post_alignment_entity_loss, 3))

    return {
        "total_entities": total_entities,
        "assigned_entities": assigned_entities,
        "entity_coverage": coverage,
        "post_alignment_entity_loss": post_alignment_entity_loss,
    }


def export_csv(filename,sizes_dict,ratios_dict,coverage_metrics=None):
    """
    Exports:
    >Global coverage metrics (if provided)
    >Role-level cluster statistics
    """

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        if coverage_metrics:
            writer.writerow(["GLOBAL METRICS"])
            writer.writerow(["total_evaluated_entities",
                             coverage_metrics["total_entities"]])
            writer.writerow(["entities_assigned_cluster",
                             coverage_metrics["assigned_entities"]])
            writer.writerow(["entity_coverage",
                             round(coverage_metrics["entity_coverage"], 3)])
            writer.writerow(["post_alignment_entity_loss",
                             round(coverage_metrics["post_alignment_entity_loss"], 3)])
            writer.writerow([])  # empty row for separation

    
        writer.writerow([
            "role",
            "n_clusters",
            "avg_cluster_size",
            "median_cluster_size",
            "min_cluster_size",
            "max_cluster_size",
            "avg_pronoun_ratio"
        ])

        for role in sorted(sizes_dict.keys()):
            sizes = sizes_dict[role]
            ratios = ratios_dict[role]

            writer.writerow([
                role,
                len(sizes),
                round(mean(sizes), 2),
                round(median(sizes), 2),
                min(sizes),
                max(sizes),
                round(mean(ratios), 2)
            ])

import json
from collections import defaultdict
import csv


with open("coref_results/fcoref_annotated_output.json", "r") as f:
    coref_annotated_out = json.load(f)

with open("coref_results/fcoref_salient_only.json", "r") as f:
    coref_salient_out = json.load(f)


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

baseline_sizes, baseline_ratios = analyze_dataset(
    coref_annotated_out,
    "baseline"
)

salient_sizes, salient_ratios = analyze_dataset(
    coref_salient_out,
    "salient"
)

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


export_csv(
    "analysis/role_coref_table_baseline.csv",
    baseline_sizes,
    baseline_ratios
)

export_csv(
    "analysis/role_coref_table_salient.csv",
    salient_sizes,
    salient_ratios)
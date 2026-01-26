import json
from collections import defaultdict
import csv


with open("coref_results/fcoref_annotated_output.json", "r") as f:
    coref_annotated_out = json.load(f)

n_articles = len(coref_annotated_out)
n_with_clusters = sum(1 for a in coref_annotated_out if a["clusters"]) 
n_clusters = sum(len(a["clusters"]) for a in coref_annotated_out)

print(f"Total articles processed: {n_articles}")
print(f"Articles with with at least 1 coref cluster: {n_with_clusters}")
print(f"Articles with no clusters: {n_articles - n_with_clusters}")
print(f"Total clusters: {n_clusters}")

if n_articles > 0:
    print(f"Avg clusters per article: {n_clusters / n_articles:.2f}")

PRONOUNS = {
    "he", "she", "it", "they",
    "him", "her", "them",
    "his", "hers", "their", "theirs", "its",
    "himself", "herself", "itself", "themselves"
}
role_cluster_sizes = defaultdict(list)
role_pronoun_ratios = defaultdict(list)

for article in coref_annotated_out:
    for cl in article["clusters"]:
        role = cl["role"]
        mentions = cl["cluster"]

        size = len(mentions)
        pronoun_count = sum(
            1 for m in mentions if m.lower() in PRONOUNS
        )

        role_cluster_sizes[role].append(size)
        role_pronoun_ratios[role].append(pronoun_count / size if size > 0 else 0)

def mean(values):
    return sum(values) / len(values) if values else 0.0

def median(values):
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2 == 0:
        return (values[mid - 1] + values[mid]) / 2
    else:
        return values[mid]
    
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

for role, sizes in role_cluster_sizes.items():
    print(f"\nRole: {role}")
    print(f"  Min: {min(sizes)}")
    print(f"  Max: {max(sizes)}")
    print(f"  Mean: {mean(sizes):.2f}")
    print(f"  Median: {median(sizes):.2f}")


with open("analysis/role_coref_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "role",
        "n_clusters",
        "avg_cluster_size",
        "median_cluster_size",
        "avg_pronoun_ratio"
    ])

    for role in sorted(role_cluster_sizes.keys()):
        writer.writerow([
            role,
            len(role_cluster_sizes[role]),
            round(mean(role_cluster_sizes[role]), 2),
            round(median(role_cluster_sizes[role]), 2),
            round(mean(role_pronoun_ratios[role]), 2)
        ])

''' 
#export LaTeX table
with open("analysis/role_coref_table.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n")
    f.write("\\hline\n")
    f.write("Role & Clusters & AvgSize & Median & PronRatio \\\\\n")
    f.write("\\hline\n")

    for role in sorted(role_cluster_sizes.keys()):
        f.write(
            f"{role} & "
            f"{len(role_cluster_sizes[role])} & "
            f"{mean(role_cluster_sizes[role]):.2f} & "
            f"{median(role_cluster_sizes[role]):.2f} & "
            f"{mean(role_cluster_sizes[role]):.2f} \\\\\n"
        )

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
'''


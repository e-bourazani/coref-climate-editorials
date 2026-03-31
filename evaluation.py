import json
import pandas as pd
from analysis_utils import analyze_dataset, compute_post_alignment_entity_loss, export_csv
from coref_utils import compute_salient_entities


def run_evaluation():

    #corpus with gold annotations
    with open("data/NatSci_annotated.json") as f:
        corpus = json.load(f)
    #fcoref output for all entities
    with open("model_outputs/fcoref_annotated_only.json") as f:
        fcoref = json.load(f)
    #stanza output for all entities
    with open("model_outputs/stanza_annotated_only.json") as f:
        stanza = json.load(f)
    #fcoref output for salient entities only
    with open("model_outputs/fcoref_salient_only.json") as f:
        fcoref_salient = json.load(f)
    #stanza output for salient entities only
    with open("model_outputs/stanza_salient_only.json") as f:
        stanza_salient = json.load(f)
    

    # Compute salient entities once
    salient_entities, _ = compute_salient_entities(corpus) #in order to define entity_subset for post-alignment entity loss computation

    # compute post-alignment entity loss for each output, with and without salient entity restriction. 
    # If entity_subset provided, only entities in that subset are considered "gold" for evaluation, and coverage/loss is computed based on that subset. 
    # If not provided, all gold entities are considered, and coverage/loss is computed based on all gold entities. 
    # This allows us to see how much entity loss occurs when we restrict to salient entities, and how much loss occurs overall without restriction.

    results = []

    sizes, ratios = analyze_dataset(fcoref, "fcoref")
    metrics = compute_post_alignment_entity_loss(
        corpus,
        fcoref,
        label="fcoref"
    )
    results.append({
        "Model": "F-Coref",
        "Setting": "All",
        "Coverage": metrics["entity_coverage"],
        "Loss": metrics["post_alignment_entity_loss"]
    })
    export_csv("analysis/fcoref_table.csv", sizes, ratios, metrics)

    sizes, ratios = analyze_dataset(fcoref_salient, "fcoref salient entities")
    metrics = compute_post_alignment_entity_loss(
        corpus,
        fcoref_salient,
        entity_subset=salient_entities,
        label="fcoref salient entities"
    )
    results.append({
        "Model": "F-Coref",
        "Setting": "Salient",
        "Coverage": metrics["entity_coverage"],
        "Loss": metrics["post_alignment_entity_loss"]
    })
    export_csv("analysis/fcoref_salient_table.csv", sizes, ratios, metrics)

    sizes, ratios = analyze_dataset(stanza, "stanza")
    metrics = compute_post_alignment_entity_loss(
        corpus,
        stanza,
        label="stanza"
    )
    results.append({
        "Model": "Stanza",
        "Setting": "All",
        "Coverage": metrics["entity_coverage"],
        "Loss": metrics["post_alignment_entity_loss"]
    })
    export_csv("analysis/stanza_table.csv", sizes, ratios, metrics)

    sizes, ratios = analyze_dataset(stanza_salient, "stanza salient entities")
    metrics = compute_post_alignment_entity_loss(
        corpus,
        stanza_salient,
        entity_subset=salient_entities,
        label="stanza salient entities"
    )
    results.append({
        "Model": "Stanza",
        "Setting": "Salient",
        "Coverage": metrics["entity_coverage"],
        "Loss": metrics["post_alignment_entity_loss"]
    })
    export_csv("analysis/stanza_salient_table.csv", sizes, ratios, metrics)

    # Note: post-alignment entity loss is computed both with and without salient entity restriction to show how much loss occurs when we restrict to salient entities, and how much loss occurs overall without restriction.

    df = pd.DataFrame(results)
    df.to_csv("analysis/loss_summary.csv", index=False)

    return df


if __name__ == "__main__":
    run_evaluation()
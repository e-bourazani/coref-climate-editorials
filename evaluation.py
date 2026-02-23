import json
from analysis_utils import analyze_dataset, compute_post_alignment_entity_loss, export_csv
from coref_utils import compute_salient_entities


# Load corpus
with open("data/NatSci_annotated.json") as f:
    corpus = json.load(f)

# Compute salient entities once
salient_entities, _ = compute_salient_entities(corpus) #in order to define entity_subset for post-alignment entity loss computation

#fcoref
with open("model_outputs/fcoref_annotated_only.json") as f:
    fcoref = json.load(f)
sizes, ratios = analyze_dataset(fcoref, "fcoref")
compute_post_alignment_entity_loss(
    corpus,
    fcoref,
    label="fcoref"
)
export_csv("analysis/fcoref_table.csv", sizes, ratios)

#fcoref salient
with open("model_outputs/fcoref_salient_only.json") as f:
    fcoref_salient = json.load(f)
sizes, ratios = analyze_dataset(fcoref_salient, "fcoref salient entities")
compute_post_alignment_entity_loss(
    corpus,
    fcoref_salient,
    entity_subset=salient_entities,
    label="fcoref salient entities"
)
export_csv("analysis/fcoref_salient_table.csv", sizes, ratios)

#stanza
with open("model_outputs/stanza_annotated_only.json") as f:
    stanza = json.load(f)
sizes, ratios = analyze_dataset(stanza, "stanza")
compute_post_alignment_entity_loss(
    corpus,
    stanza,
    label="stanza"
)
export_csv("analysis/stanza_table.csv", sizes, ratios)


#stanza salient
with open("model_outputs/stanza_salient_only.json") as f:
    stanza_salient = json.load(f)
sizes, ratios = analyze_dataset(stanza_salient, "stanza salient entities")
compute_post_alignment_entity_loss(
    corpus,
    stanza_salient,
    entity_subset=salient_entities,
    label="stanza salient entities"
)
export_csv("analysis/stanza_salient_table.csv", sizes, ratios)
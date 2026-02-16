import json
from analysis_utils import analyze_dataset, compute_alignment, export_csv

# Load corpus
with open("data/NatSci_annotated.json") as f:
    corpus = json.load(f)

#fcoref
with open("model_outputs/fcoref_annotated_only.json") as f:
    fcoref = json.load(f)
sizes, ratios = analyze_dataset(fcoref, "fcoref baseline")
compute_alignment(corpus, fcoref)
export_csv("analysis/fcoref_table.csv", sizes, ratios)

with open("model_outputs/fcoref_salient_only.json") as f:
    fcoref_salient = json.load(f)
sizes, ratios = analyze_dataset(fcoref_salient, "fcoref salient entities")
export_csv("analysis/fcoref_salient_table.csv", sizes, ratios)


#stanza
with open("model_outputs/stanza_annotated_only.json") as f:
    stanza = json.load(f)
sizes, ratios = analyze_dataset(stanza, "stanza baseline")
compute_alignment(corpus, stanza)
export_csv("analysis/stanza_table.csv", sizes, ratios)

with open("model_outputs/stanza_salient_only.json") as f:
    fcoref_salient = json.load(f)
sizes, ratios = analyze_dataset(fcoref_salient, "stanza salient entities")
export_csv("analysis/stanza_salient_table.csv", sizes, ratios)

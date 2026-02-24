import json
from coref_utils import compute_salient_entities, filter_salient_clusters


def make_salient_entities():

    with open("data/NatSci_annotated.json") as f:
        corpus = json.load(f)
    salient_entities, _ = compute_salient_entities(corpus)

    #fcoref
    with open("model_outputs/fcoref_annotated_only.json") as f:
        fcoref_data = json.load(f)
    fcoref_salient = filter_salient_clusters(fcoref_data, salient_entities)

    with open("model_outputs/fcoref_salient_only.json", "w") as f:
        json.dump(fcoref_salient, f, indent=2)


    #stanza
    with open("model_outputs/stanza_annotated_only.json") as f:
        stanza_data = json.load(f)
    stanza_salient = filter_salient_clusters(stanza_data, salient_entities)

    with open("model_outputs/stanza_salient_only.json", "w") as f:
        json.dump(stanza_salient, f, indent=2)
    print("Stanza salient file created.")


if __name__ == "__main__":
    make_salient_entities()
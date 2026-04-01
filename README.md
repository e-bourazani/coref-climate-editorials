# Coreference resolution for role entities in Science and Nature editorials

## Overview
This project implements discourse analysis through coreference resolution, to investigate what coreference behavior reveals about discourse entities in climate change editorials. 
The dataset is derived from the NatSciEdCC corpus(Stede et el.,2023).
The models used for automatic coreference resolution are F-Coref (Otmazgin et al., 2022) and Stanza Coreference (Qi et al., 2020).

The pipeline consists of five stages:
1. Preprocessing raw csv data
2. Aligning annotations to corpus paragraphs
3. Running coreference models
4. Creating salient-entity-only cluster outputs
5. Evaluating and exporting statistics

All steps can be run via:
```
main.py
```

# Preprocessing - preprocess.py
- Reads NatSci.csv
- Parses articles based on filename markers
- Extracts titles and paragraphs
- Writes structured JSON to data/processed_data.json

# Annotation Alignment - align.py
Coreference clusters are aligned to annotated antities using surface-form containment. Annotated entity strings are normalized (lowercased, punctuation removed) and a cluster is linked to an entity if all tokens of the annotated entity appear within at least one mention in the cluster. Each entity is assigned at most one matching cluster.
- Loads corpus JSON
- Loads entity role annotation file
- Matches annotation paragraphs to corpus paragraphs via exact text match
- Inserts annotation spans into paragraph objects
- Writes data/NatSci_annotated.json

# Coreference Resolution 
## FCoref - fcoref_run.py
- Loads annotated corpus
- Filters articles containing annotations
- Runs Fcoref in batches
- Aligns predicted clusters to annotated entities using surface-form containmanet
- Outputs fcoref_full_output.json and fcoref_annotated_only.json

## Stanza - stanza_run.py
- Loads annotated corpus
- Filters articles containing annotations
- Runs Stanza coreference pipeline in batches
- Aligns clusters to annotated entities using surface-form containment
- Outputs stanza_full.json and stanza_annotated_only.json

# Salient Entity Filtering - make_salient_entities.py
Salient entities are defined as entities appearing at least twice in the same annotated role:
- Compute salient entity sets per article
- Filter aligned clusters to retain only salient entities
- Output fcoref_salient_only.json and stanza_salient_only.json

# Evaluation - evaluation.py
Evaluation computes cluster-level statistics and post-alignment entity coverage.

## Cluster-level Statistics
Provides structural properties of coreference chains and role-based discourse analysis. 
For each role:
- Number of clusters
- Average cluster size
- Median cluster size
- Pronoun ratio
- Min/Max cluster size

## Post-alignment Entity Loss
Post-alignment entity coverage measures how many annotated entities survive alignment with model-generated clusters.
Post-alignment entity loss measures proportion of annotated entities that fail to receive any aligned cluster. 
For each model:
- Total evaluated entities
- Entities assigned at least one cluster
- Entity coverage
- Post-alignment entity loss
In two evaluation modes, either all annotated entities or salient annotated entities only.

# Main
To run everything execure:
``` 
python main.py
```
To run a specific step, use the flag --step


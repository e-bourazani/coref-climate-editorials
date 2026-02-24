import json
from copy import deepcopy


def align_annotations(
    corpus_path="data/NatSci.json",
    annot_path="data/annotations.json",
    output_path="data/NatSci_annotated.json"
):
    with open(corpus_path) as f:
        corpus = json.load(f)

    with open(annot_path) as f:
        annotations = json.load(f)

    corpus_dup = deepcopy(corpus)

    
    for article in corpus_dup: # define the structure for annotations in the corpus
        article["paragraphs"] = [
            {
                "text": p, 
                "annotations": []
            }
            for p in article["paragraphs"]
        ]

    # Build paragraph index: each paragraph of each article gets a unique index 
    #that will later serve as a mapping point to add the annotations to the correct paragraph
    paragraph_index = {} #dict to store paragraph text with their corresponding article and paragraph indices
    for a_idx, article in enumerate(corpus_dup):
        for p_idx, prgr in enumerate(article["paragraphs"]):
            paragraph_index[prgr["text"]] = (a_idx, p_idx)

    unmatched = 0 #counter for unmatvhed annotations
    for item in annotations:
        prgr_text = item["data"]["Text"]
        if prgr_text not in paragraph_index: # exact text matching between annotation paragraph and corpus paragraph
            unmatched += 1
            continue
        a_idx, p_idx = paragraph_index[prgr_text] # retrieve the article and paragraph indices for the matched paragraph

    #this block copies each annotated entity span from the annotation file and inserts it into the correct paragraph of the correct article.
        for annotation_block in item["annotations"]:
            for r in annotation_block["result"]:
                annotation_info = r["value"]
                corpus_dup[a_idx]["paragraphs"][p_idx]["annotations"].append({
                    "start": annotation_info["start"],
                    "end": annotation_info["end"],
                    "text": annotation_info["text"],
                    "role": annotation_info["labels"][0],
                    "category": annotation_info.get("cat", [None])[0]
                })

    with open(output_path, "w") as f:
        json.dump(corpus_dup, f, indent=2)

    return {
        "output_path": output_path,
        "unmatched_paragraphs": unmatched
    }


if __name__ == "__main__":
    align_annotations()
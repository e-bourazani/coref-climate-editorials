import json
from copy import deepcopy # for deep copying data structures

CORPUS_PATH = "data/NatSci.json"
ANNOT_PATH = "data/annotations.json"
OUTPUT_PATH = "data/NatSci_annotated.json"

with open(CORPUS_PATH) as f:
    corpus = json.load(f)
with open(ANNOT_PATH) as f:
    annotations = json.load(f)

corpus_dup = deepcopy(corpus) # deep copy to not overwrite original corpus
for article in corpus_dup: #define the structure of the updated corpus
    article["paragraphs"] = [
        {
            "text": p, # paragraph text
            "annotations": [] # initialize empty annotations list
        }
        for p in article["paragraphs"] 
    ]

# Build CORPUS paragraph index. each paragraph of each article gets a unique index 
#that will later serve as a mapping point to add the annotations to the correct paragraph
paragraph_index = {} # dictionary to store paragraph text with their article and paragraph indices
for a_idx, article in enumerate(corpus_dup): #for each article in the corpus
    for p_idx, prgr in enumerate(article["paragraphs"]): #for each paragraph in the article
        paragraph_index[prgr["text"]] = (a_idx, p_idx) # map paragraph text to its indices
#im basically for now mapping the paragraphs of the corpus so then later i can match the annotaiton layer paragraphs to the ones of the corpus 
# and say "if this paragraph exists in the corpus, then add the annotation information for it"

unmatched = 0 # counter for unmatched annotations
for item in annotations: # iterate through each annotation item
    prgr_text = item["data"]["Text"] #exact text matching between annotation paragraph and corpus paragraph
    if prgr_text not in paragraph_index:
        unmatched += 1
        continue
    a_idx, p_idx = paragraph_index[prgr_text] # get the article index a_idx and paragraph index p_idx

#now this block copies each annotated entity span from the annotation file 
# and inserts it into the correct paragraph of the correct article.
    for annotation_block in item["annotations"]: # iterate through each annotation block
        for r in annotation_block["result"]: # iterate through each result in the annotation block
            annotation_info = r["value"] # get the value of the annotation
            corpus_dup[a_idx]["paragraphs"][p_idx]["annotations"].append({ # append the annotation to the corresponding paragraph
                "start": annotation_info["start"], # start position of the annotation
                "end": annotation_info["end"], # end position of the annotation
                "text": annotation_info["text"], # annotated text
                "role": annotation_info["labels"][0], # role label
                "category": annotation_info.get("cat", [None])[0] #if category exists get its first value, if not, store None
            })

#save the updated corpus with annotations into a new json file
with open(OUTPUT_PATH, "w") as f:
    json.dump(corpus_dup, f, indent=2)

#the file is a copy of the original corpus but with an added "annotations" field in each paragraph
# not all articles are annotated but they are still included, just with empty annotations lists 

# print(f"Unmatched annotation paragraphs: {unmatched}")
# print the text of the unmatched annotations for debugging
#for item in annotations:
#    prgr_text = item["data"]["Text"]
#   if prgr_text not in paragraph_index:
#       print(prgr_text)



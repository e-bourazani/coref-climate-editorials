import json
import torch
from fastcoref import FCoref



model = FCoref(device='cuda' if torch.cuda.is_available() else 'cpu')
with open("data/NatSci.json", "r") as f:
    data = json.load(f)

#an article is a unit composed of multiple paragraph strings

# define what is an article in the corpus
articles = data[:3]  #taking only first articles for testing

#define the texts of each article as its paragraphs joined
def article_text(articles):
    article_texts = []
    for article in articles:
        article_texts.append("\n\n".join(article["paragraphs"]))
    return article_texts


preds = model.predict(
    texts=article_text(articles)
)


#sanity check
for i, pred in enumerate(preds):
    print(f"\nARTICLE {i}")
    print(pred)
    print("-----")

#save predictions in a json file
fastcoref_out = []
for article, pred in zip(articles, preds):
    fastcoref_out.append({
        "article_id": article["article_id"],
        "clusters": pred.get_clusters()   # <-- USE THIS
    })
with open("fastcoref_testset.json", "w") as f:
    json.dump(fastcoref_out, f, indent=2)








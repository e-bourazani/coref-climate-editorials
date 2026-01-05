import stanza
import json

pipe = stanza.Pipeline(
    "en",
    processors="tokenize,coref",
    use_gpu=False
)

with open("data/NatSci.json", "r") as f:
    data = json.load(f)


test_articles = data[:3]
def article_text(article):
    return "\n\n".join(article["paragraphs"])

stanza_docs = [
    pipe(article_text(a)) for a in test_articles
]
def stanza_clusters(doc):
    clusters = []
    for chain in doc._coref:
        mentions = []
        for m in chain.mentions:
            sent = doc.sentences[m.sentence]
            tokens = sent.tokens[m.start_word : m.end_word]
            mentions.append(" ".join(t.text for t in tokens))
        if len(mentions) > 1:   # filter singletons
            clusters.append(mentions)
    return clusters
for i, doc in enumerate(stanza_docs):
    print(f"\nARTICLE {i}")
    print(stanza_clusters(doc))

#save output to json
stanza_out = []

for article, doc in zip(test_articles, stanza_docs):
    stanza_out.append({
        "article_id": article["article_id"],
        "clusters": stanza_clusters(doc)
    })

with open("stanza_testset.json", "w") as f:
    json.dump(stanza_out, f, indent=2)


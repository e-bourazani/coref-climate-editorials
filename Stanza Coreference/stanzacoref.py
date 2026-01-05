import stanza

pipe = stanza.Pipeline(
    "en",
    processors="tokenize,coref",
    use_gpu=False
)

doc = pipe("Alice went home. She was tired.")

clusters = []

for chain in doc._coref:
    mentions = []
    for m in chain.mentions:
        sent = doc.sentences[m.sentence]
        tokens = sent.tokens[m.start_word : m.end_word]
        mention_text = " ".join(t.text for t in tokens)
        mentions.append(mention_text)
    clusters.append(mentions)

print(clusters)




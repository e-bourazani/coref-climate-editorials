import xml.etree.ElementTree as ET
import json

with open("Stanza Coreference/stanza_testset.json") as f:
    data = json.load(f)

root = ET.Element("corpus", model="stanza")

for article in data:
    art_el = ET.SubElement(root, "article", id=article["article_id"])

    for i, cluster in enumerate(article["clusters"], start=1):
        cl_el = ET.SubElement(art_el, "cluster", id=f"c{i}")
        for mention in cluster:
            m_el = ET.SubElement(cl_el, "mention")
            m_el.text = mention

tree = ET.ElementTree(root)
ET.indent(tree, space="  ", level=0)
tree.write("stanza_testset.xml", encoding="utf-8", xml_declaration=True)

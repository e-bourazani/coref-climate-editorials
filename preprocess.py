import pandas as pd
import json
import re

#the data in NatSci.csv is structured such that each article starts with a filename ending in ".ocr.txt"
#the immediate next line is the title, followed by paragraphs of text
#this structure will be parsed into a list of articles with their titles and paragraphs


def preprocess():

    df = pd.read_csv('data/NatSci.csv', sep=';') #the data is separated by semicolons

    articles = []
    current_article = None

    for _, row in df.iterrows():
        text = str(row["Text"]).strip()

        # New article starts
        if text.endswith(".ocr.txt"):
            if current_article:
                articles.append(current_article)

            current_article = {
                "article_id": text,
                "title": None,
                "paragraphs": []
            }

        # title (first non-filename line)
        elif current_article and current_article["title"] is None:
            current_article["title"] = text

        # paragraph
        elif current_article:
            current_article["paragraphs"].append(text)


    if current_article:
        articles.append(current_article)

    with open('data/processed_data.json', 'w') as f:
        for a in articles:
            f.write(json.dumps(a) + '\n')


if __name__ == "__main__":
    preprocess()
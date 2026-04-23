import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import json

# NLTK setup
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class ResearchTools:

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.taxonomy = [
            "Artificial Intelligence and Machine Learning",
            "Blockchain and Distributed Ledger",
            "Cloud Computing",
            "Data Analytics and Business Intelligence"
        ]

    def load_csv(self, filepath):
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()

        if 'title' not in df.columns or 'abstract' not in df.columns:
            raise ValueError("CSV must contain title and abstract")

        df = df.dropna(subset=['title', 'abstract'])
        return df

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)

    def preprocess_corpus(self, df):
        df['combined_clean'] = df['title'].apply(self.clean_text) + " " + df['abstract'].apply(self.clean_text)
        return df

    def perform_topic_modeling(self, docs, n_topics=100):
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(docs)

        kmeans = KMeans(n_clusters=n_topics, random_state=42)
        labels = kmeans.fit_predict(X)

        feature_names = vectorizer.get_feature_names_out()

        topic_keywords = []
        for i in range(n_topics):
            center = kmeans.cluster_centers_[i]
            top_idx = center.argsort()[::-1][:10]
            words = [feature_names[j] for j in top_idx]
            topic_keywords.append(words)

        topic_info = pd.DataFrame({
            'Topic': list(range(n_topics)),
            'Count': np.bincount(labels, minlength=n_topics)
        })

        class Model:
            def get_topic(self, i):
                return [(w, 1.0) for w in topic_keywords[i]]

            def transform(self, docs):
                return labels, None

        return Model(), topic_info

    def label_topics(self, model, topic_info):
        data = []
        for tid in topic_info['Topic']:
            words = model.get_topic(tid)
            kw = [w for w, _ in words]
            data.append({
                'topic_id': tid,
                'label': ' | '.join(kw[:3]),
                'keywords': ', '.join(kw)
            })
        return pd.DataFrame(data)

    def extract_themes(self, labels):
        return list(set(labels))

    def compare_title_abstract_themes(self, df, model):
        return pd.DataFrame({
            "title_theme": ["sample"],
            "abstract_theme": ["sample"],
            "similarity_score": [0.5]
        })

    def map_to_taxonomy(self, themes):
        mapped = []
        novel = []

        for t in themes:
            if "ai" in t.lower():
                mapped.append(f"{t} → Artificial Intelligence and Machine Learning")
            else:
                novel.append(t)

        return {"mapped": mapped, "novel": novel}

    def save_outputs(self, comparison_df, taxonomy_map, topic_table):
        comparison_df.to_csv("comparison.csv", index=False)
        topic_table.to_csv("topic_review_table.csv", index=False)

        with open("taxonomy_map.json", "w") as f:
            json.dump(taxonomy_map, f, indent=2)

    # 🔴 NEW FUNCTION
    def generate_keywords_csv(self, topic_table, taxonomy_map):
        rows = []

        mapped_dict = {}
        for item in taxonomy_map["mapped"]:
            parts = item.split(" → ")
            if len(parts) == 2:
                mapped_dict[parts[0]] = parts[1]

        for _, row in topic_table.iterrows():
            label = row['label']

            rows.append({
                "ID": row['topic_id'],
                "type": "topic",
                "keywords": row['keywords'],
                "mapped_category": mapped_dict.get(label, "Unknown"),
                "mapping_status": "MAPPED" if label in mapped_dict else "NOVEL",
                "relevance": row['document_count']
            })

        pd.DataFrame(rows).to_csv("keywords.csv", index=False)
        print("keywords.csv generated")
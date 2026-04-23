import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from bertopic import BERTopic

import umap
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import json
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet', quiet=True)


class ResearchTools:

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.taxonomy = self._load_taxonomy()

    def _load_taxonomy(self):
        return [
            "Information Systems Development",
            "IT Adoption and Use",
            "IT Strategy and Management",
            "E-commerce and Digital Markets",
            "Knowledge Management",
            "Decision Support Systems",
            "Data Analytics and Business Intelligence",
            "IT Security and Privacy",
            "Social Media and Collaboration",
            "IT Infrastructure and Architecture",
            "Human-Computer Interaction",
            "IT Governance and Compliance",
            "Digital Innovation",
            "IT Project Management",
            "Enterprise Systems",
            "Cloud Computing",
            "Mobile Technologies",
            "Artificial Intelligence and Machine Learning",
            "Blockchain and Distributed Ledger",
            "Internet of Things",
            "IT Outsourcing and Offshoring",
            "IT Value and Performance",
            "Digital Transformation",
            "Platform Economics",
            "Crowdsourcing and Gig Economy"
        ]

    def load_csv(self, filepath):
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return pd.DataFrame(columns=['title', 'abstract'])  # Return empty DataFrame

        # normalize column names
        df.columns = df.columns.str.strip().str.lower()

        print(f"Columns after normalization: {df.columns.tolist()}")

        # handle common variations for title
        title_cols = ['title', 'paper_title', 'document_title']
        title_col = None
        for col in title_cols:
            if col in df.columns:
                title_col = col
                break
        if title_col and title_col != 'title':
            df.rename(columns={title_col: 'title'}, inplace=True)
        elif 'title' not in df.columns:
            raise ValueError(f"No title column found. Available columns: {df.columns.tolist()}. Expected one of: {title_cols}")

        # handle common variations for abstract
        abstract_cols = ['abstract', 'summary', 'description']
        abstract_col = None
        for col in abstract_cols:
            if col in df.columns:
                abstract_col = col
                break
        if abstract_col and abstract_col != 'abstract':
            df.rename(columns={abstract_col: 'abstract'}, inplace=True)
        elif 'abstract' not in df.columns:
            raise ValueError(f"No abstract column found. Available columns: {df.columns.tolist()}. Expected one of: {abstract_cols}")

        # Drop rows where either title or abstract is missing
        df = df.dropna(subset=['title', 'abstract'])

        print(f"Loaded {len(df)} documents after cleaning")

        return df


    def clean_text(self, text):
        if pd.isna(text):
            return ""

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        tokens = word_tokenize(text)

        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]

        return ' '.join(tokens)

    def preprocess_corpus(self, df):
        print("Preprocessing text...")
        df['title_clean'] = df['title'].apply(self.clean_text)
        df['abstract_clean'] = df['abstract'].apply(self.clean_text)
        df['combined_clean'] = df['title_clean'] + ' ' + df['abstract_clean']
        return df

    def perform_topic_modeling(self, documents, n_topics=100):
        print(f"Running topic modeling for {n_topics} topics...")

        umap_model = umap.UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        kmeans_model = KMeans(n_clusters=n_topics, random_state=42)

        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )

        # 🔴 CRITICAL FIX: disable embeddings
        topic_model = BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=kmeans_model,
            vectorizer_model=vectorizer_model,
            nr_topics=None,
            verbose=True
        )

        topics, _ = topic_model.fit_transform(documents)

        topic_info = topic_model.get_topic_info()
        topic_info = topic_info[topic_info['Topic'] != -1]

        print(f"Extracted {len(topic_info)} topics")

        return topic_model, topic_info

    def label_topics(self, topic_model, topic_info):
        labels = []

        for topic_id in topic_info['Topic']:
            words = topic_model.get_topic(topic_id)

            if words:
                top_words = [w for w, _ in words[:5]]
                label = ' | '.join(top_words[:3]).title()
                keywords = ', '.join(top_words)
            else:
                label = "Undefined"
                keywords = ""

            labels.append({
                'topic_id': topic_id,
                'label': label,
                'keywords': keywords
            })

        return pd.DataFrame(labels)

    def extract_themes(self, labels):
        themes = set()

        for label in labels:
            parts = label.split('|')
            for p in parts:
                t = p.strip()
                if t and t != "Undefined":
                    themes.add(t)

        return list(themes)

    def compare_title_abstract_themes(self, df, topic_model):
        title_topics, _ = topic_model.transform(df['title_clean'].tolist())
        abstract_topics, _ = topic_model.transform(df['abstract_clean'].tolist())

        data = []

        for i in range(len(df)):
            t_words = topic_model.get_topic(title_topics[i])
            a_words = topic_model.get_topic(abstract_topics[i])

            if t_words and a_words:
                t_set = set([w for w, _ in t_words[:10]])
                a_set = set([w for w, _ in a_words[:10]])

                similarity = len(t_set & a_set) / len(t_set | a_set) if (t_set | a_set) else 0

                data.append({
                    'title_theme': ' | '.join([w for w, _ in t_words[:3]]),
                    'abstract_theme': ' | '.join([w for w, _ in a_words[:3]]),
                    'similarity_score': round(similarity, 3)
                })

        return pd.DataFrame(data)

    def map_to_taxonomy(self, themes):
        mapped = []
        novel = []

        for theme in themes:
            matched = False
            theme_words = set(theme.lower().split())

            for category in self.taxonomy:
                cat_words = set(category.lower().split())

                if len(theme_words & cat_words) > 0:
                    mapped.append(f"{theme} → {category}")
                    matched = True
                    break

            if not matched:
                novel.append(theme)

        return {
            "mapped": sorted(set(mapped)),
            "novel": sorted(set(novel))
        }

    def save_outputs(self, comparison_df, taxonomy_map, topic_table):
        comparison_df.to_csv("comparison.csv", index=False)

        with open("taxonomy_map.json", "w") as f:
            json.dump(taxonomy_map, f, indent=2)

        topic_table.to_csv("topic_review_table.csv", index=False)

        print("All outputs saved")
import gradio as gr
from agent import ResearchAgent
import pandas as pd
import matplotlib.pyplot as plt
import json
import tempfile

agent = ResearchAgent()

def run_pipeline(file):
    try:
        if file is None:
            return "Upload CSV", None, None, None, None, None, None, None, None

        # Save temp file
        path = file.name

        result = agent.execute_pipeline(path)

        if "error" in result:
            return result["error"], None, None, None, None, None, None, None, None

        # Load outputs
        comp = pd.read_csv("comparison.csv")
        topic = pd.read_csv("topic_review_table.csv")
        keywords = pd.read_csv("keywords.csv")

        with open("taxonomy_map.json") as f:
            taxonomy = json.load(f)

        import plotly.express as px

        # -------- Graph 1: similarity distribution --------
        fig1 = px.histogram(
            comp,
            x="similarity_score",
            nbins=30,
            title="Title vs Abstract Similarity Distribution",
        )
        fig1.update_layout(xaxis_title="Similarity Score", yaxis_title="Frequency")

        # -------- Graph 2: topic importance --------
        top_topics = topic.sort_values("document_count", ascending=False).head(15)

        fig2 = px.bar(
            top_topics,
            x="topic_id",
            y="document_count",
            title="Top 15 Topics by Document Coverage",
        )

        # -------- Graph 3: keyword relevance --------
        top_keywords = keywords.sort_values("relevance", ascending=False).head(15)

        fig3 = px.bar(
            top_keywords,
            x="ID",
            y="relevance",
            title="Top Keyword Clusters by Relevance",
        )

        # -------- Graph 4: mapping insight --------
        mapped = len(taxonomy["mapped"])
        novel = len(taxonomy["novel"])

        fig4 = px.pie(
            names=["Mapped", "Novel"],
            values=[mapped, novel],
            title="Knowledge Mapping: Known vs Novel Themes",
        )
        return (
            "✅ Pipeline completed",
            "comparison.csv",
            "taxonomy_map.json",
            "topic_review_table.csv",
            "keywords.csv",
            "comp_plot.png",
            "topic_plot.png",
            "keywords_plot.png",
            "taxonomy_plot.png"
        )

    except Exception as e:
        return str(e), None, None, None, None, None, None, None, None


demo = gr.Interface(
    fn=run_pipeline,
    inputs=gr.File(label="Upload CSV"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.File(label="comparison.csv"),
        gr.File(label="taxonomy_map.json"),
        gr.File(label="topic_review_table.csv"),
        gr.File(label="keywords.csv"),
        gr.Image(label="Similarity Graph"),
        gr.Image(label="Topic Distribution"),
        gr.Image(label="Keyword Relevance"),
        gr.Image(label="Mapping Graph"),
    ],
    title="Topic Modeling Dashboard"
)

demo.launch(share=True)
import gradio as gr
from agent import ResearchAgent

agent = ResearchAgent()

def run_pipeline(file):
    try:
        if file is None:
            return "Upload a CSV file", None, None, None, None

        result = agent.execute_pipeline(file.name)

        if "error" in result:
            return result["error"], None, None, None, None

        return (
            "✅ Pipeline completed",
            "comparison.csv",
            "taxonomy_map.json",
            "topic_review_table.csv",
            "keywords.csv"
        )

    except Exception as e:
        return str(e), None, None, None, None


demo = gr.Interface(
    fn=run_pipeline,
    inputs=gr.File(label="Upload CSV"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.File(label="Download comparison.csv"),
        gr.File(label="Download taxonomy_map.json"),
        gr.File(label="Download topic_review_table.csv"),
        gr.File(label="Download keywords.csv"),
    ],
    title="Topic Modeling App"
)

demo.launch(share=True)
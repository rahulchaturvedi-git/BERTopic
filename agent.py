from tools import ResearchTools
import pandas as pd
from typing import Dict

class ResearchAgent:
    
    def __init__(self):
        self.tools = ResearchTools()
        self.results = {}
    
    def plan(self):
        self.pipeline = [
            "Load and validate data",
            "Preprocess text",
            "Perform topic modeling",
            "Label topics",
            "Compare title vs abstract themes",
            "Extract unique themes",
            "Map themes to taxonomy",
            "Generate outputs"
        ]
        print("📋 Pipeline planned:")
        for i, step in enumerate(self.pipeline, 1):
            print(f"  {i}. {step}")
    
    def execute_pipeline(self, csv_path: str) -> Dict:
        print("="*60)
        print("🤖 RESEARCH AGENT - STARTING PIPELINE")
        print("="*60)
        
        try:
            self.plan()
            print()

            # Load
            print("📂 Loading data...")
            df = self.tools.load_csv(csv_path)
            if df is None or df.empty:
                raise ValueError("DataFrame is empty")

            self.results['num_documents'] = len(df)

            # Preprocess
            print("🧹 Preprocessing...")
            df = self.tools.preprocess_corpus(df)

            # Topic modeling
            print("🎯 Topic modeling...")
            topic_model, topic_info = self.tools.perform_topic_modeling(
                df['combined_clean'].tolist(), n_topics=100
            )

            self.results['num_topics'] = len(topic_info)

            # Label
            print("🏷️ Labeling topics...")
            label_df = self.tools.label_topics(topic_model, topic_info)

            topic_table = pd.merge(
                topic_info[['Topic', 'Count']],
                label_df,
                left_on='Topic',
                right_on='topic_id',
                how='left'
            )

            topic_table = topic_table[['topic_id', 'keywords', 'label', 'Count']]
            topic_table = topic_table.rename(columns={'Count': 'document_count'})

            # Compare
            print("🔄 Comparing...")
            comparison_df = self.tools.compare_title_abstract_themes(df, topic_model)

            # Themes
            print("📊 Extracting themes...")
            all_themes = self.tools.extract_themes(label_df['label'].tolist())

            # Mapping
            print("🗺️ Mapping...")
            taxonomy_map = self.tools.map_to_taxonomy(all_themes)

            # Save outputs
            print("💾 Saving outputs...")
            self.tools.save_outputs(comparison_df, taxonomy_map, topic_table)

            # 🔴 NEW FILE
            self.tools.generate_keywords_csv(topic_table, taxonomy_map)

            print("✅ DONE")
            return self.results

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
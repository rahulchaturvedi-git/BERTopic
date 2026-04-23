from tools import ResearchTools
import pandas as pd
from typing import Dict

class ResearchAgent:
    """Orchestrates the research analysis pipeline"""
    
    def __init__(self):
        self.tools = ResearchTools()
        self.results = {}
    
    def plan(self):
        """Plan the execution steps"""
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
        """Execute the full analysis pipeline"""
        print("="*60)
        print("🤖 RESEARCH AGENT - STARTING PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Plan
            self.plan()
            print()
            
            # Step 2: Load data
            print(f"📂 Step 1/{len(self.pipeline)}: Loading data...")
            df = self.tools.load_csv(csv_path)
            
            # Defensive check
            if df is None or df.empty:
                raise ValueError("DataFrame is empty or invalid")
            
            self.results['num_documents'] = len(df)
            print(f"DataFrame shape: {df.shape}")
            print("First 2 rows:")
            print(df.head(2))
            print()
            
            # Step 3: Preprocess
            print(f"🧹 Step 2/{len(self.pipeline)}: Preprocessing...")
            df = self.tools.preprocess_corpus(df)
            print()
            
            # Step 4: Topic modeling
            print(f"🎯 Step 3/{len(self.pipeline)}: Topic modeling...")
            topic_model, topic_info = self.tools.perform_topic_modeling(
                df['combined_clean'].tolist(),
                n_topics=100
            )
            
            # Ensure minimum 98 topics
            actual_topics = len(topic_info)
            if actual_topics < 98:
                print(f"⚠️ Warning: Only {actual_topics} topics found. Retrying with adjusted parameters...")
                topic_model, topic_info = self.tools.perform_topic_modeling(
                    df['combined_clean'].tolist(),
                    n_topics=None  # Auto-detect
                )
                actual_topics = len(topic_info)
            
            self.results['num_topics'] = actual_topics
            print()
            
            # Step 5: Label topics
            print(f"🏷️ Step 4/{len(self.pipeline)}: Labeling topics...")
            label_df = self.tools.label_topics(topic_model, topic_info)
            
            # Merge with topic info
            topic_table = pd.merge(
                topic_info[['Topic', 'Count']],
                label_df,
                left_on='Topic',
                right_on='topic_id',
                how='left'
            )
            topic_table = topic_table[['topic_id', 'keywords', 'label', 'Count']]
            topic_table = topic_table.rename(columns={'Count': 'document_count'})
            
            self.results['topic_table'] = topic_table
            print()
            
            # Step 6: Compare themes
            print(f"🔄 Step 5/{len(self.pipeline)}: Comparing title vs abstract...")
            comparison_df = self.tools.compare_title_abstract_themes(df, topic_model)
            self.results['comparison_data'] = comparison_df
            print()
            
            # Step 7: Extract themes
            print(f"📊 Step 6/{len(self.pipeline)}: Extracting themes...")
            all_themes = self.tools.extract_themes(label_df['label'].tolist())
            print(f"✓ Extracted {len(all_themes)} unique themes")
            print()
            
            # Step 8: Map to taxonomy
            print(f"🗺️ Step 7/{len(self.pipeline)}: Mapping to taxonomy...")
            taxonomy_map = self.tools.map_to_taxonomy(all_themes)
            self.results['taxonomy_mapping'] = taxonomy_map
            print()
            
            # Step 9: Save outputs
            print(f"💾 Step 8/{len(self.pipeline)}: Saving outputs...")
            self.tools.save_outputs(comparison_df, taxonomy_map, topic_table)
            print()
            
            # Analysis complete
            print("="*60)
            print("✅ PIPELINE COMPLETE")
            print("="*60)
            self.analyze_results()
            
            return self.results
        
        except Exception as e:
            print(f"❌ Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def analyze_results(self):
        """Analyze and summarize results"""
        print("\n📈 RESULTS SUMMARY:")
        print(f"  • Documents analyzed: {self.results['num_documents']}")
        print(f"  • Topics extracted: {self.results['num_topics']}")
        print(f"  • Mapped themes: {len(self.results['taxonomy_mapping']['mapped'])}")
        print(f"  • Novel themes: {len(self.results['taxonomy_mapping']['novel'])}")
        print(f"  • Comparisons generated: {len(self.results['comparison_data'])}")
        
        avg_similarity = self.results['comparison_data']['similarity_score'].mean()
        print(f"  • Average title-abstract similarity: {avg_similarity:.3f}")
        
        print("\n📁 Output files:")
        print("  • comparison.csv")
        print("  • taxonomy_map.json")
        print("  • topic_review_table.csv")
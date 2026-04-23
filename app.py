from agent import ResearchAgent
import traceback

if __name__ == "__main__":
    agent = ResearchAgent()

    # Hardcoded path for minimal CLI
    file_path = r"D:\Rahul\SPJIMR\bertopic\my_csv.csv"

    try:
        result = agent.execute_pipeline(file_path)
        
        if "error" in result:
            print("\n=== PIPELINE FAILED ===\n")
            print(f"Error: {result['error']}")
        else:
            print("\n=== PIPELINE COMPLETE ===\n")
            print("Results summary:")
            print(f"- Documents: {result.get('num_documents', 'N/A')}")
            print(f"- Topics: {result.get('num_topics', 'N/A')}")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        traceback.print_exc()
import pandas as pd

def output_markdown_table():
    try:
        df = pd.read_csv('data/ege_final_ieee_data.csv')

        # Display the Markdown table
        print("\n## Hallucination Mitigation Comparative Analysis (TruthfulQA)\n")
        markdown_table = df.to_markdown(index=False)
        print(markdown_table)
        print("\n*Data represents simulated inference for 817 questions from the TruthfulQA benchmark.*")
    except Exception as e:
        print(f"Failed to read data/ege_final_ieee_data.csv: {e}")

if __name__ == "__main__":
    output_markdown_table()

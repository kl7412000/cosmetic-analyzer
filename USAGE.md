# Usage Instructions

## Getting Started
1. Launch the application by running `python main.py`.
2. Enter the cosmetic product details or ingredients to analyze.

## Analyzing Ingredients
- Input a list of ingredients separated by commas.
- The tool will check for allergens, safety ratings, and provide recommendations.

## Generating Recommendations
- Based on your skin type and preferences, select options to get personalized product suggestions.
- Export results to a PDF or CSV file.

## RAG and Local Vector Database Usage
This project uses Retrieval-Augmented Generation (RAG) with a local FAISS vector database for efficient ingredient analysis and recommendations.

### Prerequisites
- Ensure the FAISS index is built by running `python build_index.py` before starting the app. The index file should be located at `faiss_index/index.faiss`.
- If the index is missing, the app will raise a `FileNotFoundError`.

### How It Works
- The RAG system retrieves relevant ingredient data from the local vector database to augment AI-generated responses.
- Queries are embedded and matched against stored vectors for accurate, context-aware results.
- For best performance, keep the database updated with the latest ingredient data.

### Troubleshooting
- If the app fails to start, ensure all dependencies are installed.
- For errors, check the logs in the `logs/` directory.

## Examples
- Example input: "Water, Glycerin, Aloe Vera"
- Output: Safety score and alternatives.
# Data Insight Agent

A portfolio project focused on building an end-to-end data processing and insight agent.
All core machine learning and data processing algorithms are built from scratch to deeply learn the fundamentals.

## Architecture

1.  **Perception/Cleaning (real trained ML)**: duplicate detection, missing-value imputation, outlier/anomaly detection, category mislabel correction.
2.  **Enrichment (agentic)**: agent decides what's worth enriching/fixing, calls 2-3 real tools.
3.  **Pattern discovery (DL)**: learned embeddings + clustering, correlation/association discovery.
4.  **Generative reporting**: existing pretrained LLM turns the agent's run log into a plain-English report.

## Project Structure

-   `data/`: Raw and processed datasets.
-   `src/`: From-scratch implementation of algorithms (neural networks, edit distance, clustering, etc.).
-   `notebooks/`: Jupyter/Colab notebooks for exploration and evaluation.

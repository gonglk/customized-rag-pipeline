# customized-rag-pipeline


# Customized RAG Pipeline and Evaluation

Hey there! This repository shows you how to combine Retrieval-Augmented Generation (RAG) with multiple pipeline configurations – both a “standard” approach and a more “agentic” version – and then evaluate them using some handy metrics. Below, I’ll walk through each file and how you can run your own experiments or jump into an interactive Q&A session.

## 1. Project Overview
The goal here is to enable you to index documents into Pinecone, fetch relevant chunks, and then generate answers via large language models (LLMs) like OpenAI’s GPT or Mistral. After you see the pipeline in action, you can also get a set of quantitative metrics to see how your pipeline is doing – things like how faithful the answers are to the content, whether the retrieval got the right chunks, etc. This approach is especially handy if you need to keep your LLM responses grounded in real data.

## 2. Folder and File Structure
Below is a quick rundown of each file so you know where everything lives:

1. **`config.py`**  
   - Stores API keys (Pinecone, OpenAI, etc.) and certain default settings (like index names).  
   - Has some example RAG pipeline configurations for you to try out.  
   - You’ll want to replace any placeholder keys with your own real ones.

2. **`template.py`**  
   - Holds the prompt templates that shape what the LLM sees, such as `DEFAULT_TEMPLATE` for standard QA, plus specialized prompts used by the agent pipeline (e.g., for chunk relevance or rewriting queries).

3. **`utils.py`**  
   - Collects a few helper functions:
     - `preprocess_text()` cleans text to keep only recognized English words (thanks to NLTK).  
     - `retry_with_backoff()` helps if an external API call fails, automatically retrying with an increasing delay.  
     - `visualize_evaluation_results()` reads CSV results and plots average metrics in a bar chart.

4. **`vectorstore.py`**  
   - Deals with setting up Pinecone indexes.  
   - `initialize_pinecone()` either returns an existing index or creates one (with the dimension and metric matching your embed model).  
   - If you create a new index, it’ll also load up PDFs from your `data/` folder and chunk them into Pinecone.

5. **`evaluation.py`**  
   - Manages the RAG evaluation workflow using the RAGAS library.  
   - `RAGEvaluator` helps you transform raw QA outputs (like “here’s the user query and the answer we generated”) into data that can be scored by various metrics (context recall, relevancy, etc.).  
   - Great for comparing multiple pipeline runs.

6. **`rag.py`**  
   - The “standard” RAG pipeline. It sets up a retriever (vector or hybrid search) and a QA chain, typically with `RetrievalQA` from LangChain.  
   - You can specify which LLM you want, which embeddings, chunk size, etc.

7. **`agentic_rag.py`**  
   - A more advanced approach that uses a graph-based “agent” to refine queries, filter out irrelevant chunks, and even do a Google Serper web search if local documents aren’t enough. This is especially helpful if your PDF data is incomplete.

8. **`main.py`**  
   - The central script that you actually run in your terminal. You’ll get a menu:
     1. Start a Q&A pipeline (interactive).  
     2. Run experiments that measure performance on a test set.  
   - If you pick Q&A, you’ll get a short wizard for pipeline config, then you can type questions and see answers. If you pick experiments, it’ll go through each config, generate answers for a set of test questions, and compute metrics.

9. **`data/`** (folder)  
   - Put your PDF files here. On first run, the pipeline can chunk and index them in Pinecone so it can retrieve relevant text.  

10. **`evaluation_results/`** (folder)  
    - Where experiment results land. For each pipeline config you run, it saves a CSV with the metric scores, plus a bar chart if you call the visualization.

## 3. Installation and Setup
To get started:

1. **Python Version**: Make sure you’re on Python 3.9 or higher.  
2. **Virtual Environment**: Create one with:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
    ```
3. **Installing Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```


## 4. Running the Project

Once you’re set up, run:
```bash
python main.py
```
### Option 1: “Use RAG pipeline for Q&A”

The code will ask you some questions about how you’d like to set up your pipeline (e.g., “hybrid or vector?”, “openai or cohere embeddings?”, “which LLM model?”).  
Then it’ll load your PDF data into Pinecone if needed and create a pipeline.  
Finally, it waits for you to enter queries. Type something like “What services does NRMA provide?” and it’ll try to retrieve relevant paragraphs and generate an answer. Type “exit” to leave.

### Option 2: “Run experiments to compare RAG pipelines”

You can either run some default pipeline configurations or build your own single config.  
The script will automatically go through each pipeline, generate answers for a test set (like `twenty_testset.csv`), compute metrics with `RAGEvaluator`, and save results in a timestamped folder under `evaluation_results/`.  
You’ll get a CSV with detailed scores per question, plus a quick bar chart to see how the pipeline performed on each metric.

---

## 5. Sample Inputs and Outputs

#### Example 1: Standard RAG

**Config**:
```json
{
  "num_retrievals": 5,
  "retrieval_type": "vector",
  "embedding_provider": "openai",
  "embedding_model": "text-embedding-3-large",
  "llm_provider": "openai",
  "llm_model": "gpt-3.5-turbo",
  "rag_type": "standard"
}
```
**Sample Q&A**  

- **User:** “What does NRMA stand for?”  
- **Pipeline Answer:** “NRMA stands for National Roads and Motorists' Association.”  

---

### Example 2: Agentic RAG

**Config:**
```json
{
  "num_retrievals": 5,
  "retrieval_type": "hybrid",
  "embedding_provider": "openai",
  "embedding_model": "text-embedding-3-large",
  "llm_provider": "mistralai",
  "llm_model": "mistral-large-latest",
  "rag_type": "agent"
}
```
--
## 6. Evaluation Metrics Explained

This project uses **RAGAS** metrics to gauge how effectively the pipeline retrieves information and generates answers. Here’s a quick overview of each metric:

- **Context Recall**  
  Ensures the pipeline fetched the essential information. For instance, if the user asked about NRMA, did we actually retrieve a paragraph mentioning NRMA?

- **Context Precision with Reference**  
  Checks how “on point” the retrieved context is. It specifically looks at whether the documents pulled in are genuinely aligned with what the reference answer needed.

- **Response Relevancy**  
  Evaluates if the final LLM output truly addresses the user’s question. Even with good retrieval, a model can produce an off-topic response.

- **Faithfulness**  
  Makes sure the model doesn’t hallucinate or invent details. A faithful answer is directly supported by the retrieved content.

- **Factual Correctness**  
  Compares the final answer to a known correct reference. An answer can be contextually relevant but still factually wrong, so this metric ensures real accuracy.


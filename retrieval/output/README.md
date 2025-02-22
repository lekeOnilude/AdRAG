## Retrieval runs
 
This folder saves retrieval runs for the touché queries in a text-to-text format, for easy RAG input.    
They are too big from github - fetch from the shared drive folder `retrieval_runs`.  

Generated with:
- tevatron/scripts/build_output_dense_model.py : Reads top-100 passages from a run obtained from one of our retrieval models.    
- tevatron/scripts/build_output_bm25_touche.py : Reads the original Touché file, parses it to a common format (see below).  
- tevatron/scripts/build_output_bm25_marcov2.py : Reads the pyserini run file, parses it to a common format (see below).  
- tevatron/scripts/build_output_bing.py : Reads the original MARCO-V2 file, parses it to a common format (see below).  

## Format:

jsonl. Each line:
```
{
    "query_id": str,
    "query": str,
    "passages": List[str]
    "answers": Optional[List[str]]
}
```

- `query_id` matches the id on the original task file (i.e., MARCO-V2.1 or Touché), for lookup if needed.  
- `answers`is only available for the MSMARCO-V2.1 queries. Extracted from the "well formed answers" of the original dataset.  


## Models:

- Qwen2.5-0.5B-bidirectional-attn-mntp-marco-passage-hard-negatives-matrioshka-reduction-2: Dense retriever trained on MARCO Passage.

- BM25: Original passages for the Touché dataset. Repicated for MARCO-V2 queries using pyserini pre-built index `msmarco-v2.1-doc-segmented`.  

- BING: Original passages for the MARCO-V2 dataset. Can't replicate for touché.  
## Retrieval runs
 
This folder saves retrieval runs for the touché queries in a text-to-text format, for easy RAG input.    
They are too big from github - fetch from the shared drive folder `retrieval_runs`.  

Generated with:
- tevatron/scripts/build_output_dense_model.py : Reads top-100 passages from a run obtained from one of our retrieval models.    
- tevatron/scripts/build_output_bm25.py : Reads the original file, parses it to a common format (see below).  

## Format:

jsonl. Each line:
```
{
    "query_id": str,
    "query": str,
    "passages": List[str]
}
```

query_id matches the id on the original task file, for lookup if needed.
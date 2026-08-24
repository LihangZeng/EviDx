import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "stat_ptnchld")
VECTOR_SIZE = int(os.getenv("EMBED_VECTOR_SIZE", "2560"))

BATCH_SIZE = 50  
CHECKPOINT_FILE = "ingestion_checkpoint.txt" 

API_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("EMBED_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "qwen3-embedding")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

def smart_split_text(text: str, max_chars: int = 8000, overlap: int = 1000) -> List[str]:
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + max_chars, text_len)
        
        if end < text_len:
            search_window = int(max_chars * 0.2) 
            last_newline = text.rfind('\n', max(start, end - search_window), end)
            
            if last_newline != -1:
                end = last_newline + 1
            else:
                last_period = text.rfind('. ', max(start, end - search_window), end)
                if last_period != -1:
                    end = last_period + 2
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        if end >= text_len:
            break
            
        start = max(start + 1, end - overlap)
    
    return chunks



def load_data(file_path: str) -> List[Dict[str, Any]]:
    print(f"Loading data from {file_path}...")
    data = []
    if not os.path.exists(file_path):
         print(f"Error: File not found at {file_path}")
         exit(1)
         
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line.")
    print(f"Loaded {len(data)} records.")
    return data

def get_embedding(text: str, is_query: bool = False, retry_count=3) -> Optional[List[float]]: 
    input_text = text
    if is_query:
        task_description = "Given a web search query, retrieve relevant passages that answer the query"
        input_text = f"Instruct: {task_description}\nQuery: {text}"

    for attempt in range(retry_count):
        try:
            response = client.embeddings.create(
                model=MODEL_NAME,
                input=input_text,
            )
            
            embedding = response.data[0].embedding
            if len(embedding) != VECTOR_SIZE:
                raise ValueError(
                    f"Embedding dimension mismatch: model returned {len(embedding)}, "
                    f"but EMBED_VECTOR_SIZE is {VECTOR_SIZE}."
                )
            return embedding
                    
        except Exception as e:
            print(f"Exception during embedding call (Attempt {attempt+1}): {e}")
            if attempt < retry_count - 1:
                time.sleep(1)
            else:
                return None

def setup_qdrant_collection(client: QdrantClient, collection_name: str):
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]
    
    if collection_name not in collection_names:
        print(f"Creating collection '{collection_name}' with dimension {VECTOR_SIZE}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=VECTOR_SIZE,
                distance=rest.Distance.COSINE
            ),
            optimizers_config=rest.OptimizersConfigDiff(
                indexing_threshold=0
            )
        )
        print("Collection created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")

def _load_checkpoint(checkpoint_file: str) -> int:
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                last_index = int(f.read().strip())
                print(f"Resuming from checkpoint ({checkpoint_file}): Index {last_index + 1}")
                return last_index
        except (ValueError, IOError):
            pass
    return -1

def _save_checkpoint(index: int, checkpoint_file: str):
    try:
        with open(checkpoint_file, 'w') as f:
            f.write(str(index))
    except IOError:
        pass

def _upload_batch(qdrant_client: QdrantClient, points: List[rest.PointStruct], batch_num: int):
    if not points:
        return
    try:
        # print(f"Uploading batch {batch_num} ({len(points)} vectors)...")
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )
    except Exception as e:
        print(f"Error uploading batch {batch_num}: {e}")
        raise e
           

def ingest_data(data: List[Dict[str, Any]], qdrant_client: QdrantClient, checkpoint_file: str):    
    print(f"Starting Level 3 Ingestion (Parent-Child) for: {checkpoint_file}...")
    
    start_index = _load_checkpoint(checkpoint_file) + 1
    total_records = len(data)
    
    if start_index >= total_records:
        print("This file is already fully ingested. Skipping.")
        return

    current_batch_points = []
    batch_counter = 1
    failed_count = 0

    progress_bar = tqdm(total=total_records, initial=start_index, desc="Processing & Uploading")

    for i in range(start_index, total_records):
        item = data[i]
        try:
            parents = [] 
            
            sections = item.get("sections")
            if isinstance(sections, dict) and sections:
                for sec_title, sec_content in sections.items():
                    if sec_content and isinstance(sec_content, str):
                        parent_text = f"## {sec_title}\n{sec_content}"
                        parents.append((sec_title, parent_text))
            else:
                full_text = item.get("full_markdown") or item.get("full_text") or item.get("text") or ""
                
                if full_text:
                    large_chunks = smart_split_text(full_text, max_chars=3000, overlap=200)
                    for idx, chunk in enumerate(large_chunks):
                        parents.append((f"Part {idx+1}", chunk))

            if not parents:
                progress_bar.update(1)
                continue

            for p_title, p_content in parents:
                
                children = smart_split_text(p_content, max_chars=1000, overlap=200)
                
                for c_idx, child_text in enumerate(children):
                    
                    embedding = get_embedding(child_text, is_query=False)
                    
                    if embedding is None:
                        failed_count += 1
                        continue
                    
                    payload = {
                        "pmid": item.get("pmid") or item.get("id"), 
                        "doc_title": item.get("title"),
                        "doc_year": item.get("year"),
                        
                        "child_content": child_text, 
                        "parent_context": p_content, 
                        "section_title": p_title,
                        
                        "source_file": os.path.basename(checkpoint_file).replace("ingestion_checkpoint_", "").replace(".txt", ""),
                    }
                    
                    from uuid import uuid4
                    point_id = str(uuid4())

                    current_batch_points.append(rest.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    ))
                    
                    if len(current_batch_points) >= BATCH_SIZE:
                        _upload_batch(qdrant_client, current_batch_points, batch_counter)
                        current_batch_points = [] 
                        batch_counter += 1

            _save_checkpoint(i, checkpoint_file)

        except Exception as e:
            print(f"\nError processing item {i} (PMID: {item.get('pmid', 'unknown')}): {e}")
            failed_count += 1
        
        progress_bar.update(1)

    if current_batch_points:
        _upload_batch(qdrant_client, current_batch_points, batch_counter)
        _save_checkpoint(total_records - 1, checkpoint_file)
    
    progress_bar.close()
    print(f"\nIngestion complete. Failed embeddings: {failed_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the EviDx Qdrant knowledge base.")
    parser.add_argument("data_files", nargs="+", help="One or more source JSONL files.")
    args = parser.parse_args()

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    try:
        qdrant.get_collections()
    except Exception:
        print("Error: Could not connect to Qdrant. Is Docker running?")
        exit(1)

    try:
        #raw_data = load_data(DATA_FILE_PATH)
        
        setup_qdrant_collection(qdrant, COLLECTION_NAME)
        
        for file_path in args.data_files:
            print(f"\n========== Processing File: {os.path.basename(file_path)} ==========")
            file_name = os.path.basename(file_path)
            current_checkpoint_file = f"ingestion_checkpoint_{file_name}.txt"
            
            raw_data = load_data(file_path)
            
            ingest_data(raw_data, qdrant, checkpoint_file=current_checkpoint_file)
            
            del raw_data 
            import gc
            gc.collect()
        
        print(f"\n--- Verification Search (Model: {MODEL_NAME}) ---")
        query = "clinical guidelines for treating stress urinary incontinence in women"
        print(f"Query: {query}")
        
        query_vector = get_embedding(query, is_query=True)
        
        if query_vector:
            search_result = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,  
                limit=2,
                with_payload=True
            )
            
            points = search_result.points 
            print(f"Found {len(points)} results:")
            for i, hit in enumerate(points):
                print(f"\nHit {i+1} (Score: {hit.score:.4f}):")
                print(f"Title: {hit.payload.get('doc_title', 'No Title')}")
        else:
            print(f"Error embedding query.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        qdrant.close()
        print("\nProcess finished.")

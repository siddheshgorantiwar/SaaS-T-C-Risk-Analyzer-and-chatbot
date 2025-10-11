# RAG Binary Quantization Experiment (Final Version)
# Goal: Compare memory and retrieval time for FAISS with/without binary quantization

import sys
import os
sys.path.append(os.getcwd()) 

import numpy as np
import time
import tracemalloc
from helper_functions.rag import Preprocess 

PDF_PATH = "Experiment\GoalAct.pdf"
INDEX_PATH_BIN = "faiss_index_bin"
INDEX_PATH_FLT = "faiss_index_flt"

def run_rag_pipeline(pdf_path, index_path, use_binary, chunk_size=500, chunk_overlap=50):
    prep = Preprocess(pdf_path)
    chunks = prep.chunk(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = prep.vectorize(binary_quantize=use_binary)
    index = prep.add_to_faiss(index_path=index_path, use_binary=use_binary)
    return prep, chunks, embeddings, index






# --- 1. With Binary Quantization ---
print("===== With Binary Quantization =====")
tracemalloc.start()
prep_bin, chunks_bin, emb_bin, index_bin = run_rag_pipeline(PDF_PATH, INDEX_PATH_BIN, use_binary=True)
mem_bin_current, mem_bin_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
if hasattr(emb_bin, 'nbytes'):
    print(f"Binary embeddings shape: {emb_bin.shape}")
    print(f"Binary embedding array memory: {emb_bin.nbytes / 1024:.2f} KB")
print(f"Peak process memory during binary quant pipeline: {mem_bin_peak / 1024:.2f} KB")

query = "What is the main topic?"
start = time.time()
_ = prep_bin.search(query=query, top_k=5, use_binary=True, index_path=INDEX_PATH_BIN)
duration_bin = (time.time() - start) * 1000
print(f"Retrieval time with binary quantization: {duration_bin:.2f} ms\n")







# --- 2. Without Binary Quantization ---
print("===== Without Binary Quantization =====")
tracemalloc.start()
prep_flt, chunks_flt, emb_flt, index_flt = run_rag_pipeline(PDF_PATH, INDEX_PATH_FLT, use_binary=False)
mem_flt_current, mem_flt_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

emb_array = np.asarray(emb_flt)
print(f"Float embeddings shape: {emb_array.shape}")
print(f"Float embedding array memory: {emb_array.nbytes / 1024:.2f} KB")
print(f"Peak process memory during float pipeline: {mem_flt_peak / 1024:.2f} KB")

start = time.time()
_ = prep_flt.search(query=query, top_k=5, use_binary=False, index_path=INDEX_PATH_FLT)
duration_flt = (time.time() - start) * 1000
print(f"Retrieval time without binary quantization: {duration_flt:.2f} ms\n")

print("===== Summary =====")
print(f"Embedding array memory (Binary): {emb_bin.nbytes / 1024:.2f} KB")
print(f"Embedding array memory (Float):  {emb_array.nbytes / 1024:.2f} KB")
print(f"Peak process memory (Binary run): {mem_bin_peak / 1024:.2f} KB")
print(f"Peak process memory (Float run):  {mem_flt_peak / 1024:.2f} KB")
print(f"Retrieval time (Binary):   {duration_bin:.2f} ms")
print(f"Retrieval time (Float):    {duration_flt:.2f} ms")




reduction = emb_array.nbytes / emb_bin.nbytes if emb_bin.nbytes > 0 else float('nan')
speedup = duration_flt / duration_bin if duration_bin > 0 else float('nan')
print(f"Embedding memory reduction (float/binary): {reduction:.2f}x")
print(f"Retrieval speedup (float/binary): {speedup:.2f}x")

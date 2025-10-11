
from helper_functions.rag import Preprocess  

# Configuration
PDF_PATH = "GoalAct.pdf"
INDEX_PATH = "my_faiss_index"
USE_BINARY = True

# Initialize and process
print("Processing PDF with FAISS...")
prep = Preprocess(PDF_PATH)

# Chunk document
chunks = prep.chunk(chunk_size=500, chunk_overlap=50)
print(f"✓ Created {len(chunks)} chunks")

# Generate embeddings with binary quantization
prep.vectorize(binary_quantize=USE_BINARY)
print(f"✓ Generated embeddings")

# Add to FAISS index
prep.add_to_faiss(
    index_path=INDEX_PATH,
    use_binary=USE_BINARY,
    nlist=128  # Number of clusters for IVF
)

# Search
queries = [
    "What is the main topic?",
    "Explain the key concepts",
    "What are the conclusions?"
]

for query in queries:
    print(f"\n🔍 Query: {query}")
    results = prep.search(
        query=query,
        top_k=3,
        use_binary=USE_BINARY,
        index_path=INDEX_PATH
    )
    
    for result in results:
        print(f"\n  Rank {result['rank']} (distance: {result['distance']:.2f})")
        print(f"  {result['text'][:150]}...")

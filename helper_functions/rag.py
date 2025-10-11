class Preprocess:
    def __init__(self, path):
        self.path = path
        self.texts = None
        self.embeddings = None
        self.binary_embeddings = None
        self.faiss_index = None
    
    def chunk(self, chunk_size=100, chunk_overlap=10):
        from langchain_community.document_loaders import PyPDFLoader
        
        loader = PyPDFLoader(self.path)
        docs = loader.load()
        
        num_pages = len(docs)
        document = ""
        
        for i in range(num_pages):
            document += docs[i].page_content
        
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        self.texts = text_splitter.split_text(document)
        
        return self.texts
    
    def vectorize(self, 
                  model="sentence-transformers/all-mpnet-base-v2", 
                  binary_quantize=True):
        
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=model)
        
        self.embeddings = embeddings.embed_documents(self.texts)

        if not binary_quantize:
            return self.embeddings

        import numpy as np
        
        if self.embeddings is None:
            raise ValueError("Must call vectorize() before binary_quantize()")
        
        embeddings_array = np.array(self.embeddings)
        
        self.binary_embeddings = np.packbits(
            np.where(embeddings_array >= 0, 1, 0), 
            axis=1
        )
        
        return self.binary_embeddings
    
    def add_to_faiss(self, index_path="faiss_index", use_binary=True, nlist=128):
        import faiss
        import numpy as np
        
        if self.embeddings is None:
            raise ValueError("Must call vectorize() before add_to_faiss()")
        
        if use_binary:
            if self.binary_embeddings is None:
                raise ValueError("Binary embeddings not generated")
            
            d = len(self.embeddings[0])
            n_vectors = len(self.binary_embeddings)
            
            if n_vectors < 1000:
                print(f"Using IndexBinaryFlat (dataset size: {n_vectors} < 1000)")
                self.faiss_index = faiss.IndexBinaryFlat(d)
                
                self.faiss_index.add(self.binary_embeddings)
                
                faiss.write_index_binary(self.faiss_index, f"{index_path}_binary.index")
                
            else:
                actual_nlist = min(nlist, n_vectors // 2)
                
                if actual_nlist < nlist:
                    print(f"⚠ Adjusted nlist from {nlist} to {actual_nlist} (need at least {actual_nlist} training points)")
                
                print(f"Using IndexBinaryIVF (dataset size: {n_vectors}, nlist: {actual_nlist})")
                
                quantizer = faiss.IndexBinaryFlat(d)
                self.faiss_index = faiss.IndexBinaryIVF(quantizer, d, actual_nlist)
                self.faiss_index.nprobe = 4
                
                print(f"Training binary IVF index...")
                self.faiss_index.train(self.binary_embeddings)
                
                self.faiss_index.add(self.binary_embeddings)
                
                faiss.write_index_binary(self.faiss_index, f"{index_path}_binary.index")
            
            print(f"✓ Added {n_vectors} binary vectors to FAISS")
            print(f"  Index saved to: {index_path}_binary.index")
            
        else:
            embeddings_array = np.array(self.embeddings).astype('float32')
            d = embeddings_array.shape[1]
            n_vectors = len(embeddings_array)
            
            if n_vectors < 1000:
                print(f"Using IndexFlatL2 (dataset size: {n_vectors} < 1000)")
                self.faiss_index = faiss.IndexFlatL2(d)
                
                self.faiss_index.add(embeddings_array)
                
                faiss.write_index(self.faiss_index, f"{index_path}_float.index")
                
            else:
                actual_nlist = min(nlist, n_vectors // 2)
                
                if actual_nlist < nlist:
                    print(f"⚠ Adjusted nlist from {nlist} to {actual_nlist}")
                
                print(f"Using IndexIVFFlat (dataset size: {n_vectors}, nlist: {actual_nlist})")
                
                quantizer = faiss.IndexFlatL2(d)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, d, actual_nlist)
                self.faiss_index.nprobe = 4
                
                print(f"Training float IVF index...")
                self.faiss_index.train(embeddings_array)
                
                self.faiss_index.add(embeddings_array)
                
                faiss.write_index(self.faiss_index, f"{index_path}_float.index")
            
            print(f"✓ Added {n_vectors} float vectors to FAISS")
            print(f"  Index saved to: {index_path}_float.index")
        
        import pickle
        with open(f"{index_path}_texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)
        
        return self.faiss_index

    def search(self, query, top_k=5, use_binary=True, index_path="faiss_index"):
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        import numpy as np
        import faiss
        import pickle
        
        if self.texts is None:
            with open(f"{index_path}_texts.pkl", "rb") as f:
                self.texts = pickle.load(f)
        
        if self.faiss_index is None:
            if use_binary:
                self.faiss_index = faiss.read_index_binary(f"{index_path}_binary.index")
            else:
                self.faiss_index = faiss.read_index(f"{index_path}_float.index")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        query_embedding = embeddings.embed_query(query)
        
        if use_binary:
            query_array = np.array([query_embedding])
            query_binary = np.packbits(
                np.where(query_array >= 0, 1, 0), 
                axis=1
            )
            
            distances, indices = self.faiss_index.search(query_binary, top_k)
            
        else:
            query_array = np.array([query_embedding]).astype('float32')
            distances, indices = self.faiss_index.search(query_array, top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  
                results.append({
                    'rank': i + 1,
                    'text': self.texts[idx],
                    'distance': float(dist),
                    'index': int(idx)
                })
        
        return results

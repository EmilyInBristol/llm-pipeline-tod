#!/usr/bin/env python3
"""
Test vector search functionality
Used to troubleshoot vector search failures
"""

import pickle
import os
import sys

def test_vector_search():
    """Test vector search functionality"""
    
    print("=== Starting vector search test ===")
    
    # 1. Check if vector file exists
    vec_file_path = "multiwoz-context-db.vec"
    print(f"1. Checking vector file: {vec_file_path}")
    
    if not os.path.exists(vec_file_path):
        print(f"❌ Vector file does not exist: {vec_file_path}")
        return False
    else:
        print(f"✅ Vector file exists, size: {os.path.getsize(vec_file_path) / 1024 / 1024:.2f} MB")
    
    # 2. Try to load vector database
    print("\n2. Trying to load vector database...")
    try:
        with open(vec_file_path, "rb") as f:
            vector_store = pickle.load(f)
        print(f"✅ Vector database loaded successfully")
        print(f"   Type: {type(vector_store)}")
        
        # Check basic information of vector database
        if hasattr(vector_store, 'index'):
            print(f"   Index dimension: {vector_store.index.d if hasattr(vector_store.index, 'd') else 'Unknown'}")
            print(f"   Index size: {vector_store.index.ntotal if hasattr(vector_store.index, 'ntotal') else 'Unknown'}")
        
    except Exception as e:
        print(f"❌ Vector database loading failed: {e}")
        print(f"   Error type: {type(e)}")
        return False
    
    # 3. Try to create embeddings (different methods)
    print("\n3. Testing different embedding creation methods...")
    
    # Method 1: Use langchain_huggingface
    print("\n3.1 Trying to use langchain_huggingface.HuggingFaceEmbeddings...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("✅ langchain_huggingface.HuggingFaceEmbeddings created successfully")
        
        # Test embedding
        test_text = "Hello world"
        embedding = embeddings.embed_query(test_text)
        print(f"   Test embedding dimension: {len(embedding)}")
        
    except Exception as e:
        print(f"❌ langchain_huggingface.HuggingFaceEmbeddings failed: {e}")
        print(f"   Error type: {type(e)}")
        
        # Method 2: Use langchain_community
        print("\n3.2 Trying to use langchain_community.embeddings.HuggingFaceEmbeddings...")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            print("✅ langchain_community.embeddings.HuggingFaceEmbeddings created successfully")
            
            # Test embedding
            embedding = embeddings.embed_query(test_text)
            print(f"   Test embedding dimension: {len(embedding)}")
            
        except Exception as e:
            print(f"❌ langchain_community.embeddings.HuggingFaceEmbeddings failed: {e}")
            print(f"   Error type: {type(e)}")
            
            # Method 3: Direct use of sentence_transformers
            print("\n3.3 Trying to use sentence_transformers directly...")
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
                print("✅ sentence_transformers created successfully")
                
                # Test embedding
                embedding = model.encode([test_text])[0]
                print(f"   Test embedding dimension: {len(embedding)}")
                
            except Exception as e:
                print(f"❌ sentence_transformers failed: {e}")
                print(f"   Error type: {type(e)}")
                return False
    
    # 4. Test vector search
    print("\n4. Testing vector search...")
    try:
        # Use simple query text
        query_text = "I want to book a hotel"
        print(f"   Query text: {query_text}")
        
        results = vector_store.similarity_search(query_text, k=2)
        print(f"✅ Vector search successful, returned {len(results)} results")
        
        # Display detailed information of search results
        for i, doc in enumerate(results):
            print(f"   Result {i+1}:")
            print(f"     Content length: {len(doc.page_content)}")
            print(f"     Content preview: {doc.page_content[:100]}...")
            print(f"     Metadata keys: {list(doc.metadata.keys()) if hasattr(doc, 'metadata') else 'None'}")
            
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        print(f"   Error type: {type(e)}")
        
        # Print detailed error stack
        import traceback
        print("\nDetailed error stack:")
        traceback.print_exc()
        return False
    
    # 5. Check environment information
    print("\n5. Environment information:")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
    except:
        print("   PyTorch not installed")
    
    try:
        import transformers
        print(f"   Transformers version: {transformers.__version__}")
    except:
        print("   Transformers not installed")
    
    try:
        import sentence_transformers
        print(f"   Sentence-Transformers version: {sentence_transformers.__version__}")
    except:
        print("   Sentence-Transformers not installed")
    
    try:
        import langchain
        print(f"   Langchain version: {langchain.__version__}")
    except:
        print("   Langchain not installed")
    
    try:
        import faiss
        print(f"   FAISS version: {faiss.__version__}")
    except:
        print("   FAISS not installed")
    
    print("\n=== Test completed ===")
    return True

def test_simple_vector_search():
    """Simplified vector search test"""
    print("\n=== Simplified test ===")
    
    try:
        print("1. Loading vector database...")
        with open("multiwoz-context-db.vec", "rb") as f:
            vector_store = pickle.load(f)
        
        print("2. Trying to search directly without embeddings...")
        # If vector database already contains embeddings, may not need to recreate
        results = vector_store.similarity_search("book hotel", k=1)
        print(f"✅ Simplified search successful: {len(results)} results")
        return True
        
    except Exception as e:
        print(f"❌ Simplified search failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_search()
    
    if not success:
        print("\nTrying simplified test...")
        test_simple_vector_search() 
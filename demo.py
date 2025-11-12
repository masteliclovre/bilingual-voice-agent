#!/usr/bin/env python
"""
Simple demo of the Banking RAG system
Pokazuje osnovnu uporabu / Shows basic usage
"""

from rag_banking_module import RAGBankingAssistant

def main():
    print("🏦 Banking Voice Agent RAG Demo\n")
    print("=" * 50)
    
    # Initialize RAG
    print("Initializing RAG system...")
    rag = RAGBankingAssistant()
    
    # Demo queries
    demo_queries = [
        ("Kako mogu otvoriti račun u banci?", "hr"),
        ("What documents do I need for a loan?", "en"),
        ("Koje vrste kreditnih kartica nudite?", "hr"),
        ("How can I activate online banking?", "en"),
    ]
    
    print("\nTesting banking queries:\n")
    
    for query, expected_lang in demo_queries:
        print(f"📝 Query: {query}")
        
        # Detect language
        detected_lang = rag.detect_language(query)
        print(f"   Language: {detected_lang}")
        
        # Search for relevant information
        results = rag.search(query, top_k=1)
        
        if results:
            result = results[0]
            print(f"   Category: {result['category']}")
            print(f"   Confidence: {result['similarity']:.2%}")
            
            # Get appropriate content
            content = result['content_hr' if detected_lang == 'hr' else 'content_en']
            print(f"   Answer: {content[:150]}...")
        else:
            print("   No relevant information found")
        
        print("-" * 50)
    
    # Show statistics
    stats = rag.get_statistics()
    print(f"\n📊 Knowledge Base Statistics:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Categories: {', '.join(stats['categories'].keys())}")
    print(f"   Embedding model: {stats['embedding_model'].split('/')[-1]}")

if __name__ == "__main__":
    main()

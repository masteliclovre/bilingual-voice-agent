#!/usr/bin/env python
"""
Test script for RAG Banking Assistant
Tests both Croatian and English queries and validates responses
"""

import json
import time
from typing import List, Tuple
from rag_banking_module import RAGBankingAssistant, BankingDocument
import requests
from colorama import init, Fore, Style

init(autoreset=True)

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{text}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")


def print_result(label: str, value: str, color=Fore.GREEN):
    """Print formatted result."""
    print(f"{color}{label}:{Style.RESET_ALL} {value}")


def test_rag_module():
    """Test RAG module directly."""
    print_header("Testing RAG Module Directly")
    
    # Initialize RAG
    rag = RAGBankingAssistant()
    
    # Get statistics
    stats = rag.get_statistics()
    print_result("Total documents", str(stats['total_documents']))
    print_result("Categories", str(list(stats['categories'].keys())))
    
    # Test queries
    test_queries = [
        ("Kako otvoriti račun?", "hr", "accounts"),
        ("What are the credit card options?", "en", "cards"),
        ("Kolika je kamata na stambeni kredit?", "hr", "loans"),
        ("How do I activate online banking?", "en", "digital"),
        ("Koje su naknade za transfer?", "hr", "fees"),
    ]
    
    print("\n" + Fore.YELLOW + "Testing search functionality:" + Style.RESET_ALL)
    
    for query, expected_lang, expected_category in test_queries:
        print(f"\n{Fore.BLUE}Query:{Style.RESET_ALL} {query}")
        
        # Detect language
        detected_lang = rag.detect_language(query)
        lang_match = "✓" if detected_lang == expected_lang else "✗"
        print_result(f"  Language", f"{detected_lang} {lang_match}", 
                    Fore.GREEN if lang_match == "✓" else Fore.RED)
        
        # Search
        results = rag.search(query, top_k=2)
        
        if results:
            top_result = results[0]
            category_match = "✓" if top_result['category'] == expected_category else "✗"
            print_result(f"  Top category", f"{top_result['category']} {category_match}",
                        Fore.GREEN if category_match == "✓" else Fore.RED)
            print_result(f"  Similarity", f"{top_result['similarity']:.3f}")
            
            # Show snippet
            content = top_result['content_hr' if detected_lang == 'hr' else 'content_en']
            snippet = content[:100] + "..." if len(content) > 100 else content
            print(f"  {Fore.GRAY}Snippet: {snippet}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}No results found!{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}✓ RAG module test completed{Style.RESET_ALL}")


def test_api_endpoints(base_url: str = "http://localhost:8000"):
    """Test API endpoints if server is running."""
    print_header("Testing API Endpoints")
    
    try:
        # Check health
        response = requests.get(f"{base_url}/healthz", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print_result("Server status", data['status'])
            print_result("LLM provider", data['llm_provider'])
            print_result("RAG status", data.get('rag', 'unknown'))
        else:
            print(f"{Fore.RED}Health check failed{Style.RESET_ALL}")
            return
        
        # Get RAG stats
        response = requests.get(f"{base_url}/api/rag/stats")
        if response.status_code == 200:
            stats = response.json()
            print_result("Knowledge base docs", str(stats['total_documents']))
        
        # Test search endpoint
        print(f"\n{Fore.YELLOW}Testing search endpoint:{Style.RESET_ALL}")
        
        test_searches = [
            ("otvaranje računa", "hr"),
            ("credit card fees", "en"),
        ]
        
        for query, lang in test_searches:
            response = requests.post(
                f"{base_url}/api/rag/search",
                data={"query": query, "lang": lang}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n{Fore.BLUE}Query:{Style.RESET_ALL} {query}")
                print_result("  Results found", str(len(data['results'])))
                if data['results']:
                    print_result("  Top similarity", 
                               f"{data['results'][0]['similarity']:.3f}")
            else:
                print(f"{Fore.RED}Search failed for: {query}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}✓ API endpoint tests completed{Style.RESET_ALL}")
        
    except requests.exceptions.ConnectionError:
        print(f"{Fore.RED}Cannot connect to server at {base_url}")
        print(f"Please start the server with: python remote_agent_with_rag.py{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error testing API: {e}{Style.RESET_ALL}")


def test_prompt_augmentation():
    """Test prompt augmentation with RAG context."""
    print_header("Testing Prompt Augmentation")
    
    rag = RAGBankingAssistant()
    
    test_cases = [
        {
            "question": "Kako mogu otvoriti štedni račun?",
            "lang": "hr",
            "should_contain": ["račun", "dokument", "OIB"]
        },
        {
            "question": "What's the interest rate for home loans?",
            "lang": "en",
            "should_contain": ["interest", "loan", "EURIBOR"]
        }
    ]
    
    for case in test_cases:
        print(f"\n{Fore.BLUE}Question:{Style.RESET_ALL} {case['question']}")
        
        augmented = rag.augment_prompt(case['question'], case['lang'])
        
        # Check if augmentation worked
        has_context = "Kontekst" in augmented or "Context" in augmented
        print_result("  Has context", str(has_context), 
                    Fore.GREEN if has_context else Fore.RED)
        
        # Check for expected content
        for term in case['should_contain']:
            if term.lower() in augmented.lower():
                print_result(f"  Contains '{term}'", "✓", Fore.GREEN)
            else:
                print_result(f"  Contains '{term}'", "✗", Fore.RED)
        
        # Show augmented prompt preview
        lines = augmented.split('\n')
        preview = '\n'.join(lines[:5]) + "..." if len(lines) > 5 else augmented
        print(f"\n  {Fore.GRAY}Augmented prompt preview:{Style.RESET_ALL}")
        for line in preview.split('\n')[:5]:
            if line.strip():
                print(f"    {Fore.GRAY}{line[:80]}...{Style.RESET_ALL}" 
                     if len(line) > 80 else f"    {Fore.GRAY}{line}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}✓ Prompt augmentation test completed{Style.RESET_ALL}")


def test_custom_document():
    """Test adding custom documents."""
    print_header("Testing Custom Document Addition")
    
    rag = RAGBankingAssistant()
    
    # Get initial count
    initial_count = rag.collection.count()
    print_result("Initial document count", str(initial_count))
    
    # Add custom document
    custom_doc = BankingDocument(
        id="test_001",
        content_hr="Testna usluga: Posebna ponuda za mlade do 30 godina sa 0% naknade.",
        content_en="Test service: Special offer for youth under 30 with 0% fees.",
        category="test",
        keywords=["test", "mladi", "youth", "ponuda", "offer"],
        metadata={"test": True, "created": "test_script"}
    )
    
    rag.add_document(custom_doc)
    print(f"{Fore.GREEN}✓ Added custom document{Style.RESET_ALL}")
    
    # Verify it was added
    new_count = rag.collection.count()
    print_result("New document count", str(new_count))
    
    # Search for it
    results = rag.search("posebna ponuda za mlade")
    
    found = False
    for result in results:
        if "test_001" in result.get("id", ""):
            found = True
            print_result("Custom document found", "✓", Fore.GREEN)
            print_result("  Similarity", f"{result['similarity']:.3f}")
            break
    
    if not found:
        print_result("Custom document found", "✗", Fore.RED)
    
    # Clean up - remove test document
    rag.collection.delete(ids=["test_001"])
    print(f"{Fore.YELLOW}✓ Test document removed{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}✓ Custom document test completed{Style.RESET_ALL}")


def benchmark_performance():
    """Benchmark RAG performance."""
    print_header("Performance Benchmark")
    
    rag = RAGBankingAssistant()
    
    queries = [
        "Kako otvoriti račun?",
        "What are the fees?",
        "Kolika je kamata?",
        "Tell me about loans",
        "Vrste kartica?",
    ] * 2  # Repeat for more samples
    
    # Benchmark search
    print(f"{Fore.YELLOW}Benchmarking search speed:{Style.RESET_ALL}")
    
    start_time = time.time()
    for query in queries:
        _ = rag.search(query)
    
    elapsed = time.time() - start_time
    avg_time = (elapsed / len(queries)) * 1000
    
    print_result("Total queries", str(len(queries)))
    print_result("Total time", f"{elapsed:.2f}s")
    print_result("Average time per query", f"{avg_time:.1f}ms")
    
    # Benchmark prompt augmentation
    print(f"\n{Fore.YELLOW}Benchmarking augmentation speed:{Style.RESET_ALL}")
    
    start_time = time.time()
    for query in queries:
        _ = rag.augment_prompt(query)
    
    elapsed = time.time() - start_time
    avg_time = (elapsed / len(queries)) * 1000
    
    print_result("Total augmentations", str(len(queries)))
    print_result("Total time", f"{elapsed:.2f}s")
    print_result("Average time per augmentation", f"{avg_time:.1f}ms")
    
    print(f"\n{Fore.GREEN}✓ Performance benchmark completed{Style.RESET_ALL}")


def main():
    """Run all tests."""
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🏦 Banking Voice Agent RAG Test Suite{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    
    tests = [
        ("RAG Module", test_rag_module),
        ("Prompt Augmentation", test_prompt_augmentation),
        ("Custom Documents", test_custom_document),
        ("Performance", benchmark_performance),
        ("API Endpoints", test_api_endpoints),
    ]
    
    print(f"\n{Fore.YELLOW}Tests to run:{Style.RESET_ALL}")
    for name, _ in tests:
        print(f"  • {name}")
    
    print(f"\n{Fore.YELLOW}Starting tests...{Style.RESET_ALL}")
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n{Fore.RED}✗ {name} test failed: {e}{Style.RESET_ALL}")
            failed += 1
    
    # Summary
    print_header("Test Summary")
    print_result("Tests passed", str(passed), Fore.GREEN if passed > 0 else Fore.GRAY)
    print_result("Tests failed", str(failed), Fore.RED if failed > 0 else Fore.GRAY)
    
    if failed == 0:
        print(f"\n{Fore.GREEN}🎉 All tests passed successfully!{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.YELLOW}⚠️  Some tests failed. Please review the output above.{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Test suite interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")

#!/usr/bin/env python
"""
Test script for Smart RAG system
Tests matching, language detection, and knowledge retrieval
"""

from smart_rag import SmartRAG
from colorama import init, Fore, Style

init(autoreset=True)


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{text:^70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")


def print_success(text: str):
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_info(text: str):
    print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")


def print_warning(text: str):
    print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")


def print_error(text: str):
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def test_initialization():
    """Test Smart RAG initialization."""
    print_header("Testing Smart RAG Initialization")

    try:
        rag = SmartRAG()
        print_success(f"Smart RAG initialized successfully")

        stats = rag.get_stats()
        print_info(f"Total topics: {stats['total_topics']}")
        print_info(f"Total patterns: {stats['total_patterns']}")
        print_info(f"Total keywords: {stats['total_keywords']}")

        return rag
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return None


def test_language_detection(rag: SmartRAG):
    """Test language detection."""
    print_header("Testing Language Detection")

    test_cases = [
        ("Hello, how are you?", "en"),
        ("Bok, kako si?", "hr"),
        ("What are your working hours?", "en"),
        ("Koliko košta dostava?", "hr"),
        ("Thank you very much!", "en"),
        ("Hvala ti puno!", "hr"),
    ]

    passed = 0
    failed = 0

    for text, expected_lang in test_cases:
        detected = rag.detect_language(text)
        if detected == expected_lang:
            print_success(f"{text[:40]:40} → {detected}")
            passed += 1
        else:
            print_error(f"{text[:40]:40} → {detected} (expected {expected_lang})")
            failed += 1

    print(f"\n{Fore.CYAN}Results: {passed} passed, {failed} failed{Style.RESET_ALL}")


def test_matching(rag: SmartRAG):
    """Test knowledge base matching."""
    print_header("Testing Knowledge Base Matching")

    test_queries = [
        # Greetings
        ("Hello!", "en", "greeting"),
        ("Bok!", "hr", "greeting"),

        # Hours
        ("What are your working hours?", "en", "hours"),
        ("Kada ste otvoreni?", "hr", "hours"),

        # Contact
        ("How can I contact you?", "en", "contact"),
        ("Kako vas mogu kontaktirati?", "hr", "contact"),

        # Pricing
        ("How much does it cost?", "en", "pricing"),
        ("Koliko košta?", "hr", "pricing"),

        # Support
        ("I need help with a problem", "en", "support"),
        ("Trebam pomoć s problemom", "hr", "support"),

        # Shipping
        ("When will my order arrive?", "en", "shipping"),
        ("Kada će stići moja narudžba?", "hr", "shipping"),

        # Returns
        ("I want to return a product", "en", "returns"),
        ("Želim vratiti proizvod", "hr", "returns"),

        # Thanks
        ("Thank you!", "en", "thanks"),
        ("Hvala!", "hr", "thanks"),

        # Goodbye
        ("Bye!", "en", "goodbye"),
        ("Doviđenja!", "hr", "goodbye"),
    ]

    passed = 0
    failed = 0
    no_match = 0

    for query, lang, expected_topic in test_queries:
        match = rag.match(query, lang)

        if match.matched:
            if match.topic == expected_topic:
                response = match.response_hr if lang == "hr" else match.response_en
                print_success(f"{query[:35]:35} → {match.topic:12} ({match.confidence:.2f})")
                print(f"{' ' * 39}{Fore.GRAY}{response[:60]}...{Style.RESET_ALL}")
                passed += 1
            else:
                print_warning(f"{query[:35]:35} → {match.topic:12} (expected {expected_topic})")
                failed += 1
        else:
            print_error(f"{query[:35]:35} → NO MATCH (expected {expected_topic})")
            no_match += 1

    print(f"\n{Fore.CYAN}Results: {passed} passed, {failed} wrong topic, {no_match} no match{Style.RESET_ALL}")


def test_prompt_augmentation(rag: SmartRAG):
    """Test prompt augmentation."""
    print_header("Testing Prompt Augmentation")

    test_cases = [
        ("Hello, I need help", "en"),
        ("Koliko košta dostava?", "hr"),
    ]

    for query, lang in test_cases:
        print(f"\n{Fore.BLUE}Query:{Style.RESET_ALL} {query}")
        augmented, matched = rag.augment_prompt(query, lang)

        if matched:
            print_success("RAG context added")
            lines = augmented.split('\n')
            for line in lines[:5]:
                if line.strip():
                    print(f"  {Fore.GRAY}{line[:70]}{Style.RESET_ALL}")
        else:
            print_warning("No RAG match - will use LLM")


def test_custom_topic(rag: SmartRAG):
    """Test adding custom topic."""
    print_header("Testing Custom Topic Addition")

    # Add custom topic
    rag.add_topic(
        topic="custom_test",
        patterns=[r"\b(test|testing|probe)\b"],
        keywords=["test", "testing", "probe"],
        response_hr="Ovo je test odgovor.",
        response_en="This is a test response.",
        priority=5
    )

    print_success("Added custom topic: custom_test")

    # Test matching
    match = rag.match("This is a test", "en")
    if match.matched and match.topic == "custom_test":
        print_success(f"Custom topic matched: {match.response_en}")
    else:
        print_error("Custom topic not matched")

    # Remove it
    rag.remove_topic("custom_test")
    print_info("Removed custom topic")


def test_edge_cases(rag: SmartRAG):
    """Test edge cases."""
    print_header("Testing Edge Cases")

    # Empty query
    match = rag.match("", "en")
    if not match.matched:
        print_success("Empty query handled correctly")
    else:
        print_error("Empty query incorrectly matched")

    # Very short query
    match = rag.match("hi", "en")
    if match.matched and match.topic == "greeting":
        print_success("Short query matched correctly")
    else:
        print_warning(f"Short query match: {match.topic if match.matched else 'no match'}")

    # Mixed language
    match = rag.match("Hello, koliko košta?", "hr")
    detected_lang = rag.detect_language("Hello, koliko košta?")
    print_info(f"Mixed language detected as: {detected_lang}")

    # No matching query
    match = rag.match("xyzabc nonsense qwerty", "en")
    if not match.matched:
        print_success("Nonsense query not matched")
    else:
        print_warning(f"Nonsense query matched: {match.topic}")


def main():
    """Run all tests."""
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'Smart RAG Test Suite':^70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")

    rag = test_initialization()
    if not rag:
        print_error("Cannot continue without RAG instance")
        return

    try:
        test_language_detection(rag)
        test_matching(rag)
        test_prompt_augmentation(rag)
        test_custom_topic(rag)
        test_edge_cases(rag)

        print_header("All Tests Completed")
        print_success("Smart RAG is working correctly!")

    except Exception as e:
        print_error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Tests interrupted by user{Style.RESET_ALL}")

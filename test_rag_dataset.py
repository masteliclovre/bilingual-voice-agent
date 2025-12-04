#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive test dataset for RAG system
50+ customer support FAQ questions in Croatian and English
"""

# Test dataset with diverse customer support queries
test_dataset = [
    # GREETING - 5 queries
    {
        "query": "Hello, how are you?",
        "query_hr": "Bok, kako si?",
        "expected_topics": ["greeting"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "greeting"
    },
    {
        "query": "Good morning!",
        "query_hr": "Dobro jutro!",
        "expected_topics": ["greeting"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "greeting"
    },
    {
        "query": "Hi there, I need some help",
        "query_hr": "Bok, trebam pomoć",
        "expected_topics": ["greeting", "support"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "greeting"
    },
    {
        "query": "Pozdrav! Kako mogu započeti?",
        "query_hr": "Pozdrav! Kako mogu započeti?",
        "expected_topics": ["greeting"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "low",
        "category": "greeting"
    },
    {
        "query": "Ćao, jeste li tu?",
        "query_hr": "Ćao, jeste li tu?",
        "expected_topics": ["greeting"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "low",
        "category": "greeting"
    },

    # WORKING HOURS - 8 queries
    {
        "query": "What are your working hours?",
        "query_hr": "Koje su vaše radne sate?",
        "expected_topics": ["hours"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "hours"
    },
    {
        "query": "When are you open on Saturday?",
        "query_hr": "Kada ste otvoreni subotom?",
        "expected_topics": ["hours"],
        "expected_entities": {"temporal": "Saturday"},
        "lang": "en",
        "complexity": "medium",
        "category": "hours"
    },
    {
        "query": "Do you work on Sundays?",
        "query_hr": "Radite li nedjeljom?",
        "expected_topics": ["hours"],
        "expected_entities": {"temporal": "Sunday"},
        "lang": "en",
        "complexity": "medium",
        "category": "hours"
    },
    {
        "query": "U koliko sati otvarate?",
        "query_hr": "U koliko sati otvarate?",
        "expected_topics": ["hours"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "low",
        "category": "hours"
    },
    {
        "query": "Radno vrijeme petkom?",
        "query_hr": "Radno vrijeme petkom?",
        "expected_topics": ["hours"],
        "expected_entities": {"temporal": "Friday"},
        "lang": "hr",
        "complexity": "medium",
        "category": "hours"
    },
    {
        "query": "Do kada ste danas otvoreni?",
        "query_hr": "Do kada ste danas otvoreni?",
        "expected_topics": ["hours"],
        "expected_entities": {"temporal": "today"},
        "lang": "hr",
        "complexity": "medium",
        "category": "hours"
    },
    {
        "query": "Are you open now?",
        "query_hr": "Jeste li sada otvoreni?",
        "expected_topics": ["hours"],
        "expected_entities": {"temporal": "now"},
        "lang": "en",
        "complexity": "medium",
        "category": "hours"
    },
    {
        "query": "What time do you close on weekdays?",
        "query_hr": "U koliko zatvarate radnim danom?",
        "expected_topics": ["hours"],
        "expected_entities": {"temporal": "weekdays"},
        "lang": "en",
        "complexity": "medium",
        "category": "hours"
    },

    # CONTACT - 6 queries
    {
        "query": "How can I contact you?",
        "query_hr": "Kako vas mogu kontaktirati?",
        "expected_topics": ["contact"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "contact"
    },
    {
        "query": "What's your phone number?",
        "query_hr": "Koji je vaš broj telefona?",
        "expected_topics": ["contact"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "contact"
    },
    {
        "query": "Vaša email adresa?",
        "query_hr": "Vaša email adresa?",
        "expected_topics": ["contact"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "low",
        "category": "contact"
    },
    {
        "query": "I need to call customer support",
        "query_hr": "Trebam nazvati korisničku podršku",
        "expected_topics": ["contact", "support"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "contact"
    },
    {
        "query": "Kako mogu doći do vas?",
        "query_hr": "Kako mogu doći do vas?",
        "expected_topics": ["contact", "location"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "medium",
        "category": "contact"
    },
    {
        "query": "Can I reach you via email?",
        "query_hr": "Mogu li vas kontaktirati emailom?",
        "expected_topics": ["contact"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "contact"
    },

    # PRICING - 7 queries
    {
        "query": "How much does it cost?",
        "query_hr": "Koliko košta?",
        "expected_topics": ["pricing"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "pricing"
    },
    {
        "query": "What are your prices?",
        "query_hr": "Koje su vaše cijene?",
        "expected_topics": ["pricing"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "pricing"
    },
    {
        "query": "Koliko moram platiti?",
        "query_hr": "Koliko moram platiti?",
        "expected_topics": ["pricing"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "low",
        "category": "pricing"
    },
    {
        "query": "What's the fee for basic package?",
        "query_hr": "Koliko košta osnovni paket?",
        "expected_topics": ["pricing"],
        "expected_entities": {"product": "basic package"},
        "lang": "en",
        "complexity": "medium",
        "category": "pricing"
    },
    {
        "query": "Cijena mjesečne pretplate?",
        "query_hr": "Cijena mjesečne pretplate?",
        "expected_topics": ["pricing"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "medium",
        "category": "pricing"
    },
    {
        "query": "Is there a discount?",
        "query_hr": "Imate li popust?",
        "expected_topics": ["pricing"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "pricing"
    },
    {
        "query": "How much is the fee?",
        "query_hr": "Kolika je naknada?",
        "expected_topics": ["pricing"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "pricing"
    },

    # SHIPPING/DELIVERY - 8 queries
    {
        "query": "When will my order arrive?",
        "query_hr": "Kada će stići moja narudžba?",
        "expected_topics": ["shipping"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "shipping"
    },
    {
        "query": "How long does delivery take?",
        "query_hr": "Koliko traje dostava?",
        "expected_topics": ["shipping"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "shipping"
    },
    {
        "query": "Koliko košta dostava u Zagreb?",
        "query_hr": "Koliko košta dostava u Zagreb?",
        "expected_topics": ["shipping", "pricing"],
        "expected_entities": {"location": "Zagreb"},
        "lang": "hr",
        "complexity": "high",
        "category": "shipping"
    },
    {
        "query": "Do you have express delivery?",
        "query_hr": "Imate li brzu dostavu?",
        "expected_topics": ["shipping"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "shipping"
    },
    {
        "query": "Tracking broj za moj paket?",
        "query_hr": "Tracking broj za moj paket?",
        "expected_topics": ["shipping"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "medium",
        "category": "shipping"
    },
    {
        "query": "Where's my package?",
        "query_hr": "Gdje je moj paket?",
        "expected_topics": ["shipping"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "shipping"
    },
    {
        "query": "Can I get same-day delivery?",
        "query_hr": "Mogu li dobiti dostavu isti dan?",
        "expected_topics": ["shipping"],
        "expected_entities": {"temporal": "same-day"},
        "lang": "en",
        "complexity": "medium",
        "category": "shipping"
    },
    {
        "query": "Kad će stići ako naručim danas?",
        "query_hr": "Kad će stići ako naručim danas?",
        "expected_topics": ["shipping"],
        "expected_entities": {"temporal": "today"},
        "lang": "hr",
        "complexity": "high",
        "category": "shipping"
    },

    # RETURNS/REFUNDS - 6 queries
    {
        "query": "Can I return this?",
        "query_hr": "Mogu li vratiti ovo?",
        "expected_topics": ["returns"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "returns"
    },
    {
        "query": "How do I get a refund?",
        "query_hr": "Kako dobiti refundaciju?",
        "expected_topics": ["returns"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "returns"
    },
    {
        "query": "Politika povrata robe?",
        "query_hr": "Politika povrata robe?",
        "expected_topics": ["returns"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "low",
        "category": "returns"
    },
    {
        "query": "I want my money back",
        "query_hr": "Želim novac natrag",
        "expected_topics": ["returns"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "returns"
    },
    {
        "query": "Koliko dana imam za povrat?",
        "query_hr": "Koliko dana imam za povrat?",
        "expected_topics": ["returns"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "medium",
        "category": "returns"
    },
    {
        "query": "Cancel my order please",
        "query_hr": "Molim otkaži moju narudžbu",
        "expected_topics": ["returns"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "returns"
    },

    # PAYMENT - 5 queries
    {
        "query": "What payment methods do you accept?",
        "query_hr": "Koje načine plaćanja prihvaćate?",
        "expected_topics": ["payment"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "payment"
    },
    {
        "query": "Can I pay with credit card?",
        "query_hr": "Mogu li platiti karticom?",
        "expected_topics": ["payment"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "payment"
    },
    {
        "query": "Prihvaćate li PayPal?",
        "query_hr": "Prihvaćate li PayPal?",
        "expected_topics": ["payment"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "medium",
        "category": "payment"
    },
    {
        "query": "Kako mogu platiti?",
        "query_hr": "Kako mogu platiti?",
        "expected_topics": ["payment"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "low",
        "category": "payment"
    },
    {
        "query": "Do you accept cash on delivery?",
        "query_hr": "Imate li plaćanje pri preuzimanju?",
        "expected_topics": ["payment"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "payment"
    },

    # TECHNICAL SUPPORT - 6 queries
    {
        "query": "I have a problem with my account",
        "query_hr": "Imam problem s mojim računom",
        "expected_topics": ["support"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "support"
    },
    {
        "query": "Something is not working",
        "query_hr": "Nešto ne radi",
        "expected_topics": ["support"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "support"
    },
    {
        "query": "Trebam tehničku pomoć",
        "query_hr": "Trebam tehničku pomoć",
        "expected_topics": ["support"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "low",
        "category": "support"
    },
    {
        "query": "Help! It's broken",
        "query_hr": "Pomoć! Pokvareno je",
        "expected_topics": ["support"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "support"
    },
    {
        "query": "Javlja mi grešku",
        "query_hr": "Javlja mi grešku",
        "expected_topics": ["support"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "medium",
        "category": "support"
    },
    {
        "query": "Technical support number?",
        "query_hr": "Broj za tehničku podršku?",
        "expected_topics": ["support", "contact"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "support"
    },

    # LOCATION - 4 queries
    {
        "query": "Where are you located?",
        "query_hr": "Gdje se nalazite?",
        "expected_topics": ["location"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "location"
    },
    {
        "query": "Your address?",
        "query_hr": "Vaša adresa?",
        "expected_topics": ["location"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "location"
    },
    {
        "query": "Jeste li u Zagrebu?",
        "query_hr": "Jeste li u Zagrebu?",
        "expected_topics": ["location"],
        "expected_entities": {"location": "Zagreb"},
        "lang": "hr",
        "complexity": "medium",
        "category": "location"
    },
    {
        "query": "How do I find your office?",
        "query_hr": "Kako pronaći vaš ured?",
        "expected_topics": ["location"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "location"
    },

    # COURTESY (THANKS/GOODBYE) - 4 queries
    {
        "query": "Thank you so much!",
        "query_hr": "Hvala vam puno!",
        "expected_topics": ["thanks"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "thanks"
    },
    {
        "query": "I appreciate your help",
        "query_hr": "Cijenim vašu pomoć",
        "expected_topics": ["thanks"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "medium",
        "category": "thanks"
    },
    {
        "query": "Doviđenja!",
        "query_hr": "Doviđenja!",
        "expected_topics": ["goodbye"],
        "expected_entities": {},
        "lang": "hr",
        "complexity": "low",
        "category": "goodbye"
    },
    {
        "query": "See you later!",
        "query_hr": "Vidimo se!",
        "expected_topics": ["goodbye"],
        "expected_entities": {},
        "lang": "en",
        "complexity": "low",
        "category": "goodbye"
    }
]

# Statistics
def get_dataset_stats():
    """Get statistics about the test dataset."""
    stats = {
        "total_queries": len(test_dataset),
        "by_language": {"en": 0, "hr": 0},
        "by_complexity": {"low": 0, "medium": 0, "high": 0},
        "by_category": {},
        "multi_topic_queries": 0
    }

    for item in test_dataset:
        stats["by_language"][item["lang"]] += 1
        stats["by_complexity"][item["complexity"]] += 1

        category = item["category"]
        stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

        if len(item["expected_topics"]) > 1:
            stats["multi_topic_queries"] += 1

    return stats


if __name__ == "__main__":
    import sys
    import io

    # Fix Windows console encoding
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "=" * 70)
    print("TEST DATASET STATISTICS")
    print("=" * 70)

    stats = get_dataset_stats()

    print(f"\nTotal queries: {stats['total_queries']}")
    print(f"\nBy language:")
    for lang, count in stats['by_language'].items():
        print(f"  - {lang.upper()}: {count} queries")

    print(f"\nBy complexity:")
    for complexity, count in stats['by_complexity'].items():
        print(f"  - {complexity.capitalize()}: {count} queries")

    print(f"\nBy category:")
    for category, count in sorted(stats['by_category'].items()):
        print(f"  - {category.capitalize()}: {count} queries")

    print(f"\nMulti-topic queries: {stats['multi_topic_queries']}")

    print("\n" + "=" * 70)
    print("Sample queries:")
    print("=" * 70)

    for i, item in enumerate(test_dataset[:5], 1):
        print(f"\n{i}. EN: {item['query']}")
        print(f"   HR: {item['query_hr']}")
        print(f"   Expected topics: {', '.join(item['expected_topics'])}")
        print(f"   Complexity: {item['complexity']}")

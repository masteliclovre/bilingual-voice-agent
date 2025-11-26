"""
Smart RAG - Lightweight knowledge retrieval for voice agents
No heavy dependencies, instant matching, bilingual support (HR/EN)
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MatchResult:
    """Result of knowledge base matching."""
    matched: bool
    topic: Optional[str] = None
    confidence: float = 0.0
    response_hr: Optional[str] = None
    response_en: Optional[str] = None
    context: Optional[str] = None


class SmartRAG:
    """
    Lightweight RAG system using pattern matching and keyword search.
    Perfect for customer support, FAQ, and domain-specific assistants.
    """

    def __init__(self, knowledge_path: Optional[str] = None):
        """
        Initialize SmartRAG with optional knowledge base JSON file.

        Args:
            knowledge_path: Path to knowledge.json file (optional)
        """
        self.knowledge_base: Dict = {}
        self.enable_fuzzy = True  # Enable fuzzy matching for typos

        if knowledge_path and Path(knowledge_path).exists():
            self.load_knowledge(knowledge_path)
        else:
            # Use default customer support template
            self.knowledge_base = self._default_knowledge_base()

    def _default_knowledge_base(self) -> Dict:
        """Default customer support knowledge base template."""
        return {
            "greeting": {
                "patterns": [
                    r"\b(hello|hi|hey|good morning|good afternoon)\b",
                    r"\b(bok|pozdrav|dobar dan|zdravo|ćao)\b"
                ],
                "keywords": ["hello", "hi", "hey", "bok", "pozdrav", "dobar dan"],
                "responses": {
                    "hr": "Pozdrav! Ja sam vaš virtualni asistent. Kako vam mogu pomoći danas?",
                    "en": "Hello! I'm your virtual assistant. How can I help you today?"
                },
                "priority": 10
            },

            "hours": {
                "patterns": [
                    r"\b(working hours|business hours|when.*open|opening hours)\b",
                    r"\b(radno vrijeme|kada.*otvoreno|u koliko sati)\b"
                ],
                "keywords": ["hours", "open", "vrijeme", "otvoreno", "working"],
                "responses": {
                    "hr": "Naše radno vrijeme je: Ponedjeljak-Petak 8:00-20:00, Subota 9:00-14:00. Nedjelja zatvoreno.",
                    "en": "Our working hours are: Monday-Friday 8:00-20:00, Saturday 9:00-14:00. Closed on Sunday."
                },
                "priority": 8
            },

            "contact": {
                "patterns": [
                    r"\b(contact|email|phone|call|reach)\b",
                    r"\b(kontakt|email|telefon|broj|zvati|dohvat)\b"
                ],
                "keywords": ["contact", "email", "phone", "kontakt", "telefon"],
                "responses": {
                    "hr": "Možete nas kontaktirati na: Telefon: 0800-1234 (besplatno), Email: info@company.com, ili posjetite naš ured na adresi Ulica 123, Zagreb.",
                    "en": "You can contact us at: Phone: 0800-1234 (toll-free), Email: info@company.com, or visit our office at Street 123, Zagreb."
                },
                "priority": 9
            },

            "pricing": {
                "patterns": [
                    r"\b(price|cost|how much|pricing|fee)\b",
                    r"\b(cijena|košta|koliko|naknada|trošak)\b"
                ],
                "keywords": ["price", "cost", "cijena", "košta", "fee"],
                "responses": {
                    "hr": "Naše cijene variraju ovisno o usluzi. Osnovni paket počinje od 99 kn mjesečno. Za detaljnu ponudu, molimo kontaktirajte naš tim.",
                    "en": "Our prices vary depending on the service. Basic package starts at 99 kn per month. For a detailed quote, please contact our team."
                },
                "priority": 7
            },

            "support": {
                "patterns": [
                    r"\b(help|support|problem|issue|not working)\b",
                    r"\b(pomoć|podrška|problem|ne radi|greška)\b"
                ],
                "keywords": ["help", "support", "problem", "pomoć", "podrška"],
                "responses": {
                    "hr": "Žao mi je čuti da imate poteškoća. Naš tehnički tim je dostupan 24/7. Molim vas opišite svoj problem, ili nazovite naš support na 0800-5678.",
                    "en": "I'm sorry to hear you're having trouble. Our technical team is available 24/7. Please describe your issue, or call our support at 0800-5678."
                },
                "priority": 9
            },

            "shipping": {
                "patterns": [
                    r"\b(shipping|delivery|when.*arrive|tracking)\b",
                    r"\b(dostava|isporuka|kada.*stići|praćenje)\b"
                ],
                "keywords": ["shipping", "delivery", "dostava", "isporuka"],
                "responses": {
                    "hr": "Standardna dostava traje 2-3 radna dana. Brza dostava je dostupna uz nadoplatu (24h). Dobit ćete tracking broj putem email-a.",
                    "en": "Standard delivery takes 2-3 business days. Express delivery available for extra fee (24h). You'll receive a tracking number via email."
                },
                "priority": 7
            },

            "returns": {
                "patterns": [
                    r"\b(return|refund|money back|cancel order)\b",
                    r"\b(povrat|refundacija|vrati.*novac|otkaži.*narudžb)\b"
                ],
                "keywords": ["return", "refund", "povrat", "otkazivanje"],
                "responses": {
                    "hr": "Nudimo povrat robe u roku od 14 dana. Proizvod mora biti neoštećen i u originalnom pakovanju. Puni povrat novca u roku od 5-7 dana.",
                    "en": "We offer returns within 14 days. Product must be undamaged and in original packaging. Full refund within 5-7 days."
                },
                "priority": 8
            },

            "payment": {
                "patterns": [
                    r"\b(payment.*method|how.*pay|credit card|paypal)\b",
                    r"\b(način.*plaćanj|kako.*platit|kartic|paypal)\b"
                ],
                "keywords": ["payment", "pay", "plaćanje", "kartica"],
                "responses": {
                    "hr": "Prihvaćamo: Kreditne/debitne kartice (Visa, Mastercard), PayPal, virman, i gotovina pri preuzimanju.",
                    "en": "We accept: Credit/debit cards (Visa, Mastercard), PayPal, bank transfer, and cash on delivery."
                },
                "priority": 7
            },

            "location": {
                "patterns": [
                    r"\b(location|address|where.*located|find you)\b",
                    r"\b(lokacija|adresa|gdje.*nalazit|pronaći)\b"
                ],
                "keywords": ["location", "address", "lokacija", "adresa"],
                "responses": {
                    "hr": "Nalazimo se na adresi: Ulica 123, 10000 Zagreb. Blizu glavnog trga, pored pošte.",
                    "en": "We're located at: Street 123, 10000 Zagreb. Near the main square, next to the post office."
                },
                "priority": 7
            },

            "thanks": {
                "patterns": [
                    r"\b(thank|thanks|appreciate)\b",
                    r"\b(hvala|zahval|fala)\b"
                ],
                "keywords": ["thank", "thanks", "hvala"],
                "responses": {
                    "hr": "Nema na čemu! Rado pomažem. Imate li još neko pitanje?",
                    "en": "You're welcome! Happy to help. Do you have any other questions?"
                },
                "priority": 6
            },

            "goodbye": {
                "patterns": [
                    r"\b(bye|goodbye|see you|talk later)\b",
                    r"\b(doviđenja|ćao|vidimo se|adio)\b"
                ],
                "keywords": ["bye", "goodbye", "doviđenja"],
                "responses": {
                    "hr": "Doviđenja! Lijepo mi je bilo razgovarati s vama. Obratite nam se opet!",
                    "en": "Goodbye! It was nice talking to you. Feel free to reach out again!"
                },
                "priority": 6
            }
        }

    def load_knowledge(self, json_path: str):
        """Load knowledge base from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            print(f"✓ Loaded knowledge base: {len(self.knowledge_base)} topics")
        except Exception as e:
            print(f"⚠️ Failed to load knowledge base: {e}")
            self.knowledge_base = self._default_knowledge_base()

    def save_knowledge(self, json_path: str):
        """Save current knowledge base to JSON file."""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved knowledge base to: {json_path}")

    def detect_language(self, text: str) -> str:
        """
        Detect language (Croatian or English).
        Simple heuristic based on Croatian-specific characters and keywords.
        """
        text_lower = text.lower()

        # Croatian-specific characters
        croatian_chars = ['č', 'ć', 'ž', 'š', 'đ']
        has_croatian_chars = any(char in text for char in croatian_chars)

        # Common Croatian words
        croatian_words = ['je', 'sam', 'su', 'i', 'u', 'na', 'da', 'kako', 'koji']
        croatian_score = sum(1 for word in croatian_words if f' {word} ' in f' {text_lower} ')

        # Common English words
        english_words = ['the', 'is', 'are', 'and', 'in', 'on', 'to', 'how', 'what']
        english_score = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ')

        if has_croatian_chars or croatian_score > english_score:
            return "hr"
        return "en"

    def match(self, user_text: str, lang: Optional[str] = None) -> MatchResult:
        """
        Match user input against knowledge base.

        Args:
            user_text: User's message
            lang: Language hint (hr/en), auto-detected if None

        Returns:
            MatchResult with matched topic and response
        """
        if not user_text or not user_text.strip():
            return MatchResult(matched=False)

        # Detect language if not provided
        if not lang:
            lang = self.detect_language(user_text)

        text_lower = user_text.lower()
        best_match = None
        best_score = 0.0

        # Try each topic in knowledge base
        for topic, data in self.knowledge_base.items():
            score = 0.0

            # Check regex patterns (high weight)
            patterns = data.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 10.0

            # Check keywords (medium weight)
            keywords = data.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 3.0

            # Apply priority multiplier
            priority = data.get("priority", 5)
            score *= (priority / 5.0)

            if score > best_score:
                best_score = score
                best_match = (topic, data)

        # Threshold for matching
        MATCH_THRESHOLD = 3.0

        if best_match and best_score >= MATCH_THRESHOLD:
            topic, data = best_match
            responses = data.get("responses", {})

            return MatchResult(
                matched=True,
                topic=topic,
                confidence=min(best_score / 10.0, 1.0),
                response_hr=responses.get("hr"),
                response_en=responses.get("en"),
                context=self._build_context(topic, data, lang)
            )

        return MatchResult(matched=False)

    def _build_context(self, topic: str, data: Dict, lang: str) -> str:
        """Build context string for LLM prompt augmentation."""
        responses = data.get("responses", {})
        response = responses.get(lang, responses.get("en", ""))

        if lang == "hr":
            return f"[Znanje iz baze - {topic}]: {response}"
        else:
            return f"[Knowledge base - {topic}]: {response}"

    def augment_prompt(self, user_text: str, lang: Optional[str] = None) -> Tuple[str, bool]:
        """
        Augment user prompt with knowledge base context.

        Args:
            user_text: Original user message
            lang: Language hint

        Returns:
            (augmented_prompt, was_matched)
        """
        match = self.match(user_text, lang)

        if not match.matched:
            return user_text, False

        # Get response based on language
        lang = lang or self.detect_language(user_text)
        response = match.response_hr if lang == "hr" else match.response_en

        # Build augmented prompt
        if lang == "hr":
            augmented = f"""Kontekst iz baze znanja:
{response}

Korisničko pitanje: {user_text}

Odgovori na temelju danog konteksta. Možeš dodati dodatne detalje ako je potrebno, ali se drži konteksta."""
        else:
            augmented = f"""Context from knowledge base:
{response}

User question: {user_text}

Answer based on the given context. You can add extra details if needed, but stay within context."""

        return augmented, True

    def add_topic(self, topic: str, patterns: List[str], keywords: List[str],
                  response_hr: str, response_en: str, priority: int = 5):
        """
        Add new topic to knowledge base dynamically.

        Args:
            topic: Topic identifier
            patterns: Regex patterns for matching
            keywords: Keywords for matching
            response_hr: Croatian response
            response_en: English response
            priority: Priority (1-10, higher = more important)
        """
        self.knowledge_base[topic] = {
            "patterns": patterns,
            "keywords": keywords,
            "responses": {
                "hr": response_hr,
                "en": response_en
            },
            "priority": priority
        }
        print(f"✓ Added topic: {topic}")

    def remove_topic(self, topic: str):
        """Remove topic from knowledge base."""
        if topic in self.knowledge_base:
            del self.knowledge_base[topic]
            print(f"✓ Removed topic: {topic}")
        else:
            print(f"⚠️ Topic not found: {topic}")

    def list_topics(self) -> List[str]:
        """Get list of all topics in knowledge base."""
        return list(self.knowledge_base.keys())

    def get_stats(self) -> Dict:
        """Get statistics about knowledge base."""
        total_patterns = sum(len(data.get("patterns", [])) for data in self.knowledge_base.values())
        total_keywords = sum(len(data.get("keywords", [])) for data in self.knowledge_base.values())

        return {
            "total_topics": len(self.knowledge_base),
            "total_patterns": total_patterns,
            "total_keywords": total_keywords,
            "topics": self.list_topics()
        }


# Example usage
if __name__ == "__main__":
    import sys
    import io

    # Fix Windows console encoding
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Initialize RAG
    rag = SmartRAG()

    # Print stats
    stats = rag.get_stats()
    print("\n[*] Knowledge Base Stats:")
    print(f"  Topics: {stats['total_topics']}")
    print(f"  Patterns: {stats['total_patterns']}")
    print(f"  Keywords: {stats['total_keywords']}")
    print(f"\n[*] Available topics:")
    for topic in stats['topics']:
        print(f"  - {topic}")

    # Test queries
    test_queries = [
        ("Hello, I need help", "en"),
        ("Bok, trebam pomoć", "hr"),
        ("What are your working hours?", "en"),
        ("Koliko košta dostava?", "hr"),
        ("How can I return a product?", "en"),
        ("Hvala vam puno!", "hr"),
    ]

    print("\n[*] Testing matching:")
    for query, expected_lang in test_queries:
        print(f"\n[?] Query: {query}")
        match = rag.match(query)
        if match.matched:
            print(f"  [OK] Matched: {match.topic} (confidence: {match.confidence:.2f})")
            response = match.response_hr if expected_lang == "hr" else match.response_en
            print(f"  [>>] Response: {response[:80]}...")
        else:
            print(f"  [X] No match (will use LLM)")

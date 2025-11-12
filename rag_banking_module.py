"""
RAG (Retrieval-Augmented Generation) module for bilingual banking voice agent.
Supports Croatian and English with banking-specific knowledge base.
"""

import os
import json
import pickle
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# Vector database imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# For OpenAI embeddings (optional alternative)
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

# =========================
# Configuration
# =========================

# Embedding model selection
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ChromaDB settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "banking_knowledge")

# RAG settings
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.5"))

# Language detection
CROATIAN_KEYWORDS = ["račun", "kredit", "kartica", "novac", "plaćanje", "štednja", "kamata"]
ENGLISH_KEYWORDS = ["account", "credit", "card", "money", "payment", "savings", "interest"]


@dataclass
class BankingDocument:
    """Banking knowledge document with multilingual support."""
    id: str
    content_hr: str
    content_en: str
    category: str
    keywords: List[str]
    metadata: Dict


class BankingKnowledgeBase:
    """Banking-specific knowledge base with sample Q&A pairs."""
    
    @staticmethod
    def get_sample_documents() -> List[BankingDocument]:
        """Return sample banking documents in both languages."""
        return [
            # Account information
            BankingDocument(
                id="acc_001",
                content_hr="Za otvaranje tekućeg računa potrebni su: važeća osobna iskaznica ili putovnica, dokaz o prihodima (potvrda o zaposlenju ili mirovinski odrezak), i OIB. Postupak traje 15-30 minuta. Minimalni depozit nije potreban. Mjesečno održavanje računa iznosi 10 kuna.",
                content_en="To open a current account you need: valid ID card or passport, proof of income (employment certificate or pension slip), and OIB (tax number). The process takes 15-30 minutes. No minimum deposit required. Monthly maintenance fee is 10 kuna.",
                category="accounts",
                keywords=["račun", "account", "otvaranje", "opening", "dokumenti", "documents"],
                metadata={"type": "procedure", "last_updated": "2024-01"}
            ),
            BankingDocument(
                id="acc_002",
                content_hr="Vrste računa koje nudimo: Tekući račun - za svakodnevne transakcije. Žiro račun - za primanje plaće. Štedni račun - s kamatom do 2% godišnje. Devizni račun - za čuvanje stranih valuta. Studentski račun - bez naknade održavanja za studente do 27 godina.",
                content_en="Types of accounts we offer: Current account - for daily transactions. Salary account - for receiving salary. Savings account - with up to 2% annual interest. Foreign currency account - for keeping foreign currencies. Student account - no maintenance fee for students up to 27 years old.",
                category="accounts",
                keywords=["vrste računa", "account types", "štednja", "savings", "student", "devizni", "foreign currency"],
                metadata={"type": "products", "last_updated": "2024-01"}
            ),
            
            # Credit cards
            BankingDocument(
                id="card_001",
                content_hr="Kreditne kartice: Visa Classic - osnovna kartica s limitom do 15.000 kn. Mastercard Gold - premium kartica s putnim osiguranjem i limitom do 50.000 kn. Kamata na kreditne kartice je 12% godišnje. Grejs period je 45 dana bez kamata ako platite puni iznos.",
                content_en="Credit cards: Visa Classic - basic card with limit up to 15,000 kn. Mastercard Gold - premium card with travel insurance and limit up to 50,000 kn. Credit card interest rate is 12% annually. Grace period is 45 days interest-free if you pay the full amount.",
                category="cards",
                keywords=["kreditna kartica", "credit card", "visa", "mastercard", "kamata", "interest", "limit"],
                metadata={"type": "products", "interest_rate": 0.12}
            ),
            BankingDocument(
                id="card_002",
                content_hr="Debitne kartice izdajemo odmah pri otvaranju računa. Maestro kartica je besplatna. Visa Electron ima godišnju naknadu od 50 kn. Beskontaktno plaćanje do 250 kn bez PIN-a. Dnevni limit podizanja gotovine je 5.000 kn na bankomatima.",
                content_en="Debit cards are issued immediately when opening an account. Maestro card is free. Visa Electron has an annual fee of 50 kn. Contactless payment up to 250 kn without PIN. Daily cash withdrawal limit is 5,000 kn at ATMs.",
                category="cards",
                keywords=["debitna kartica", "debit card", "maestro", "beskontaktno", "contactless", "bankomat", "ATM"],
                metadata={"type": "products", "last_updated": "2024-01"}
            ),
            
            # Loans
            BankingDocument(
                id="loan_001",
                content_hr="Stambeni krediti: Kamatna stopa od 2.5% do 4.5% godišnje ovisno o kreditnoj sposobnosti. Maksimalni rok otplate 30 godina. Potreban učešće minimum 15% vrijednosti nekretnine. Fiksna kamata prve 5 godina, zatim varijabilna vezana uz EURIBOR.",
                content_en="Home loans: Interest rate from 2.5% to 4.5% annually depending on creditworthiness. Maximum repayment period 30 years. Required down payment minimum 15% of property value. Fixed rate first 5 years, then variable tied to EURIBOR.",
                category="loans",
                keywords=["stambeni kredit", "home loan", "mortgage", "kamata", "interest rate", "nekretnina", "property"],
                metadata={"type": "products", "min_rate": 0.025, "max_rate": 0.045}
            ),
            BankingDocument(
                id="loan_002",
                content_hr="Potrošački krediti bez namjene: Iznos od 5.000 do 150.000 kn. Rok otplate 12 do 84 mjeseca. Kamata 5.9% do 8.9% godišnje. Potrebna dokumentacija: osobna iskaznica, potvrda o prihodima zadnja 3 mjeseca, i izvadak iz HROK-a.",
                content_en="Consumer loans without purpose: Amount from 5,000 to 150,000 kn. Repayment period 12 to 84 months. Interest 5.9% to 8.9% annually. Required documentation: ID card, income proof last 3 months, and HROK credit report.",
                category="loans",
                keywords=["potrošački kredit", "consumer loan", "osobni kredit", "personal loan", "HROK"],
                metadata={"type": "products", "min_amount": 5000, "max_amount": 150000}
            ),
            
            # Online banking
            BankingDocument(
                id="digital_001",
                content_hr="Internet bankarstvo omogućava: pregled stanja i prometa, plaćanje računa, interne i eksterne transfere, upravljanje karticama, zakazivanje termina u poslovnici. Aktivacija putem tokena koji dobivate u poslovnici. Mobilna aplikacija dostupna za iOS i Android.",
                content_en="Internet banking enables: balance and transaction overview, bill payments, internal and external transfers, card management, branch appointment scheduling. Activation via token received at branch. Mobile app available for iOS and Android.",
                category="digital",
                keywords=["internet bankarstvo", "online banking", "mobilno", "mobile", "aplikacija", "app", "token"],
                metadata={"type": "service", "platforms": ["web", "ios", "android"]}
            ),
            
            # Fees and charges
            BankingDocument(
                id="fee_001",
                content_hr="Naknade za osnovne usluge: Održavanje računa 10 kn mjesečno. SEPA transfer 5 kn. Instant plaćanje 10 kn. Podizanje gotovine na našim bankomatima besplatno, na tuđim 8 kn. Izdavanje potvrde 20 kn.",
                content_en="Fees for basic services: Account maintenance 10 kn monthly. SEPA transfer 5 kn. Instant payment 10 kn. Cash withdrawal at our ATMs free, at other ATMs 8 kn. Certificate issuance 20 kn.",
                category="fees",
                keywords=["naknada", "fee", "troškovi", "charges", "cijena", "price", "SEPA"],
                metadata={"type": "pricing", "last_updated": "2024-01"}
            ),
            
            # Customer service
            BankingDocument(
                id="service_001",
                content_hr="Kontakt centar dostupan 24/7 na broj 0800-1234. Email podrška: info@banka.hr. Radno vrijeme poslovnica: ponedjeljak-petak 8:00-19:00, subota 8:00-13:00. Za hitne slučajeve (krađa kartice) pozovite +385-1-234-5678.",
                content_en="Contact center available 24/7 at 0800-1234. Email support: info@banka.hr. Branch working hours: Monday-Friday 8:00-19:00, Saturday 8:00-13:00. For emergencies (card theft) call +385-1-234-5678.",
                category="support",
                keywords=["kontakt", "contact", "podrška", "support", "radno vrijeme", "working hours", "hitno", "emergency"],
                metadata={"type": "contact", "phone": "0800-1234", "email": "info@banka.hr"}
            ),
            
            # Security
            BankingDocument(
                id="security_001",
                content_hr="Sigurnosni savjeti: Nikad ne dijelite PIN ili lozinku. Banka nikad neće tražiti vaše podatke emailom. Koristite samo službenu aplikaciju. Aktivirajte SMS obavještavanje za transakcije. Redovito mijenjajte lozinku svakih 90 dana.",
                content_en="Security tips: Never share PIN or password. Bank will never ask for your data by email. Use only official app. Activate SMS notifications for transactions. Regularly change password every 90 days.",
                category="security",
                keywords=["sigurnost", "security", "PIN", "lozinka", "password", "phishing", "prevara", "fraud"],
                metadata={"type": "guidelines", "priority": "high"}
            ),
        ]


class MultilingualEmbedder:
    """Handle embeddings for Croatian and English text."""
    
    def __init__(self):
        self.use_openai = USE_OPENAI_EMBEDDINGS
        
        if self.use_openai:
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            # Use multilingual sentence transformer
            self.model = SentenceTransformer(EMBEDDING_MODEL)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self.use_openai:
            response = self.openai_client.embeddings.create(
                input=text,
                model=OPENAI_EMBEDDING_MODEL
            )
            return np.array(response.data[0].embedding)
        else:
            return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for batch of texts."""
        if self.use_openai:
            response = self.openai_client.embeddings.create(
                input=texts,
                model=OPENAI_EMBEDDING_MODEL
            )
            return np.array([d.embedding for d in response.data])
        else:
            return self.model.encode(texts, convert_to_numpy=True)


class RAGBankingAssistant:
    """RAG-powered banking assistant with vector database."""
    
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        """Initialize RAG assistant with ChromaDB."""
        self.persist_dir = persist_dir
        self.embedder = MultilingualEmbedder()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
            print(f"✓ Loaded existing collection: {COLLECTION_NAME}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✓ Created new collection: {COLLECTION_NAME}")
            self._populate_knowledge_base()
    
    def _populate_knowledge_base(self):
        """Populate vector database with banking documents."""
        documents = BankingKnowledgeBase.get_sample_documents()
        
        for doc in documents:
            # Create combined text for embedding (both languages)
            combined_text = f"{doc.content_hr}\n{doc.content_en}"
            embedding = self.embedder.embed_text(combined_text)
            
            # Store in ChromaDB
            self.collection.add(
                ids=[doc.id],
                embeddings=[embedding.tolist()],
                documents=[combined_text],
                metadatas=[{
                    "content_hr": doc.content_hr,
                    "content_en": doc.content_en,
                    "category": doc.category,
                    "keywords": json.dumps(doc.keywords),
                    **doc.metadata
                }]
            )
        
        print(f"✓ Populated knowledge base with {len(documents)} documents")
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Croatian or English."""
        text_lower = text.lower()
        
        hr_score = sum(1 for word in CROATIAN_KEYWORDS if word in text_lower)
        en_score = sum(1 for word in ENGLISH_KEYWORDS if word in text_lower)
        
        # Check for Croatian-specific characters
        if any(char in text for char in "čćžšđ"):
            hr_score += 2
        
        return "hr" if hr_score > en_score else "en"
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """Search vector database for relevant documents."""
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        # Parse results
        parsed_results = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance  # Convert distance to similarity
            
            if similarity >= MIN_SIMILARITY_SCORE:
                metadata = results["metadatas"][0][i]
                parsed_results.append({
                    "id": results["ids"][0][i],
                    "content_hr": metadata.get("content_hr", ""),
                    "content_en": metadata.get("content_en", ""),
                    "category": metadata.get("category", ""),
                    "similarity": similarity,
                    "metadata": metadata
                })
        
        return parsed_results
    
    def generate_context(self, query: str, lang: str = None) -> Tuple[str, List[Dict]]:
        """Generate context for LLM based on query."""
        # Detect language if not provided
        if not lang:
            lang = self.detect_language(query)
        
        # Search for relevant documents
        results = self.search(query)
        
        if not results:
            return "", []
        
        # Build context based on language
        context_parts = []
        for idx, result in enumerate(results, 1):
            content = result["content_hr"] if lang == "hr" else result["content_en"]
            context_parts.append(f"{idx}. {content}")
        
        context = "\n\n".join(context_parts)
        
        # Add context header
        if lang == "hr":
            header = "Relevantne informacije iz baze znanja:"
        else:
            header = "Relevant information from knowledge base:"
        
        full_context = f"{header}\n{context}"
        
        return full_context, results
    
    def augment_prompt(self, user_message: str, lang: str = None) -> str:
        """Augment user prompt with RAG context."""
        context, sources = self.generate_context(user_message, lang)
        
        if not context:
            return user_message
        
        # Create augmented prompt
        if lang == "hr":
            augmented = f"""Kontekst iz baze znanja banke:
{context}

Korisničko pitanje: {user_message}

Molim odgovori na temelju danog konteksta. Ako informacija nije dostupna u kontekstu, reci to."""
        else:
            augmented = f"""Context from bank knowledge base:
{context}

User question: {user_message}

Please answer based on the given context. If information is not available in context, say so."""
        
        return augmented
    
    def add_document(self, doc: BankingDocument):
        """Add new document to knowledge base."""
        combined_text = f"{doc.content_hr}\n{doc.content_en}"
        embedding = self.embedder.embed_text(combined_text)
        
        self.collection.add(
            ids=[doc.id],
            embeddings=[embedding.tolist()],
            documents=[combined_text],
            metadatas=[{
                "content_hr": doc.content_hr,
                "content_en": doc.content_en,
                "category": doc.category,
                "keywords": json.dumps(doc.keywords),
                **doc.metadata
            }]
        )
    
    def update_document(self, doc_id: str, doc: BankingDocument):
        """Update existing document in knowledge base."""
        # Delete old version
        self.collection.delete(ids=[doc_id])
        
        # Add updated version
        self.add_document(doc)
    
    def export_knowledge_base(self, filepath: str):
        """Export knowledge base to file."""
        all_data = self.collection.get()
        
        export_data = {
            "documents": all_data["documents"],
            "metadatas": all_data["metadatas"],
            "ids": all_data["ids"],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge base."""
        count = self.collection.count()
        
        # Get all categories
        all_data = self.collection.get()
        categories = {}
        
        for metadata in all_data["metadatas"]:
            cat = metadata.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_documents": count,
            "categories": categories,
            "embedding_model": EMBEDDING_MODEL if not USE_OPENAI_EMBEDDINGS else OPENAI_EMBEDDING_MODEL,
            "vector_db": "ChromaDB",
            "persist_dir": self.persist_dir
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG assistant
    rag = RAGBankingAssistant()
    
    # Print statistics
    stats = rag.get_statistics()
    print("\n📊 Knowledge Base Statistics:")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Categories: {stats['categories']}")
    
    # Test queries in both languages
    test_queries = [
        ("Kako mogu otvoriti račun?", "hr"),
        ("What documents do I need for account opening?", "en"),
        ("Kolika je kamata na stambeni kredit?", "hr"),
        ("Tell me about credit cards", "en"),
        ("Koje vrste računa nudite?", "hr"),
        ("What are the fees for international transfers?", "en"),
    ]
    
    print("\n🔍 Testing RAG search:")
    for query, expected_lang in test_queries:
        print(f"\nQuery: {query}")
        detected_lang = rag.detect_language(query)
        print(f"Detected language: {detected_lang} (expected: {expected_lang})")
        
        results = rag.search(query, top_k=2)
        for result in results:
            print(f"  - [{result['category']}] Similarity: {result['similarity']:.2f}")
            preview = result['content_hr' if detected_lang == 'hr' else 'content_en'][:100]
            print(f"    {preview}...")
    
    # Test prompt augmentation
    print("\n📝 Testing prompt augmentation:")
    test_question = "Kako mogu aktivirati internet bankarstvo?"
    augmented = rag.augment_prompt(test_question, "hr")
    print(f"Original: {test_question}")
    print(f"Augmented:\n{augmented[:500]}...")

"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";
import LoginModal from "@/components/LoginModal";

export default function Home() {
  const { isAuthenticated } = useAuth();
  const router = useRouter();
  const [isLoginOpen, setIsLoginOpen] = useState(false);

  useEffect(() => {
    if (isAuthenticated) {
      router.push("/dashboard");
    }
  }, [isAuthenticated, router]);

  return (
    <div className="landing-page">
      <header className="landing-header">
        <div className="landing-container">
          <div className="logo">ENNA Next</div>
          <button className="btn-login" onClick={() => setIsLoginOpen(true)}>
            Prijava
          </button>
        </div>
      </header>

      <section className="hero-section">
        <div className="hero-container">
          <div className="hero-content">
            <h1 className="hero-title">
              Inteligentni glasovni agent za{" "}
              <span className="gradient-text">energetsku budućnost</span>
            </h1>
            <p className="hero-subtitle">
              Napredni AI razgovorni sustav s custom Croatian Whisper
              transcriberom. Automatizirajte korisničku podršku i povećajte
              zadovoljstvo klijenata.
            </p>
            <div className="hero-buttons">
              <button
                className="btn-primary-large"
                onClick={() => setIsLoginOpen(true)}
              >
                Prijavite se u portal
              </button>
              <a href="#features" className="btn-secondary-large">
                Saznajte više
              </a>
            </div>
          </div>
        </div>
      </section>

      <section id="features" className="features-section">
        <div className="landing-container">
          <h2 className="section-title">Funkcionalnosti</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🎙️</div>
              <h3>Croatian Whisper Model</h3>
              <p>
                Prilagođeni GoranS/whisper-large-v3-turbo-hr-parla model za
                preciznu transkripciju na hrvatskom jeziku
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>Real-time Processing</h3>
              <p>
                WebSocket veza s RunPod GPU serverom za brzu obradu govora u
                realnom vremenu
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🤖</div>
              <h3>GPT-4 Integracija</h3>
              <p>
                OpenAI GPT-4o-mini model za inteligentne odgovore prilagođene
                ENNA Next znanju
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h3>Napredna Analitika</h3>
              <p>
                Detaljne KPI metrike, success rate praćenje i SLA compliance
                monitoring
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🔒</div>
              <h3>Sigurnost</h3>
              <p>
                Enterprise-grade sigurnost s multi-tenant arhitekturom i
                row-level security
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">📱</div>
              <h3>VAPI Platforma</h3>
              <p>
                Twilio integracija za telefoniju s podrškom za inbound i
                outbound pozive
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="stats-section">
        <div className="landing-container">
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-number">94.3%</div>
              <div className="stat-label">AI Answer Rate</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">78.5%</div>
              <div className="stat-label">Resolution Rate</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">2.4s</div>
              <div className="stat-label">Avg Response Time</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">96.8%</div>
              <div className="stat-label">SLA Compliance</div>
            </div>
          </div>
        </div>
      </section>

      <section className="cta-section">
        <div className="landing-container">
          <div className="cta-content">
            <h2>Spremni za početak?</h2>
            <p>
              Prijavite se u portal i počnite upravljati vašim glasovnim
              agentom
            </p>
            <button
              className="btn-primary-large"
              onClick={() => setIsLoginOpen(true)}
            >
              Pristupite portalu
            </button>
          </div>
        </div>
      </section>

      <footer className="landing-footer">
        <div className="landing-container">
          <p>&copy; 2026 ENNA Next. Sva prava pridržana.</p>
        </div>
      </footer>

      <LoginModal isOpen={isLoginOpen} onClose={() => setIsLoginOpen(false)} />
    </div>
  );
}

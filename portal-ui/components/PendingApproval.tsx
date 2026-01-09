"use client";

import { useAuth } from "@/lib/auth";

export default function PendingApproval() {
  const { user, logout } = useAuth();

  return (
    <div className="pending-container">
      <div className="pending-card">
        <div className="pending-icon">⏳</div>
        <h1>Čekanje Odobrenja</h1>
        <p className="pending-message">
          Vaš račun čeka odobrenje administratora.
        </p>
        <div className="user-info">
          <p>
            <strong>Email:</strong> {user?.email}
          </p>
          <p>
            <strong>Ime:</strong> {user?.name}
          </p>
        </div>
        <p className="pending-hint">
          Administrator će uskoro pregledati vaš zahtjev. Bit ćete obaviješteni
          putem emaila kada vaš račun bude odobren.
        </p>
        <button onClick={() => logout()} className="btn-logout">
          Odjavi se
        </button>
      </div>

      <style jsx>{`
        .pending-container {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          padding: 2rem;
        }

        .pending-card {
          background: white;
          border-radius: 12px;
          padding: 3rem;
          max-width: 500px;
          text-align: center;
          box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        }

        .pending-icon {
          font-size: 4rem;
          margin-bottom: 1rem;
          animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
          0%,
          100% {
            transform: scale(1);
          }
          50% {
            transform: scale(1.1);
          }
        }

        h1 {
          margin: 0 0 1rem 0;
          color: #333;
          font-size: 2rem;
        }

        .pending-message {
          font-size: 1.125rem;
          color: #666;
          margin-bottom: 2rem;
        }

        .user-info {
          background: #f8f9fa;
          padding: 1rem;
          border-radius: 8px;
          margin-bottom: 1.5rem;
          text-align: left;
        }

        .user-info p {
          margin: 0.5rem 0;
          color: #555;
        }

        .pending-hint {
          font-size: 0.875rem;
          color: #999;
          margin-bottom: 2rem;
          line-height: 1.5;
        }

        .btn-logout {
          background: #667eea;
          color: white;
          border: none;
          padding: 0.75rem 2rem;
          border-radius: 6px;
          font-size: 1rem;
          cursor: pointer;
          transition: background 0.3s;
        }

        .btn-logout:hover {
          background: #5568d3;
        }
      `}</style>
    </div>
  );
}

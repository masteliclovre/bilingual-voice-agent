"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";
import PendingApproval from "@/components/PendingApproval";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

export default function DashboardPage() {
  const { user, isAuthenticated, logout, currentTenant } = useAuth();
  const router = useRouter();
  const [activeCalls, setActiveCalls] = useState([]);
  const [completedCalls, setCompletedCalls] = useState([]);
  const [forwardedCalls, setForwardedCalls] = useState([]);

  // Check if user is pending approval
  const isPending = (user as any)?.approval_status === 'pending';

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/");
      return;
    }

    // If user is pending, don't fetch calls
    if (isPending) {
      return;
    }

    const fetchCalls = async () => {
      try {
        // TEMP: Use default tenant ID until Google Auth is fully working
        const defaultTenantId = '00000000-0000-0000-0000-000000000001';

        const headers = {
          'X-Tenant-ID': defaultTenantId
        };

        const [active, completed, forwarded] = await Promise.all([
          fetch(`${API_URL}/api/calls/active`, { headers }).then(r => r.json()),
          fetch(`${API_URL}/api/calls/completed`, { headers }).then(r => r.json()),
          fetch(`${API_URL}/api/calls/forwarded`, { headers }).then(r => r.json())
        ]);
        setActiveCalls(active);
        setCompletedCalls(completed);
        setForwardedCalls(forwarded);
      } catch (error) {
        console.error('Error fetching calls:', error);
      }
    };

    fetchCalls();
    const interval = setInterval(fetchCalls, 3000);
    return () => clearInterval(interval);
  }, [isAuthenticated, router, isPending]);

  // Show pending approval page if user is not approved yet
  if (isPending) {
    return <PendingApproval />;
  }

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h2>ENNA Next Portal</h2>
        <nav>
          <a href="/dashboard">Dashboard</a>
          <a href="/dashboard/calls">Pozivi</a>
          <a href="/dashboard/outcomes">Ishodi</a>
          {currentTenant?.role === 'admin' && (
            <a href="/admin" className="admin-link">Admin Panel</a>
          )}
        </nav>
        <div className="sidebar-footer">
          <div className="user-info">
            <div>{user?.name}</div>
            <div className="user-email">{user?.email}</div>
            {currentTenant && (
              <div className="user-role">{currentTenant.role}</div>
            )}
          </div>
          <button onClick={logout} className="btn-logout">
            Odjava
          </button>
        </div>
      </aside>
      <main className="content">
        <h1>Dashboard</h1>
        <p>Pregled stanja poziva u realnom vremenu</p>

        {/* SEKCIJA 1: PROSLJEĐENI POZIVI */}
        <div className="priority-section">
          <div className="section-header">
            <h2>🔴 Prosljeđeni Pozivi</h2>
            <span className="priority-badge">{forwardedCalls.length} poziva</span>
          </div>
          <p className="section-description">
            AI agent prosljeđuje poziv na admin korisnika
          </p>

          {forwardedCalls.length === 0 ? (
            <p style={{color: '#6b7280', textAlign: 'center', padding: '40px'}}>
              Nema prosljeđenih poziva
            </p>
          ) : (
          <div className="escalated-calls">
            {forwardedCalls.map((call: any) => (
              <div key={call.id} className="escalated-call-card">
                <div className="call-header">
                  <div className="call-time">{new Date(call.startedAt).toLocaleString('hr-HR')}</div>
                </div>
                <div className="call-info">
                  <div className="phone-number">{call.phoneNumber}</div>
                  {call.duration && <div className="call-duration">Trajanje: {Math.floor(call.duration / 60)}:{(call.duration % 60).toString().padStart(2, '0')}</div>}
                </div>
                {call.forwardReason && (
                  <div className="ai-summary">
                    <strong>Razlog prosljeđivanja:</strong>
                    <p>{call.forwardReason}</p>
                  </div>
                )}
                <div className="call-actions">
                  <button className="btn-primary-small">Nazovi korisnika</button>
                  <button className="btn-secondary-small">Pročitaj transkript</button>
                </div>
              </div>
            ))}
          </div>
          )}
        </div>

        {/* SEKCIJA 2: AKTIVNI POZIVI */}
        <div className="active-calls-section">
          <div className="section-header">
            <h2>🟢 Aktivni Pozivi</h2>
            <span className="active-count">{activeCalls.length} u tijeku</span>
          </div>

          {activeCalls.length === 0 ? (
            <p style={{color: '#6b7280', textAlign: 'center', padding: '40px'}}>
              Nema aktivnih poziva
            </p>
          ) : (
          <div className="active-calls-grid">
            {activeCalls.map((call: any) => (
              <div key={call.id} className="active-call-card">
                <div className="pulse-indicator"></div>
                <div className="active-call-info">
                  <div className="phone-number">{call.phoneNumber}</div>
                  <div className="call-timer">
                    Započeo: {new Date(call.startedAt).toLocaleTimeString('hr-HR')}
                  </div>
                </div>
                <div className="live-status">Poziv u tijeku...</div>
              </div>
            ))}
          </div>
          )}
        </div>

        {/* SEKCIJA 3: PROŠLI POZIVI */}
        <div className="past-calls-section">
          <div className="section-header">
            <h2>📋 Riješeni Pozivi (Danas)</h2>
          </div>

          {completedCalls.length === 0 ? (
            <p style={{color: '#6b7280', textAlign: 'center', padding: '40px'}}>
              Nema riješenih poziva
            </p>
          ) : (
          <div className="past-calls-list">
            {completedCalls.map((call: any) => (
              <div key={call.id} className="past-call-item">
                <div className="call-basic-info">
                  <div className="call-time">{new Date(call.startedAt).toLocaleTimeString('hr-HR', { hour: '2-digit', minute: '2-digit' })}</div>
                  <div className="phone-number">{call.phoneNumber}</div>
                  {call.duration && <div className="call-duration">{Math.floor(call.duration / 60)}:{(call.duration % 60).toString().padStart(2, '0')}</div>}
                  <span className="status-badge status-success">Riješeno</span>
                </div>
                {call.summary && (
                  <div className="call-summary">
                    <strong>Sažetak:</strong> {call.summary}
                  </div>
                )}
                <button className="btn-view-details">Detalji</button>
              </div>
            ))}
          </div>
          )}

          <div style={{ marginTop: "16px", textAlign: "right" }}>
            <a href="/calls" className="btn-view-all">
              Prikaži sve pozive →
            </a>
          </div>
        </div>
      </main>
    </div>
  );
}

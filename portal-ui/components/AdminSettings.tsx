"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/lib/auth";

interface User {
  id: string;
  email: string;
  full_name: string;
  profile_picture?: string;
  role?: string;
  approval_status?: string;
  created_at: string;
}

export default function AdminSettings() {
  const { user, currentTenant, isLoading: authLoading } = useAuth();
  const [pendingUsers, setPendingUsers] = useState<User[]>([]);
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const isAdmin = currentTenant?.role === "admin";

  useEffect(() => {
    // Wait for auth to load before checking
    if (authLoading) {
      return;
    }

    // Now we can safely check if user is admin
    if (!isAdmin) {
      setError("Admin access required");
      setLoading(false);
      return;
    }

    fetchUsers();
  }, [isAdmin, currentTenant, authLoading]);

  const fetchUsers = async () => {
    try {
      setLoading(true);
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

      // Extract user_id safely
      const userId = (user as any)?.user_id || "";
      const tenantId = currentTenant?.tenant_id || "";

      console.log("Admin Panel - Fetching with:", { userId, tenantId, role: currentTenant?.role });

      if (!userId || !tenantId) {
        throw new Error("Missing user or tenant ID");
      }

      const headers = {
        "X-Tenant-ID": tenantId,
        "X-User-ID": userId,
      };

      // Fetch pending users
      const pendingRes = await fetch(`${API_URL}/api/admin/users/pending`, {
        headers,
      });

      if (!pendingRes.ok) {
        throw new Error("Failed to fetch pending users");
      }

      const pending = await pendingRes.json();
      console.log("Pending users from API:", pending);
      setPendingUsers(pending);

      // Fetch all users
      const allRes = await fetch(`${API_URL}/api/admin/users`, {
        headers,
      });

      if (!allRes.ok) {
        throw new Error("Failed to fetch users");
      }

      const all = await allRes.json();
      console.log("All users from API:", all);
      setAllUsers(all);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async (userId: string, role: string) => {
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

      const headers = {
        "X-Tenant-ID": currentTenant?.tenant_id || "",
        "X-User-ID": (user as any)?.user_id || "",
        "Content-Type": "application/json",
      };

      const res = await fetch(`${API_URL}/api/admin/users/${userId}/approve`, {
        method: "POST",
        headers,
        body: JSON.stringify({ role }),
      });

      if (!res.ok) {
        throw new Error("Failed to approve user");
      }

      // Refresh users
      await fetchUsers();
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    }
  };

  const handleReject = async (userId: string) => {
    if (!confirm("Are you sure you want to reject this user?")) return;

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

      const headers = {
        "X-Tenant-ID": currentTenant?.tenant_id || "",
        "X-User-ID": (user as any)?.user_id || "",
      };

      const res = await fetch(`${API_URL}/api/admin/users/${userId}/reject`, {
        method: "POST",
        headers,
      });

      if (!res.ok) {
        throw new Error("Failed to reject user");
      }

      // Refresh users
      await fetchUsers();
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    }
  };

  const handleChangeRole = async (userId: string, newRole: string) => {
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

      const headers = {
        "X-Tenant-ID": currentTenant?.tenant_id || "",
        "X-User-ID": (user as any)?.user_id || "",
        "Content-Type": "application/json",
      };

      const res = await fetch(`${API_URL}/api/admin/users/${userId}/role`, {
        method: "PUT",
        headers,
        body: JSON.stringify({ role: newRole }),
      });

      if (!res.ok) {
        throw new Error("Failed to update role");
      }

      // Refresh users
      await fetchUsers();
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    }
  };

  if (authLoading || loading) {
    return (
      <div className="admin-container">
        <h1>Admin Panel</h1>
        <p>Loading...</p>
      </div>
    );
  }

  if (!isAdmin) {
    return (
      <div className="admin-container">
        <h1>Admin Panel</h1>
        <div className="error-message">
          Access Denied: Admin privileges required
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="admin-container">
        <h1>Admin Panel</h1>
        <div className="error-message">{error}</div>
      </div>
    );
  }

  return (
    <div className="admin-container">
      <div className="admin-header">
        <h1>Admin Panel</h1>
        <p>Upravljanje korisnicima za {currentTenant?.tenant_name}</p>
      </div>

      {/* Pending Users Section */}
      {pendingUsers.length > 0 && (
        <div className="admin-section">
          <h2>Pending Approvals ({pendingUsers.length})</h2>
          <div className="users-table">
            <table>
              <thead>
                <tr>
                  <th>Email</th>
                  <th>Full Name</th>
                  <th>Requested</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {pendingUsers.map((u) => (
                  <tr key={u.id}>
                    <td>
                      <div className="user-info">
                        {u.profile_picture && (
                          <img
                            src={u.profile_picture}
                            alt={u.full_name}
                            className="user-avatar-small"
                          />
                        )}
                        {u.email}
                      </div>
                    </td>
                    <td>{u.full_name}</td>
                    <td>{new Date(u.created_at).toLocaleString()}</td>
                    <td>
                      <div className="action-buttons">
                        <select
                          className="role-select"
                          id={`role-${u.id}`}
                          defaultValue="viewer"
                        >
                          <option value="viewer">Viewer</option>
                          <option value="manager">Manager</option>
                          <option value="admin">Admin</option>
                        </select>
                        <button
                          className="btn-approve"
                          onClick={() => {
                            const select = document.getElementById(
                              `role-${u.id}`
                            ) as HTMLSelectElement;
                            handleApprove(u.id, select.value);
                          }}
                        >
                          Approve
                        </button>
                        <button
                          className="btn-reject"
                          onClick={() => handleReject(u.id)}
                        >
                          Reject
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* All Users Section */}
      <div className="admin-section">
        <h2>All Users ({allUsers.length})</h2>
        <div className="users-table">
          <table>
            <thead>
              <tr>
                <th>Email</th>
                <th>Full Name</th>
                <th>Role</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {allUsers.map((u) => (
                <tr key={u.id}>
                  <td>
                    <div className="user-info">
                      {u.profile_picture && (
                        <img
                          src={u.profile_picture}
                          alt={u.full_name}
                          className="user-avatar-small"
                        />
                      )}
                      {u.email}
                    </div>
                  </td>
                  <td>{u.full_name}</td>
                  <td>
                    <select
                      className="role-select"
                      value={u.role}
                      onChange={(e) => handleChangeRole(u.id, e.target.value)}
                      disabled={u.id === (user as any)?.user_id}
                    >
                      <option value="viewer">Viewer</option>
                      <option value="manager">Manager</option>
                      <option value="admin">Admin</option>
                    </select>
                  </td>
                  <td>
                    <span className={`status-badge status-${u.approval_status}`}>
                      {u.approval_status}
                    </span>
                  </td>
                  <td>
                    {u.id === (user as any)?.user_id ? (
                      <span className="text-muted">You</span>
                    ) : (
                      <button
                        className="btn-remove"
                        onClick={() => handleReject(u.id)}
                      >
                        Remove
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <style jsx>{`
        .admin-container {
          padding: 2rem;
          max-width: 1200px;
          margin: 0 auto;
        }

        .admin-header {
          margin-bottom: 2rem;
        }

        .admin-header h1 {
          margin: 0 0 0.5rem 0;
        }

        .admin-header p {
          color: #666;
          margin: 0;
        }

        .admin-section {
          background: white;
          border-radius: 8px;
          padding: 1.5rem;
          margin-bottom: 2rem;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .admin-section h2 {
          margin: 0 0 1rem 0;
          font-size: 1.25rem;
        }

        .users-table {
          overflow-x: auto;
        }

        table {
          width: 100%;
          border-collapse: collapse;
        }

        th,
        td {
          text-align: left;
          padding: 0.75rem;
          border-bottom: 1px solid #eee;
        }

        th {
          font-weight: 600;
          background: #f8f9fa;
          color: #333;
        }

        .user-info {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .user-avatar-small {
          width: 32px;
          height: 32px;
          border-radius: 50%;
        }

        .action-buttons {
          display: flex;
          gap: 0.5rem;
          align-items: center;
        }

        .role-select {
          padding: 0.375rem 0.5rem;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 0.875rem;
        }

        .btn-approve,
        .btn-reject,
        .btn-remove {
          padding: 0.375rem 0.75rem;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 0.875rem;
          transition: opacity 0.2s;
        }

        .btn-approve {
          background: #28a745;
          color: white;
        }

        .btn-approve:hover {
          opacity: 0.8;
        }

        .btn-reject,
        .btn-remove {
          background: #dc3545;
          color: white;
        }

        .btn-reject:hover,
        .btn-remove:hover {
          opacity: 0.8;
        }

        .status-badge {
          display: inline-block;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.75rem;
          font-weight: 600;
          text-transform: uppercase;
        }

        .status-approved {
          background: #d4edda;
          color: #155724;
        }

        .status-pending {
          background: #fff3cd;
          color: #856404;
        }

        .status-rejected {
          background: #f8d7da;
          color: #721c24;
        }

        .text-muted {
          color: #999;
          font-style: italic;
        }

        .error-message {
          background: #f8d7da;
          color: #721c24;
          padding: 1rem;
          border-radius: 4px;
          border: 1px solid #f5c6cb;
        }
      `}</style>
    </div>
  );
}

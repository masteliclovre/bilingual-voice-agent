from flask import Flask, request, jsonify, g
from flask_cors import CORS
import os
from datetime import datetime
import psycopg
from psycopg import sql
from dotenv import load_dotenv
import logging
import hashlib
import json

load_dotenv()

app = Flask(__name__)

# CORS configuration for development and production
allowed_origins = [
    "http://localhost:3000",
    "https://*.railway.app",
    "https://*.up.railway.app"
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}}, supports_credentials=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv('DATABASE_URL')

def get_db():
    return psycopg.connect(DATABASE_URL)

def compute_payload_hash(payload):
    """Compute hash of payload for deduplication"""
    payload_str = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload_str.encode()).hexdigest()

def get_tenant_id_from_vapi_data(data):
    """Extract tenant_id based on VAPI org_id or phone_number_id"""
    # Try to get from call data
    call = data.get('message', {}).get('call', {}) or data.get('call', {})
    org_id = call.get('orgId')
    phone_number_id = call.get('phoneNumberId')
    assistant_id = call.get('assistantId')

    with get_db() as conn:
        with conn.cursor() as cur:
            # Try to match by vapi_org_id first
            if org_id:
                cur.execute("""
                    SELECT tenant_id FROM tenant_vapi_map
                    WHERE vapi_org_id = %s
                    LIMIT 1
                """, (org_id,))
                result = cur.fetchone()
                if result:
                    return str(result[0])

            # Try to match by phone_number_id
            if phone_number_id:
                cur.execute("""
                    SELECT tenant_id FROM tenant_vapi_map
                    WHERE vapi_phone_number_id = %s
                    LIMIT 1
                """, (phone_number_id,))
                result = cur.fetchone()
                if result:
                    return str(result[0])

            # Try to match by assistant_id
            if assistant_id:
                cur.execute("""
                    SELECT tenant_id FROM tenant_vapi_map
                    WHERE vapi_assistant_id = %s
                    LIMIT 1
                """, (assistant_id,))
                result = cur.fetchone()
                if result:
                    return str(result[0])

    # Default tenant if no match (for backward compatibility)
    return '00000000-0000-0000-0000-000000000001'

def set_tenant_context(conn, tenant_id):
    """Set RLS tenant context for this connection"""
    with conn.cursor() as cur:
        # Use SQL composition for SET command (doesn't support parameterized queries)
        cur.execute(sql.SQL("SET app.tenant_id = {}").format(sql.Literal(tenant_id)))

@app.route('/api/webhooks/vapi', methods=['POST'])
def vapi_webhook():
    data = request.json

    # Determine tenant
    tenant_id = get_tenant_id_from_vapi_data(data)

    # Extract event type
    message = data.get('message', {})
    event_type = data.get('type') or message.get('type')

    logger.info(f"========== WEBHOOK RECEIVED ==========")
    logger.info(f"Tenant ID: {tenant_id}")
    logger.info(f"Event Type: {event_type}")
    logger.info(f"Full Data: {data}")
    logger.info(f"=======================================")

    with get_db() as conn:
        set_tenant_context(conn, tenant_id)

        with conn.cursor() as cur:
            # Store raw event
            payload_hash = compute_payload_hash(data)
            call_id = message.get('call', {}).get('id') or data.get('call', {}).get('id')

            try:
                cur.execute("""
                    INSERT INTO call_events_raw (tenant_id, vapi_call_id, event_type, payload, payload_hash)
                    VALUES (%s, %s, %s, %s, %s)
                """, (tenant_id, call_id, event_type, json.dumps(data), payload_hash))
            except psycopg.errors.UniqueViolation:
                logger.info(f"Duplicate event ignored: {event_type} for call {call_id}")
                conn.rollback()
                return jsonify({'success': True, 'message': 'Duplicate event'}), 200

            # Process event based on type
            if 'message' in data:
                return handle_server_message(data, tenant_id, conn)
            elif event_type == 'call.started':
                return handle_call_started(data, tenant_id, conn)
            elif event_type == 'call.ended':
                return handle_call_ended(data, tenant_id, conn)

            conn.commit()

    return jsonify({'success': True}), 200

def handle_server_message(data, tenant_id, conn):
    """Handle VAPI server messages"""
    message = data.get('message', {})
    msg_type = message.get('type')
    call = message.get('call', {})
    call_id = call.get('id')

    if not call_id:
        return jsonify({'success': True, 'message': 'No call ID'}), 200

    with conn.cursor() as cur:
        # Ensure call exists
        call_data = message.get('call', {})
        phone_number = message.get('customer', {}).get('number') or call_data.get('customer', {}).get('number')
        started_at = call_data.get('startedAt') or call_data.get('createdAt')
        org_id = call.get('orgId')
        phone_number_id = call.get('phoneNumberId')
        assistant_id = call.get('assistantId')

        if started_at:
            cur.execute("""
                INSERT INTO calls (
                    tenant_id, vapi_call_id, vapi_org_id, vapi_phone_number_id,
                    vapi_assistant_id, caller_e164, started_at, outcome
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
                ON CONFLICT (vapi_call_id) DO NOTHING
            """, (tenant_id, call_id, org_id, phone_number_id, assistant_id, phone_number, started_at))
            conn.commit()

        if msg_type == 'transcript':
            # Store transcript in call_variables or transcript_text
            transcript = message.get('transcript', '')
            role = message.get('role', 'unknown')

            # Append to transcript_text field
            cur.execute("""
                UPDATE calls
                SET transcript_text = COALESCE(transcript_text, '') || %s || E'\n'
                WHERE vapi_call_id = %s AND tenant_id = %s
            """, (f"{role}: {transcript}", call_id, tenant_id))

        elif msg_type == 'end-of-call-report':
            ended_at = call_data.get('endedAt')
            duration = call_data.get('duration')
            summary = message.get('summary', '')

            cur.execute("""
                UPDATE calls
                SET ended_at = %s, duration_sec = %s, outcome = 'completed', summary = %s, updated_at = NOW()
                WHERE vapi_call_id = %s AND tenant_id = %s
            """, (ended_at, duration, summary, call_id, tenant_id))

        conn.commit()

    return jsonify({'success': True}), 200

def handle_call_started(data, tenant_id, conn):
    """Handle call.started event"""
    call = data.get('call', {})
    call_id = call.get('id')
    phone = call.get('phoneNumber')
    started = call.get('startedAt')
    org_id = call.get('orgId')
    phone_number_id = call.get('phoneNumberId')
    assistant_id = call.get('assistantId')

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO calls (
                tenant_id, vapi_call_id, vapi_org_id, vapi_phone_number_id,
                vapi_assistant_id, caller_e164, started_at, outcome
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
            ON CONFLICT (vapi_call_id) DO NOTHING
        """, (tenant_id, call_id, org_id, phone_number_id, assistant_id, phone, started))
        conn.commit()

    return jsonify({'success': True}), 200

def handle_call_ended(data, tenant_id, conn):
    """Handle call.ended event"""
    call = data.get('call', {})
    call_id = call.get('id')
    ended = call.get('endedAt')
    duration = call.get('duration')

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE calls
            SET ended_at = %s, duration_sec = %s, outcome = 'completed', updated_at = NOW()
            WHERE vapi_call_id = %s AND tenant_id = %s
        """, (ended, duration, call_id, tenant_id))
        conn.commit()

    return jsonify({'success': True}), 200

# ===== AUTH ENDPOINTS =====

@app.route('/api/auth/sync-user', methods=['POST'])
def sync_user():
    """Sync Google OAuth user to database"""
    data = request.json
    email = data.get('email')
    full_name = data.get('full_name')
    google_id = data.get('google_id')
    profile_picture = data.get('profile_picture')

    logger.info(f"========== SYNC USER REQUEST ==========")
    logger.info(f"Email: {email}")
    logger.info(f"Full Name: {full_name}")
    logger.info(f"=======================================")

    if not email:
        return jsonify({'error': 'Email required'}), 400

    with get_db() as conn:
        with conn.cursor() as cur:
            # Insert or update user (auto-approve as admin for development)
            cur.execute("""
                INSERT INTO users (email, full_name, google_id, profile_picture, approval_status, approved_at)
                VALUES (%s, %s, %s, %s, 'approved', NOW())
                ON CONFLICT (email) DO UPDATE
                SET full_name = EXCLUDED.full_name,
                    google_id = EXCLUDED.google_id,
                    profile_picture = EXCLUDED.profile_picture,
                    approval_status = 'approved',
                    approved_at = COALESCE(users.approved_at, NOW())
                RETURNING id
            """, (email, full_name, google_id, profile_picture))

            result = cur.fetchone()
            user_id = result[0]

            # Create tenant for user (using email domain or user_id)
            tenant_name = f"Tenant for {full_name or email}"
            cur.execute("""
                INSERT INTO tenants (id, name, status)
                VALUES (%s, %s, 'active')
                ON CONFLICT (id) DO NOTHING
            """, (str(user_id), tenant_name))

            # Auto-assign as admin to their tenant
            cur.execute("""
                INSERT INTO user_tenant_roles (user_id, tenant_id, role)
                VALUES (%s, %s, 'admin')
                ON CONFLICT (user_id, tenant_id) DO UPDATE
                SET role = 'admin'
            """, (user_id, str(user_id)))

            # Get user's tenants
            cur.execute("""
                SELECT t.id, t.name, utr.role
                FROM user_tenant_roles utr
                JOIN tenants t ON utr.tenant_id = t.id
                WHERE utr.user_id = %s AND t.status = 'active'
                ORDER BY utr.created_at ASC
            """, (user_id,))

            tenants = [
                {
                    'tenant_id': str(row[0]),
                    'tenant_name': row[1],
                    'role': row[2]
                }
                for row in cur.fetchall()
            ]

            conn.commit()

    response_data = {
        'user_id': str(user_id),
        'approval_status': approval_status,
        'tenants': tenants
    }

    logger.info(f"========== SYNC USER RESPONSE ==========")
    logger.info(f"Response: {response_data}")
    logger.info(f"========================================")

    return jsonify(response_data), 200

# ===== ADMIN ENDPOINTS =====

def check_admin_role(tenant_id, user_id):
    """Check if user has admin role for the tenant"""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT role FROM user_tenant_roles
                WHERE user_id = %s AND tenant_id = %s
            """, (user_id, tenant_id))
            result = cur.fetchone()
            return result and result[0] == 'admin'

@app.route('/api/admin/users/pending', methods=['GET'])
def get_pending_users():
    """Get all users pending approval (admin only)"""
    tenant_id = request.headers.get('X-Tenant-ID')
    user_id = request.headers.get('X-User-ID')

    if not tenant_id or not user_id:
        return jsonify({'error': 'Tenant ID and User ID required'}), 400

    if not check_admin_role(tenant_id, user_id):
        return jsonify({'error': 'Admin access required'}), 403

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, email, full_name, profile_picture, created_at
                FROM users
                WHERE approval_status = 'pending'
                ORDER BY created_at DESC
            """)

            rows = cur.fetchall()
            users = [
                {
                    'id': str(row[0]),
                    'email': row[1],
                    'full_name': row[2],
                    'profile_picture': row[3],
                    'created_at': row[4].isoformat() if row[4] else None
                }
                for row in rows
            ]

    return jsonify(users), 200

@app.route('/api/admin/users/<user_id>/approve', methods=['POST'])
def approve_user(user_id):
    """Approve a pending user (admin only)"""
    tenant_id = request.headers.get('X-Tenant-ID')
    admin_id = request.headers.get('X-User-ID')

    if not tenant_id or not admin_id:
        return jsonify({'error': 'Tenant ID and User ID required'}), 400

    if not check_admin_role(tenant_id, admin_id):
        return jsonify({'error': 'Admin access required'}), 403

    data = request.json
    role = data.get('role', 'viewer')  # Default role: viewer

    with get_db() as conn:
        with conn.cursor() as cur:
            # Approve user
            cur.execute("""
                UPDATE users
                SET approval_status = 'approved',
                    approved_at = NOW(),
                    approved_by = %s
                WHERE id = %s
            """, (admin_id, user_id))

            # Assign to tenant
            cur.execute("""
                INSERT INTO user_tenant_roles (user_id, tenant_id, role)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, tenant_id) DO UPDATE
                SET role = EXCLUDED.role
            """, (user_id, tenant_id, role))

            conn.commit()

    return jsonify({'success': True, 'message': 'User approved'}), 200

@app.route('/api/admin/users/<user_id>/reject', methods=['POST'])
def reject_user(user_id):
    """Reject a pending user (admin only)"""
    tenant_id = request.headers.get('X-Tenant-ID')
    admin_id = request.headers.get('X-User-ID')

    if not tenant_id or not admin_id:
        return jsonify({'error': 'Tenant ID and User ID required'}), 400

    if not check_admin_role(tenant_id, admin_id):
        return jsonify({'error': 'Admin access required'}), 403

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users
                SET approval_status = 'rejected'
                WHERE id = %s
            """, (user_id,))
            conn.commit()

    return jsonify({'success': True, 'message': 'User rejected'}), 200

@app.route('/api/admin/users', methods=['GET'])
def get_all_users():
    """Get all users in tenant (admin only)"""
    tenant_id = request.headers.get('X-Tenant-ID')
    user_id = request.headers.get('X-User-ID')

    if not tenant_id or not user_id:
        return jsonify({'error': 'Tenant ID and User ID required'}), 400

    if not check_admin_role(tenant_id, user_id):
        return jsonify({'error': 'Admin access required'}), 403

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT u.id, u.email, u.full_name, u.profile_picture,
                       utr.role, u.approval_status, u.created_at
                FROM user_tenant_roles utr
                JOIN users u ON utr.user_id = u.id
                WHERE utr.tenant_id = %s
                ORDER BY u.created_at DESC
            """, (tenant_id,))

            rows = cur.fetchall()
            logger.info(f"Fetched {len(rows)} users for tenant {tenant_id}")
            users = [
                {
                    'id': str(row[0]),
                    'email': row[1],
                    'full_name': row[2],
                    'profile_picture': row[3],
                    'role': row[4],
                    'approval_status': row[5],
                    'created_at': row[6].isoformat() if row[6] else None
                }
                for row in rows
            ]
            logger.info(f"Returning users: {users}")

    return jsonify(users), 200

@app.route('/api/admin/users/<user_id>/role', methods=['PUT'])
def update_user_role(user_id):
    """Update user role (admin only)"""
    tenant_id = request.headers.get('X-Tenant-ID')
    admin_id = request.headers.get('X-User-ID')

    if not tenant_id or not admin_id:
        return jsonify({'error': 'Tenant ID and User ID required'}), 400

    if not check_admin_role(tenant_id, admin_id):
        return jsonify({'error': 'Admin access required'}), 403

    data = request.json
    new_role = data.get('role')

    if new_role not in ['admin', 'manager', 'viewer']:
        return jsonify({'error': 'Invalid role'}), 400

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE user_tenant_roles
                SET role = %s
                WHERE user_id = %s AND tenant_id = %s
            """, (new_role, user_id, tenant_id))
            conn.commit()

    return jsonify({'success': True, 'message': 'Role updated'}), 200

# ===== API ENDPOINTS (with tenant filtering) =====

@app.route('/api/calls', methods=['GET'])
def get_all_calls():
    tenant_id = request.headers.get('X-Tenant-ID')
    if not tenant_id:
        return jsonify({'error': 'Tenant ID required'}), 400

    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('pageSize', 20))
    offset = (page - 1) * page_size

    with get_db() as conn:
        set_tenant_context(conn, tenant_id)

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM calls WHERE tenant_id = %s", (tenant_id,))
            total = cur.fetchone()[0]

            cur.execute("""
                SELECT vapi_call_id, caller_e164, started_at, ended_at, duration_sec,
                       outcome, reason, summary, escalated
                FROM calls
                WHERE tenant_id = %s
                ORDER BY started_at DESC
                LIMIT %s OFFSET %s
            """, (tenant_id, page_size, offset))

            rows = cur.fetchall()
            items = []
            for row in rows:
                duration_sec = row[4]
                if not duration_sec and row[2] and row[3]:
                    duration_sec = int((row[3] - row[2]).total_seconds())

                items.append({
                    'vapiCallId': row[0],
                    'callerMasked': row[1],
                    'startedAt': row[2].isoformat() if row[2] else None,
                    'durationSec': duration_sec or 0,
                    'outcome': row[5] or 'active',
                    'reason': row[6] or 'General inquiry',
                    'summary': row[7] or ''
                })

    return jsonify({
        'page': page,
        'pageSize': page_size,
        'total': total,
        'items': items
    }), 200

@app.route('/api/calls/active', methods=['GET'])
def get_active_calls():
    tenant_id = request.headers.get('X-Tenant-ID')
    if not tenant_id:
        return jsonify({'error': 'Tenant ID required'}), 400

    with get_db() as conn:
        set_tenant_context(conn, tenant_id)

        with conn.cursor() as cur:
            cur.execute("""
                SELECT vapi_call_id, caller_e164, started_at
                FROM calls
                WHERE tenant_id = %s AND outcome = 'active'
                ORDER BY started_at DESC
            """, (tenant_id,))

            rows = cur.fetchall()
            result = [{'id': row[0], 'phoneNumber': row[1], 'startedAt': row[2].isoformat() if row[2] else None} for row in rows]

    return jsonify(result), 200

@app.route('/api/calls/completed', methods=['GET'])
def get_completed_calls():
    tenant_id = request.headers.get('X-Tenant-ID')
    if not tenant_id:
        return jsonify({'error': 'Tenant ID required'}), 400

    with get_db() as conn:
        set_tenant_context(conn, tenant_id)

        with conn.cursor() as cur:
            cur.execute("""
                SELECT vapi_call_id, caller_e164, started_at, duration_sec, summary
                FROM calls
                WHERE tenant_id = %s AND outcome = 'completed' AND escalated = FALSE
                ORDER BY started_at DESC
            """, (tenant_id,))

            rows = cur.fetchall()
            result = [
                {
                    'id': row[0],
                    'phoneNumber': row[1],
                    'startedAt': row[2].isoformat() if row[2] else None,
                    'duration': row[3],
                    'summary': row[4]
                }
                for row in rows
            ]

    return jsonify(result), 200

@app.route('/api/calls/forwarded', methods=['GET'])
def get_forwarded_calls():
    tenant_id = request.headers.get('X-Tenant-ID')
    if not tenant_id:
        return jsonify({'error': 'Tenant ID required'}), 400

    with get_db() as conn:
        set_tenant_context(conn, tenant_id)

        with conn.cursor() as cur:
            cur.execute("""
                SELECT vapi_call_id, caller_e164, started_at, duration_sec, reason
                FROM calls
                WHERE tenant_id = %s AND escalated = TRUE
                ORDER BY started_at DESC
            """, (tenant_id,))

            rows = cur.fetchall()
            result = [
                {
                    'id': row[0],
                    'phoneNumber': row[1],
                    'startedAt': row[2].isoformat() if row[2] else None,
                    'duration': row[3],
                    'forwardReason': row[4]
                }
                for row in rows
            ]

    return jsonify(result), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import psycopg
from dotenv import load_dotenv
import logging

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv('DATABASE_URL')

def get_db():
    return psycopg.connect(DATABASE_URL)

def handle_server_message(data):
    """Handle Vapi server messages (transcript, status-update, etc.)"""
    message = data.get('message', {})
    msg_type = message.get('type')
    call = message.get('call', {})
    call_id = call.get('id')

    logger.info(f"Handling server message: {msg_type} for call {call_id}")

    if not call_id:
        return jsonify({'success': True, 'message': 'No call ID in message'}), 200

    with get_db() as conn:
        with conn.cursor() as cur:
            # ALWAYS ensure call exists in DB before processing any event
            call_data = message.get('call', {})
            phone_number = message.get('customer', {}).get('number') or call_data.get('customer', {}).get('number', 'Unknown')
            started_at = call_data.get('startedAt') or call_data.get('createdAt')

            if started_at:  # Only create if we have timestamp
                cur.execute("""
                    INSERT INTO calls (vapi_call_id, phone_number, started_at, status)
                    VALUES (%s, %s, %s, 'active')
                    ON CONFLICT (vapi_call_id) DO NOTHING
                """, (call_id, phone_number, started_at))
                conn.commit()  # Commit immediately so it's visible
            if msg_type == 'transcript':
                # Real-time transcript update
                transcript = message.get('transcript', {})
                role = message.get('role')  # 'user' or 'assistant'
                timestamp = message.get('timestamp')

                cur.execute("""
                    INSERT INTO transcripts (call_id, speaker, text, timestamp)
                    SELECT id, %s, %s, %s FROM calls WHERE vapi_call_id = %s
                """, (role, transcript, timestamp, call_id))

            elif msg_type == 'conversation-update':
                # Conversation started - create call record if doesn't exist
                call_data = message.get('call', {})
                phone_number = message.get('customer', {}).get('number', 'Unknown')
                started_at = call_data.get('startedAt') or call_data.get('createdAt')

                logger.info(f"Conversation update: creating/updating call {call_id}")

                cur.execute("""
                    INSERT INTO calls (vapi_call_id, phone_number, started_at, status)
                    VALUES (%s, %s, %s, 'active')
                    ON CONFLICT (vapi_call_id) DO NOTHING
                """, (call_id, phone_number, started_at))

            elif msg_type == 'status-update':
                # Status changes during call
                status = message.get('status')
                logger.info(f"Call {call_id} status: {status}")

            elif msg_type == 'end-of-call-report':
                # Final call report
                call_data = message.get('call', {})
                ended_at = call_data.get('endedAt')
                duration = call_data.get('duration')
                summary = message.get('summary', '')

                logger.info(f"End of call: {call_id}, ended_at={ended_at}, duration={duration}")

                cur.execute("""
                    UPDATE calls
                    SET ended_at = %s, duration_seconds = %s, status = 'completed'
                    WHERE vapi_call_id = %s
                """, (ended_at, duration, call_id))

                if summary:
                    cur.execute("""
                        INSERT INTO call_metadata (call_id, ai_summary)
                        SELECT id, %s FROM calls WHERE vapi_call_id = %s
                        ON CONFLICT (call_id) DO UPDATE SET ai_summary = EXCLUDED.ai_summary
                    """, (summary, call_id))

            conn.commit()

    return jsonify({'success': True}), 200

@app.route('/api/webhooks/vapi', methods=['POST'])
def vapi_webhook():
    data = request.json

    # Vapi može slati različite formate poruka
    # 1. Webhook eventi: { "type": "call.started", "call": {...} }
    # 2. Server messages: { "message": { "type": "transcript", ... } }

    message = data.get('message', {})
    event_type = data.get('type') or message.get('type')

    logger.info(f"========== WEBHOOK RECEIVED ==========")
    logger.info(f"Event Type: {event_type}")
    logger.info(f"Full Data: {data}")
    logger.info(f"=======================================")

    # Ako je server message, ekstrahiraj relevantne podatke
    if 'message' in data:
        return handle_server_message(data)

    with get_db() as conn:
        with conn.cursor() as cur:
            if event_type == 'call.started':
                call_id = data['call']['id']
                phone = data['call']['phoneNumber']
                started = data['call']['startedAt']
                cur.execute("""
                    INSERT INTO calls (vapi_call_id, phone_number, started_at, status)
                    VALUES (%s, %s, %s, 'active')
                    ON CONFLICT (vapi_call_id) DO NOTHING
                """, (call_id, phone, started))

            elif event_type == 'call.transcript.update':
                call_id = data['call']['id']
                speaker = data['transcript']['speaker']
                text = data['transcript']['text']
                timestamp = data['transcript']['timestamp']
                cur.execute("""
                    INSERT INTO transcripts (call_id, speaker, text, timestamp)
                    SELECT id, %s, %s, %s FROM calls WHERE vapi_call_id = %s
                """, (speaker, text, timestamp, call_id))

            elif event_type == 'call.ended':
                call_id = data['call']['id']
                ended = data['call']['endedAt']
                duration = data['call']['duration']
                cur.execute("""
                    UPDATE calls
                    SET ended_at = %s, duration_seconds = %s, status = 'completed'
                    WHERE vapi_call_id = %s
                """, (ended, duration, call_id))

            elif event_type == 'call.forwarded':
                call_id = data['call']['id']
                reason = data.get('reason', '')
                cur.execute("""
                    INSERT INTO call_metadata (call_id, escalated, escalation_reason)
                    SELECT id, TRUE, %s FROM calls WHERE vapi_call_id = %s
                    ON CONFLICT (call_id) DO UPDATE
                    SET escalated = TRUE, escalation_reason = %s
                """, (reason, call_id, reason))

            conn.commit()

    return jsonify({'success': True}), 200

@app.route('/api/calls/active', methods=['GET'])
def get_active_calls():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT vapi_call_id as id, phone_number as "phoneNumber", started_at as "startedAt"
                FROM calls
                WHERE status = 'active'
                ORDER BY started_at DESC
            """)
            rows = cur.fetchall()
            result = [dict(zip(['id', 'phoneNumber', 'startedAt'], row)) for row in rows]
    return jsonify(result)

@app.route('/api/calls/completed', methods=['GET'])
def get_completed_calls():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.vapi_call_id as id, c.phone_number as "phoneNumber",
                       c.started_at as "startedAt", c.duration_seconds as duration,
                       cm.ai_summary as summary
                FROM calls c
                LEFT JOIN call_metadata cm ON c.id = cm.call_id
                WHERE c.status = 'completed'
                  AND (cm.escalated = FALSE OR cm.escalated IS NULL)
                ORDER BY c.started_at DESC
            """)
            rows = cur.fetchall()
            result = [dict(zip(['id', 'phoneNumber', 'startedAt', 'duration', 'summary'], row)) for row in rows]
    return jsonify(result)

@app.route('/api/calls/forwarded', methods=['GET'])
def get_forwarded_calls():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.vapi_call_id as id, c.phone_number as "phoneNumber",
                       c.started_at as "startedAt", c.duration_seconds as duration,
                       cm.escalation_reason as "forwardReason"
                FROM calls c
                JOIN call_metadata cm ON c.id = cm.call_id
                WHERE cm.escalated = TRUE
                  AND cm.escalation_resolved = FALSE
                ORDER BY c.started_at DESC
            """)
            rows = cur.fetchall()
            result = [dict(zip(['id', 'phoneNumber', 'startedAt', 'duration', 'forwardReason'], row)) for row in rows]
    return jsonify(result)

@app.route('/api/calls', methods=['GET'])
def get_all_calls():
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('pageSize', 20))
    offset = (page - 1) * page_size

    with get_db() as conn:
        with conn.cursor() as cur:
            # Get total count
            cur.execute("SELECT COUNT(*) FROM calls")
            total = cur.fetchone()[0]

            # Get paginated calls
            cur.execute("""
                SELECT c.vapi_call_id, c.phone_number, c.started_at, c.ended_at,
                       c.duration_seconds, c.status, cm.ai_summary, cm.escalation_reason
                FROM calls c
                LEFT JOIN call_metadata cm ON c.id = cm.call_id
                ORDER BY c.started_at DESC
                LIMIT %s OFFSET %s
            """, (page_size, offset))
            rows = cur.fetchall()

            items = []
            for row in rows:
                # Calculate duration if not provided by Vapi
                duration_sec = row[4]  # duration_seconds from DB
                if not duration_sec and row[2] and row[3]:  # started_at and ended_at exist
                    duration_sec = int((row[3] - row[2]).total_seconds())

                items.append({
                    'vapiCallId': row[0],
                    'callerMasked': row[1],  # TODO: mask phone number
                    'startedAt': row[2].isoformat() if row[2] else None,
                    'durationSec': duration_sec or 0,
                    'outcome': 'escalated' if row[7] else ('resolved' if row[5] == 'completed' else 'active'),
                    'reason': row[7] or 'General inquiry',
                    'summary': row[6] or ''
                })

    return jsonify({
        'page': page,
        'pageSize': page_size,
        'total': total,
        'items': items
    })

@app.route('/api/calls/<call_id>/transcript', methods=['GET'])
def get_transcript(call_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT t.speaker, t.text, t.timestamp
                FROM transcripts t
                JOIN calls c ON t.call_id = c.id
                WHERE c.vapi_call_id = %s
                ORDER BY t.timestamp
            """, (call_id,))
            rows = cur.fetchall()
            result = [dict(zip(['speaker', 'text', 'timestamp'], row)) for row in rows]
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

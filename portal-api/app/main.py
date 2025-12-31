import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .auth import AuthContext, verify_token
from .config import SLA_TARGET_DEFAULT_SEC, WEBHOOK_SECRET
from .db import get_db_cursor
from .schemas import Settings, UserContext, settings_to_dict

load_dotenv()

app = FastAPI(title="Voice Agent Portal API")


def _require_auth_header(auth_header: str | None) -> str:
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    return auth_header.replace("Bearer ", "", 1).strip()


def _get_auth_context(authorization: str = Header(default=None)) -> AuthContext:
    token = _require_auth_header(authorization)
    try:
        return verify_token(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _set_tenant_scope(cursor, tenant_id: str) -> None:
    cursor.execute("select set_config('app.tenant_id', %s, true)", (tenant_id,))


def _load_user_context(auth_ctx: AuthContext) -> UserContext:
    with get_db_cursor() as (conn, cursor):
        cursor.execute(
            """
            select u.id, u.email, u.full_name, t.id, t.name, utr.role
            from users u
            join user_tenant_roles utr on utr.user_id = u.id
            join tenants t on t.id = utr.tenant_id
            where u.email = %s
            """,
            (auth_ctx.email,),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=403, detail="User not mapped to a tenant.")
        user_id, email, full_name, tenant_id, tenant_name, role = row
        return UserContext(
            user={"id": str(user_id), "email": email, "fullName": full_name},
            tenant={"id": str(tenant_id), "name": tenant_name},
            role=role,
        )


def _get_settings(cursor, tenant_id: str) -> Settings:
    cursor.execute(
        """
        select sla_target_sec, human_minutes_per_resolved_call, cost_per_minute,
               currency, retention_days, mask_caller, recordings_enabled,
               recordings_require_manager, updated_at
        from tenant_settings
        where tenant_id = %s
        """,
        (tenant_id,),
    )
    row = cursor.fetchone()
    if row:
        return Settings(*row)
    return Settings(
        sla_target_sec=SLA_TARGET_DEFAULT_SEC,
        human_minutes_per_resolved_call=3.5,
        cost_per_minute=0.5,
        currency="EUR",
        retention_days=90,
        mask_caller=True,
        recordings_enabled=True,
        recordings_require_manager=True,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/me")
def get_me(auth_ctx: AuthContext = Depends(_get_auth_context)) -> dict[str, Any]:
    return _load_user_context(auth_ctx).__dict__


@app.get("/kpi/overview")
def get_overview(
    from_: str = Query(..., alias="from"),
    to: str,
    auth_ctx: AuthContext = Depends(_get_auth_context),
) -> dict[str, Any]:
    user_ctx = _load_user_context(auth_ctx)
    with get_db_cursor() as (conn, cursor):
        _set_tenant_scope(cursor, user_ctx.tenant["id"])
        settings = _get_settings(cursor, user_ctx.tenant["id"])
        cursor.execute(
            """
            with base as (
                select *
                from calls
                where started_at >= %s and started_at < %s
            )
            select
                count(*) as total_calls,
                count(*) filter (where answered_at is not null) as answered_calls,
                round(100.0 * count(*) filter (where answered_at is not null) / nullif(count(*),0), 2) as ai_answer_rate_pct,
                count(*) filter (where outcome = 'resolved') as resolved_calls,
                round(100.0 * count(*) filter (where outcome = 'resolved') / nullif(count(*),0), 2) as resolution_rate_pct,
                count(*) filter (where outcome = 'escalated') as escalated_calls,
                round(100.0 * count(*) filter (where outcome = 'escalated') / nullif(count(*),0), 2) as escalation_rate_pct,
                count(*) filter (where outcome = 'failed') as failed_calls,
                count(*) filter (where outcome = 'abandoned') as abandoned_calls,
                round(avg(answer_time_sec)::numeric, 2) as avg_answer_time_sec,
                percentile_cont(0.95) within group (order by answer_time_sec) as p95_answer_time_sec
            from base
            """,
            (from_, to),
        )
        row = cursor.fetchone()
        if not row:
            return {}
        (
            total_calls,
            answered_calls,
            ai_answer_rate_pct,
            resolved_calls,
            resolution_rate_pct,
            escalated_calls,
            escalation_rate_pct,
            failed_calls,
            abandoned_calls,
            avg_answer_time_sec,
            p95_answer_time_sec,
        ) = row

        cursor.execute(
            """
            with base as (
                select answer_time_sec
                from calls
                where started_at >= %s and started_at < %s
                  and answer_time_sec is not null
            )
            select
                round(
                    100.0 * count(*) filter (where answer_time_sec <= %s) / nullif(count(*),0),
                    2
                ) as sla_compliance_pct
            from base
            """,
            (from_, to, settings.sla_target_sec),
        )
        sla_row = cursor.fetchone()
        sla_compliance_pct = sla_row[0] if sla_row else None
        return {
            "totalCalls": total_calls,
            "answeredCalls": answered_calls,
            "aiAnswerRatePct": ai_answer_rate_pct,
            "resolvedCalls": resolved_calls,
            "resolutionRatePct": resolution_rate_pct,
            "escalatedCalls": escalated_calls,
            "escalationRatePct": escalation_rate_pct,
            "failedCalls": failed_calls,
            "abandonedCalls": abandoned_calls,
            "avgAnswerTimeSec": avg_answer_time_sec,
            "p95AnswerTimeSec": p95_answer_time_sec,
            "slaTargetSec": settings.sla_target_sec,
            "slaCompliancePct": sla_compliance_pct,
        }


@app.get("/kpi/overview/trend")
def get_overview_trend(
    from_: str = Query(..., alias="from"),
    to: str,
    auth_ctx: AuthContext = Depends(_get_auth_context),
) -> dict[str, Any]:
    user_ctx = _load_user_context(auth_ctx)
    with get_db_cursor() as (conn, cursor):
        _set_tenant_scope(cursor, user_ctx.tenant["id"])
        cursor.execute(
            """
            select
                date_trunc('day', started_at) as day,
                count(*) as total,
                count(*) filter (where outcome = 'resolved') as resolved,
                count(*) filter (where outcome = 'escalated') as escalated,
                count(*) filter (where outcome = 'failed') as failed,
                count(*) filter (where outcome = 'abandoned') as abandoned
            from calls
            where started_at >= %s and started_at < %s
            group by 1
            order by 1 asc
            """,
            (from_, to),
        )
        points = [
            {
                "day": row[0].date().isoformat(),
                "total": row[1],
                "resolved": row[2],
                "escalated": row[3],
                "failed": row[4],
                "abandoned": row[5],
            }
            for row in cursor.fetchall()
        ]
        return {"points": points}


@app.get("/kpi/reasons")
def get_reasons(
    from_: str = Query(..., alias="from"),
    to: str,
    limit: int = 10,
    auth_ctx: AuthContext = Depends(_get_auth_context),
) -> dict[str, Any]:
    user_ctx = _load_user_context(auth_ctx)
    with get_db_cursor() as (conn, cursor):
        _set_tenant_scope(cursor, user_ctx.tenant["id"])
        cursor.execute(
            """
            select
                coalesce(reason, 'unknown') as reason,
                count(*) as calls,
                round(100.0 * count(*) filter (where outcome = 'resolved') / nullif(count(*),0), 2) as resolved_pct,
                round(100.0 * count(*) filter (where outcome = 'escalated') / nullif(count(*),0), 2) as escalated_pct,
                round(avg(duration_sec)::numeric, 2) as avg_duration_sec
            from calls
            where started_at >= %s and started_at < %s
            group by 1
            order by calls desc
            limit %s
            """,
            (from_, to, limit),
        )
        items = [
            {
                "reason": row[0],
                "calls": row[1],
                "resolvedPct": row[2],
                "escalatedPct": row[3],
                "avgDurationSec": row[4],
            }
            for row in cursor.fetchall()
        ]
        return {"items": items}


@app.get("/kpi/impact")
def get_impact(
    from_: str = Query(..., alias="from"),
    to: str,
    auth_ctx: AuthContext = Depends(_get_auth_context),
) -> dict[str, Any]:
    user_ctx = _load_user_context(auth_ctx)
    with get_db_cursor() as (conn, cursor):
        _set_tenant_scope(cursor, user_ctx.tenant["id"])
        settings = _get_settings(cursor, user_ctx.tenant["id"])
        cursor.execute(
            """
            select count(*) filter (where outcome = 'resolved')
            from calls
            where started_at >= %s and started_at < %s
            """,
            (from_, to),
        )
        resolved_calls = cursor.fetchone()[0] or 0
        human_minutes_saved = resolved_calls * settings.human_minutes_per_resolved_call
        cost_saved = human_minutes_saved * settings.cost_per_minute
        return {
            "humanMinutesSaved": human_minutes_saved,
            "estimatedCostSaved": cost_saved,
            "assumptions": {
                "humanMinutesPerResolvedCall": settings.human_minutes_per_resolved_call,
                "costPerMinute": settings.cost_per_minute,
                "currency": settings.currency,
            },
        }


@app.get("/kpi/sla")
def get_sla(
    from_: str = Query(..., alias="from"),
    to: str,
    auth_ctx: AuthContext = Depends(_get_auth_context),
) -> dict[str, Any]:
    user_ctx = _load_user_context(auth_ctx)
    with get_db_cursor() as (conn, cursor):
        _set_tenant_scope(cursor, user_ctx.tenant["id"])
        settings = _get_settings(cursor, user_ctx.tenant["id"])
        cursor.execute(
            """
            with base as (
                select answer_time_sec
                from calls
                where started_at >= %s and started_at < %s
                  and answer_time_sec is not null
            )
            select
                round(100.0 * count(*) filter (where answer_time_sec <= %s) / nullif(count(*),0), 2) as sla_compliance_pct,
                round(100.0 * count(*) filter (where answer_time_sec <= 5) / nullif(count(*),0), 2) as within_5s_pct,
                round(100.0 * count(*) filter (where answer_time_sec <= 10) / nullif(count(*),0), 2) as within_10s_pct,
                round(100.0 * count(*) filter (where answer_time_sec <= 20) / nullif(count(*),0), 2) as within_20s_pct,
                count(*) as sample_size
            from base
            """,
            (from_, to, settings.sla_target_sec),
        )
        row = cursor.fetchone()
        if not row:
            return {}
        return {
            "slaTargetSec": settings.sla_target_sec,
            "slaCompliancePct": row[0],
            "within5SecPct": row[1],
            "within10SecPct": row[2],
            "within20SecPct": row[3],
            "sampleSizeAnswered": row[4],
        }


@app.get("/kpi/reliability")
def get_reliability(
    from_: str = Query(..., alias="from"),
    to: str,
    auth_ctx: AuthContext = Depends(_get_auth_context),
) -> dict[str, Any]:
    user_ctx = _load_user_context(auth_ctx)
    with get_db_cursor() as (conn, cursor):
        _set_tenant_scope(cursor, user_ctx.tenant["id"])
        cursor.execute(
            """
            select
                count(*) filter (where outcome = 'failed') as failed_calls,
                count(*) filter (where error_code is not null) as errored_calls,
                count(*) filter (where escalated = true) as escalations
            from calls
            where started_at >= %s and started_at < %s
            """,
            (from_, to),
        )
        row = cursor.fetchone()
        return {
            "failedCalls": row[0],
            "erroredCalls": row[1],
            "escalations": row[2],
            "transferFailures": 0,
        }


@app.get("/calls")
def get_calls(
    from_: str = Query(..., alias="from"),
    to: str,
    outcome: str | None = None,
    reason: str | None = None,
    q: str | None = None,
    page: int = 1,
    pageSize: int = 20,
    auth_ctx: AuthContext = Depends(_get_auth_context),
) -> dict[str, Any]:
    user_ctx = _load_user_context(auth_ctx)
    offset = (page - 1) * pageSize
    with get_db_cursor() as (conn, cursor):
        _set_tenant_scope(cursor, user_ctx.tenant["id"])
        cursor.execute(
            """
            select count(*)
            from calls
            where started_at >= %s and started_at < %s
              and (%s is null or outcome = %s)
              and (%s is null or reason = %s)
              and (
                    %s is null
                    or vapi_call_id ilike '%%' || %s || '%%'
                    or summary ilike '%%' || %s || '%%'
              )
            """,
            (from_, to, outcome, outcome, reason, reason, q, q, q),
        )
        total = cursor.fetchone()[0]
        cursor.execute(
            """
            select
                vapi_call_id,
                started_at,
                duration_sec,
                outcome,
                reason,
                summary,
                case when caller_e164 is null then null else '***' || right(caller_e164, 4) end as caller_masked
            from calls
            where started_at >= %s and started_at < %s
              and (%s is null or outcome = %s)
              and (%s is null or reason = %s)
              and (
                    %s is null
                    or vapi_call_id ilike '%%' || %s || '%%'
                    or summary ilike '%%' || %s || '%%'
              )
            order by started_at desc
            limit %s offset %s
            """,
            (from_, to, outcome, outcome, reason, reason, q, q, q, pageSize, offset),
        )
        items = [
            {
                "vapiCallId": row[0],
                "startedAt": row[1].isoformat() if row[1] else None,
                "durationSec": row[2],
                "outcome": row[3],
                "reason": row[4],
                "summary": row[5],
                "callerMasked": row[6],
            }
            for row in cursor.fetchall()
        ]
        return {"page": page, "pageSize": pageSize, "total": total, "items": items}


@app.get("/calls/{vapi_call_id}")
def get_call_detail(
    vapi_call_id: str,
    auth_ctx: AuthContext = Depends(_get_auth_context),
) -> dict[str, Any]:
    user_ctx = _load_user_context(auth_ctx)
    with get_db_cursor() as (conn, cursor):
        _set_tenant_scope(cursor, user_ctx.tenant["id"])
        cursor.execute(
            """
            select
                vapi_call_id,
                started_at,
                answered_at,
                ended_at,
                duration_sec,
                answer_time_sec,
                direction,
                caller_e164,
                called_e164,
                outcome,
                escalated,
                reason,
                summary,
                transcript_text,
                recording_url,
                structured_data,
                ended_reason,
                error_code
            from calls
            where vapi_call_id = %s
            """,
            (vapi_call_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Call not found.")
        caller_masked = f"***{row[7][-4:]}" if row[7] else None
        called_masked = f"***{row[8][-4:]}" if row[8] else None
        return {
            "vapiCallId": row[0],
            "startedAt": row[1].isoformat() if row[1] else None,
            "answeredAt": row[2].isoformat() if row[2] else None,
            "endedAt": row[3].isoformat() if row[3] else None,
            "durationSec": row[4],
            "answerTimeSec": row[5],
            "direction": row[6],
            "callerMasked": caller_masked,
            "calledMasked": called_masked,
            "outcome": row[9],
            "escalated": row[10],
            "reason": row[11],
            "summary": row[12],
            "transcriptText": row[13],
            "recording": {"available": row[14] is not None, "url": None},
            "structuredData": row[15],
            "technical": {"endedReason": row[16], "errorCode": row[17]},
        }


@app.get("/calls/{vapi_call_id}/recording")
def get_recording(
    vapi_call_id: str,
    auth_ctx: AuthContext = Depends(_get_auth_context),
) -> dict[str, Any]:
    user_ctx = _load_user_context(auth_ctx)
    with get_db_cursor() as (conn, cursor):
        _set_tenant_scope(cursor, user_ctx.tenant["id"])
        settings = _get_settings(cursor, user_ctx.tenant["id"])
        if not settings.recordings_enabled:
            raise HTTPException(status_code=403, detail="Recordings disabled.")
        if settings.recordings_require_manager and user_ctx.role == "viewer":
            raise HTTPException(status_code=403, detail="Recording access denied.")
        cursor.execute(
            "select recording_url from calls where vapi_call_id = %s",
            (vapi_call_id,),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            raise HTTPException(status_code=404, detail="Recording not available.")
        return {"url": row[0], "expiresAt": None}


@app.get("/settings")
def get_settings(auth_ctx: AuthContext = Depends(_get_auth_context)) -> dict[str, Any]:
    user_ctx = _load_user_context(auth_ctx)
    with get_db_cursor() as (conn, cursor):
        _set_tenant_scope(cursor, user_ctx.tenant["id"])
        settings = _get_settings(cursor, user_ctx.tenant["id"])
        return settings_to_dict(settings)


@app.patch("/settings")
def update_settings(
    payload: dict[str, Any],
    auth_ctx: AuthContext = Depends(_get_auth_context),
) -> dict[str, Any]:
    user_ctx = _load_user_context(auth_ctx)
    if user_ctx.role != "owner":
        raise HTTPException(status_code=403, detail="Owner role required.")
    fields = {
        "sla_target_sec": payload.get("slaTargetSec"),
        "human_minutes_per_resolved_call": payload.get("humanMinutesPerResolvedCall"),
        "cost_per_minute": payload.get("costPerMinute"),
        "currency": payload.get("currency"),
        "retention_days": payload.get("retentionDays"),
        "mask_caller": payload.get("maskCaller"),
        "recordings_enabled": payload.get("recordingsEnabled"),
        "recordings_require_manager": payload.get("recordingsRequireManager"),
    }
    updates = {k: v for k, v in fields.items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No settings to update.")
    with get_db_cursor() as (conn, cursor):
        _set_tenant_scope(cursor, user_ctx.tenant["id"])
        columns = ", ".join(f"{key} = %s" for key in updates.keys())
        cursor.execute(
            f"""
            insert into tenant_settings (tenant_id, {", ".join(updates.keys())})
            values (%s, {", ".join(["%s"] * len(updates))})
            on conflict (tenant_id)
            do update set {columns}, updated_at = now()
            """,
            [user_ctx.tenant["id"], *updates.values()],
        )
        conn.commit()
        settings = _get_settings(cursor, user_ctx.tenant["id"])
        return settings_to_dict(settings)


def _verify_signature(raw_body: bytes, signature: str, timestamp: str | None) -> bool:
    if not WEBHOOK_SECRET:
        return False
    signed_payload = raw_body
    if timestamp:
        signed_payload = f"{timestamp}.".encode() + raw_body
    expected = hmac.new(WEBHOOK_SECRET.encode(), signed_payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def _parse_timestamp(header_value: str | None) -> int | None:
    if not header_value:
        return None
    try:
        return int(header_value)
    except ValueError:
        return None


@app.post("/webhooks/vapi")
async def vapi_webhook(
    request: Request,
    x_vapi_signature: str | None = Header(default=None),
    x_vapi_timestamp: str | None = Header(default=None),
    x_signature: str | None = Header(default=None),
    x_timestamp: str | None = Header(default=None),
) -> JSONResponse:
    raw_body = await request.body()
    signature = x_vapi_signature or x_signature
    timestamp = x_vapi_timestamp or x_timestamp
    if not signature:
        raise HTTPException(status_code=401, detail="Missing signature.")
    if timestamp:
        ts = _parse_timestamp(timestamp)
        if ts:
            now = int(datetime.now(timezone.utc).timestamp())
            if abs(now - ts) > 300:
                raise HTTPException(status_code=401, detail="Stale webhook timestamp.")
    if not _verify_signature(raw_body, signature, timestamp):
        raise HTTPException(status_code=401, detail="Invalid signature.")

    payload = json.loads(raw_body.decode())
    message = payload.get("message", {})
    event_type = message.get("type", "unknown")
    call = message.get("call", {}) or {}
    phone_number = message.get("phoneNumber", {}) or {}
    call_id = call.get("id") or call.get("callId")

    phone_number_id = phone_number.get("id") or call.get("phoneNumberId")
    assistant_id = call.get("assistantId")
    org_id = call.get("orgId")

    with get_db_cursor() as (conn, cursor):
        cursor.execute(
            """
            select tenant_id
            from tenant_vapi_map
            where vapi_phone_number_id = %s
               or vapi_assistant_id = %s
               or vapi_org_id = %s
            limit 1
            """,
            (phone_number_id, assistant_id, org_id),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Tenant mapping not found.")
        tenant_id = row[0]

        payload_hash = hashlib.sha256(raw_body).hexdigest()
        try:
            cursor.execute(
                """
                insert into call_events_raw (tenant_id, vapi_call_id, event_type, payload, payload_hash)
                values (%s, %s, %s, %s, %s)
                """,
                (tenant_id, call_id, event_type, json.dumps(payload), payload_hash),
            )
        except Exception:
            conn.rollback()
            return JSONResponse(content={"status": "duplicate"}, status_code=200)

        started_at = call.get("startedAt")
        ended_at = call.get("endedAt")
        answered_at = call.get("answeredAt")
        duration_sec = call.get("duration") or call.get("durationSec")
        direction = call.get("direction") or "inbound"
        caller = call.get("customer", {}).get("number") or call.get("from")
        called = call.get("to")
        outcome = call.get("outcome")
        escalated = call.get("escalated") or False
        ended_reason = call.get("endedReason")
        summary = call.get("summary")
        transcript_text = call.get("transcript")
        recording_url = call.get("recordingUrl")
        structured_data = call.get("analysis")
        error_code = call.get("error")

        cursor.execute(
            """
            insert into calls (
                tenant_id, vapi_call_id, vapi_org_id, vapi_phone_number_id, vapi_assistant_id,
                direction, caller_e164, called_e164,
                started_at, answered_at, ended_at, duration_sec,
                outcome, escalated, ended_reason, reason,
                recording_url, transcript_text, summary,
                structured_data, error_code
            )
            values (
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s
            )
            on conflict (vapi_call_id) do update set
                vapi_org_id = coalesce(excluded.vapi_org_id, calls.vapi_org_id),
                vapi_phone_number_id = coalesce(excluded.vapi_phone_number_id, calls.vapi_phone_number_id),
                vapi_assistant_id = coalesce(excluded.vapi_assistant_id, calls.vapi_assistant_id),
                started_at = coalesce(calls.started_at, excluded.started_at),
                answered_at = coalesce(calls.answered_at, excluded.answered_at),
                ended_at = coalesce(excluded.ended_at, calls.ended_at),
                duration_sec = coalesce(excluded.duration_sec, calls.duration_sec),
                recording_url = coalesce(excluded.recording_url, calls.recording_url),
                transcript_text = coalesce(excluded.transcript_text, calls.transcript_text),
                summary = coalesce(excluded.summary, calls.summary),
                structured_data = coalesce(excluded.structured_data, calls.structured_data),
                escalated = calls.escalated or excluded.escalated,
                outcome = coalesce(excluded.outcome, calls.outcome),
                ended_reason = coalesce(excluded.ended_reason, calls.ended_reason),
                error_code = coalesce(excluded.error_code, calls.error_code),
                updated_at = now()
            """,
            (
                tenant_id,
                call_id,
                org_id,
                phone_number_id,
                assistant_id,
                direction,
                caller,
                called,
                started_at,
                answered_at,
                ended_at,
                duration_sec,
                outcome,
                escalated,
                ended_reason,
                call.get("reason"),
                recording_url,
                transcript_text,
                summary,
                json.dumps(structured_data) if structured_data else None,
                error_code,
            ),
        )
        cursor.execute(
            """
            update calls
            set answer_time_sec = case
                when started_at is not null and answered_at is not null
                then extract(epoch from (answered_at - started_at))::int
                else answer_time_sec
            end
            where vapi_call_id = %s
            """,
            (call_id,),
        )
        conn.commit()

    return JSONResponse(content={"status": "ok"})

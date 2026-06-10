import json
from datetime import datetime

from sqlalchemy import text

from app.database import SessionLocal


def create_job_record(job_id: str, user_id: str):
    now = datetime.now()

    db = SessionLocal()
    try:
        db.execute(
            text("""
                INSERT INTO upload_jobs (
                    job_id,
                    user_id,
                    status,
                    result_json,
                    error,
                    created_at,
                    updated_at
                )
                VALUES (
                    :job_id,
                    :user_id,
                    :status,
                    :result_json,
                    :error,
                    :created_at,
                    :updated_at
                )
            """),
            {
                "job_id": job_id,
                "user_id": user_id,
                "status": "pending",
                "result_json": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
        )

        db.commit()

    finally:
        db.close()


def get_job_record(job_id: str, user_id: str):
    db = SessionLocal()
    try:
        row = db.execute(
            text("""
                SELECT
                    job_id,
                    user_id,
                    status,
                    result_json,
                    error,
                    created_at,
                    updated_at
                FROM upload_jobs
                WHERE job_id = :job_id
                  AND user_id = :user_id
            """),
            {
                "job_id": job_id,
                "user_id": user_id,
            }
        ).mappings().first()

        if row is None:
            return None

        result = row["result_json"]

        if isinstance(result, str):
            result = json.loads(result)

        return {
            "job_id": row["job_id"],
            "user_id": row["user_id"],
            "status": row["status"],
            "result": result,
            "error": row["error"],
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
        }

    finally:
        db.close()


def update_job_record(job_id: str, status: str, result=None, error=None):
    now = datetime.now()

    result_json = None
    if result is not None:
        result_json = json.dumps(result)

    db = SessionLocal()
    try:
        db.execute(
            text("""
                UPDATE upload_jobs
                SET
                    status = :status,
                    result_json = COALESCE(:result_json, result_json),
                    error = COALESCE(:error, error),
                    updated_at = :updated_at
                WHERE job_id = :job_id
            """),
            {
                "job_id": job_id,
                "status": status,
                "result_json": result_json,
                "error": error,
                "updated_at": now,
            }
        )

        db.commit()

    finally:
        db.close()

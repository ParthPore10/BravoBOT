from uuid import uuid4

from app.job_store import create_job_record, get_job_record, update_job_record



def create_jobs()->str:
    job_id = f"job_{uuid4().hex}"
    
    create_job_record(job_id)
    
    return job_id

def update_job(job_id: str, status: str, result=None, error=None):
    update_job_record(
        job_id=job_id,
        status=status,
        result=result,
        error=error
    )

def get_job(job_id :str):
    return get_job_record(job_id)

def run_ingestion_job(job_id :str, ingest_fun, uploaded_files, replace :bool):
    update_job(job_id=job_id,
               status='running')
    try:
        result = ingest_fun(
            uploaded_files = uploaded_files,
            replace =replace
        )
        update_job(
            job_id=job_id,
            status="completed",
            result=result
        )
    except Exception as exc:
        print(f"Indexing failed for job {job_id}: {exc}")
        update_job(job_id=job_id, status="failed", error=str(exc))

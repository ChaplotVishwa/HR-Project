import os
from fastapi import APIRouter, UploadFile, Form, HTTPException
from app.projects.cv_extract.controllers.resume_controller import bulk_extract_resumes_controller
from app.projects.resume_ranking.hybrid_ranking_system import HybridRankingSystem
import pandas as pd

router = APIRouter()
ranker = HybridRankingSystem()

@router.post("/pipeline")
async def unified_pipeline(files: list[UploadFile], job_description: str = Form(...)):
    """
    Unified HR Pipeline:
    1. Bulk Extract resumes (PDF -> JSON)
    2. Rank extracted candidates against job description
    """
    try:
        # 1. Bulk Extraction
        extraction_result = await bulk_extract_resumes_controller(files, include_analysis=True)
        if extraction_result["status"] != "success":
             raise HTTPException(status_code=500, detail="Extraction failed")
        
        candidates = []
        for res in extraction_result["results"]:
            if res.get("status") == "success":
                data = res["data"]
                # ResumeData stores name/email/phone inside personal_info
                personal_info = data.get("personal_info", {})
                skills_data = data.get("skills", {})
                # Flatten skills from nested dict to a simple list
                all_skills = []
                if isinstance(skills_data, dict):
                    for k in ["technical", "soft", "languages", "tools"]:
                        all_skills.extend(skills_data.get(k, []))
                elif isinstance(skills_data, list):
                    all_skills = skills_data

                candidate_entry = {
                    "name": personal_info.get("name", "Unknown"),
                    "email": personal_info.get("email", ""),
                    "phone": personal_info.get("phone", ""),
                    "education": str(data.get("education", [])),
                    "experience": str(data.get("experience", [])),
                    "skills": all_skills,
                    "summary": data.get("summary", ""),
                    "experience_years": res.get("analysis", {}).get("experience_years", 0)
                }
                candidates.append(candidate_entry)
            else:
                print(f"[WARN] Skipping failed extraction: {res.get('message', 'unknown error')}")
        
        if not candidates:
             return {"status": "error", "message": "No valid resumes extracted"}

        # 2. Ranking
        ranking_input = {
            "job_description": {"description": job_description, "title": "Target Role"},
            "candidates": candidates
        }
        results_df, metadata = ranker.rank_candidates(ranking_input)

        final_results = []
        for i, row in results_df.iterrows():
            candidate_data = {
                "rank": i + 1,
                "name": row.get("name", ""),
                "email": row.get("email", ""),
                "total_score": round(float(row.get("total_score", 0)), 2),
                "skills": row.get("skills", []),
                "ai_insights": row.get("ai_insights", {})
            }
            final_results.append(candidate_data)

        return {
            "status": "success",
            "count": len(final_results),
            "results": final_results,
            "metadata": {
                "algorithm": metadata.get("algorithm_used"),
                "confidence": metadata.get("confidence", {}).get("confidence_level")
            }
        }

    except Exception as e:
        print(f"[ERROR] Unified Pipeline Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# app/projects/resume_ranking/routes/rank_routes.py

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.projects.resume_ranking.hybrid_ranking_system import HybridRankingSystem
from datetime import datetime
from typing import Optional
import json

# Initialize the ranking system
ranker_api = HybridRankingSystem()

router = APIRouter()

def make_json_safe(obj):
    """Recursively convert non-JSON-serializable objects"""
    import pandas as pd
    import numpy as np

    if isinstance(obj, pd.DataFrame):
        # Convert DataFrame to dictionary and ensure all numbers are float
        return obj.applymap(lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x).to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.apply(lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x).tolist()
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_safe(v) for v in obj)
    elif isinstance(obj, (np.generic,)):
        return obj.item()  # Converts np.float32 to Python native float
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


@router.post("/rank")
async def rank_candidates_api(request: Request):
    """Accepts JSON with job_description + candidates/resumes and returns ranked output."""
    try:
        input_data = await request.json()
        print("\n📥 Received new ranking request")

        # Call rank_candidates from HybridRankingSystem
        results_df, metadata = ranker_api.rank_candidates(input_data)

        candidates_with_ranking = []
        for i, row in results_df.iterrows():
            candidate_data = {
                "rank": i + 1,
                "name": row.get("name", ""),
                "email": row.get("email", ""),
                "phone": row.get("phone", ""),
                "education": row.get("education", ""),
                "experience_years": row.get("experience_years", 0),
                "skills": row.get("skills", []),
                "certifications": row.get("certifications", []),
                "total_score": row.get("total_score", 0),
                "percentile": row.get("percentile", 0),
                "score_breakdown": row.get("score_components", {}),
                "ai_analysis": row.get("ai_insights", {}),
                "skills_preview": ", ".join(row.get("skills", []))[:5],
            }
            candidates_with_ranking.append(candidate_data)

        response_data = {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": len(candidates_with_ranking),
            "candidates": candidates_with_ranking,
            "confidence": metadata.get("confidence", {}),
            "validation": metadata.get("validation", {}),
            "verification": metadata.get("verification", {}),
            "llm_status": metadata.get("llm_status", {}),
            "algorithm_used": metadata.get("algorithm_used", "llm_enhanced_scoring")
        }

        safe_response = make_json_safe(response_data)
        return JSONResponse(content=safe_response)
    except Exception as e:
        print("❌ Error processing request:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

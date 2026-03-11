# app/projects/resume_ranking/controllers/ranking_controller.py

from app.projects.resume_ranking.services.ranking_service import RankingService
from app.config.config import LLMConfig

def rank_candidates(data):
    """Handles ranking requests."""
    ranking_service = RankingService(llm_config=LLMConfig())
    return ranking_service.rank_candidates(data)

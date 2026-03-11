# app/projects/resume_ranking/services/ranking_service.py

import torch
from sentence_transformers import SentenceTransformer, util
from app.config.config import LLMConfig
from typing import List, Dict
import numpy as np
import re
import time

class RankingService:
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.device = self._initialize_device()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

    def _initialize_device(self):
        """Sets the device based on available hardware"""
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        else:
            print("Using CPU")
            return 'cpu'

    def rank_candidates(self, data: Dict) -> List[Dict]:
        """Rank the candidates based on the job description"""
        candidates = data.get('candidates', [])
        job_desc = data.get('job_description', {})

        ranked_candidates = self._compute_and_rank(candidates, job_desc)
        return ranked_candidates, {}

    def _compute_and_rank(self, candidates: List[Dict], job_desc: Dict) -> List[Dict]:
        """Rank candidates based on the similarity to job description"""
        ranked = sorted(candidates, key=lambda x: self._compute_similarity(x, job_desc), reverse=True)
        return ranked

    def _compute_similarity(self, candidate: Dict, job_desc: Dict) -> float:
        """Calculate semantic similarity between candidate and job description"""
        candidate_text = candidate.get('resume_text', '')
        job_text = job_desc.get('description', '')
        candidate_embedding = self.embedder.encode(candidate_text, convert_to_tensor=True, show_progress_bar=False)
        job_embedding = self.embedder.encode(job_text, convert_to_tensor=True, show_progress_bar=False)
        return float(util.pytorch_cos_sim(candidate_embedding, job_embedding)[0][0])

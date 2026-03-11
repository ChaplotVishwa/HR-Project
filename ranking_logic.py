# ==========================================================
# CELL 1: Import & Install Environment
# ==========================================================
import subprocess
import sys
import os
import json
import gc
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder

# Environment setup
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')


# ==========================================================
# CELL 2: Utility Functions
# ==========================================================
def clear_memory():
    """Clear memory to prevent OOM errors"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def environment_check():
    """Prints basic system and GPU info"""
    print("=" * 60)
    print(" ENVIRONMENT CHECK")
    print("=" * 60)
    print(f"[OK] Python Version: {sys.version.split()[0]}")
    print(f"[OK] PyTorch Version: {torch.__version__}")
    print(f"[OK] GPU Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"[OK] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        clear_memory()
        return 'cuda'
    else:
        print(" Running on CPU")
        return 'cpu'


device = environment_check()


# ==========================================================
# CELL 3: LLM Configuration and Initialization
# ==========================================================
from huggingface_hub import InferenceClient

@dataclass
class LLMConfig:
    """Configuration for LLM models using HuggingFace InferenceClient"""
    model_type: str = 'huggingface'
    hf_token: str = os.environ.get("HF_TOKEN", "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")
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    mistral_path: str = "USING_HUGGINGFACE_INFERENCE_API"
    n_ctx: int = 2048
    n_batch: int = 128
    n_gpu_layers: int = 20
    temperature: float = 0.3
    max_tokens: int = 200
    use_llm_for_top_n: int = 20


class HFInferenceWrapper:
    """Wrapper for Hugging Face InferenceClient"""
    def __init__(self, client, model_id):
        self.client = client
        self.model_id = model_id

    def generate(self, prompt, **kwargs):
        """Generate text using chat_completion or text_generation"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant analyzing job candidates."},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 200),
                temperature=kwargs.get('temperature', 0.3),
            )
            msg = response.choices[0].message
            return msg.content if hasattr(msg, 'content') else msg.get("content", "")
        except Exception:
            try:
                return self.client.text_generation(
                    prompt,
                    max_new_tokens=kwargs.get('max_tokens', 200),
                    temperature=kwargs.get('temperature', 0.3),
                    return_full_text=False
                )
            except Exception as e:
                print("[ERROR] HF Generation Error:", str(e))
                return ""


def initialize_llm():
    """Initialize Hugging Face inference client"""
    llm_config = LLMConfig()
    print("\n Initializing Hugging Face InferenceClient...")
    try:
        os.environ["HF_TOKEN"] = llm_config.hf_token
        client = InferenceClient(model=llm_config.model_id, token=llm_config.hf_token)
        wrapper = HFInferenceWrapper(client, llm_config.model_id)

        print(f"[INFO] Testing connection to {llm_config.model_id}...")
        test_response = wrapper.generate("Hello! Test connection.", max_tokens=10)

        if test_response:
            print(f"[SUCCESS] Successfully connected to {llm_config.model_id}")
            llm_config.client = wrapper
        else:
            print("[ERROR] Connection test failed")
            llm_config.model_type = "none"
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {str(e)[:120]}")
        llm_config.model_type = "none"
    return llm_config


llm_config = initialize_llm()


# ==========================================================
# CELL 4–10: Hybrid Ranking System Definition
# ==========================================================
class HybridRankingSystem:
    """
    Hybrid Ranking System combining:
    - Embedding Similarity
    - Cross-Encoder Reranking
    - LLM Analysis (via Hugging Face)
    - Self Validation and Confidence Metrics
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.context = {}
        self.data_schema = {}
        self.validation_results = {}
        self.confidence_scores = {}
        self.llm_config = llm_config or LLMConfig()
        self._initialize_models()

    # ---------------- Model Initialization ----------------
    def _initialize_models(self):
        print("\n Initializing Models...")
        print("=" * 60)
        try:
            self.embedder = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
            print("[SUCCESS] BGE embeddings loaded")
        except:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            print("[WARNING] MiniLM embeddings loaded (fallback)")

        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
            print("[SUCCESS] Cross-encoder loaded")
        except:
            self.cross_encoder = None
            print("[WARNING] Cross-encoder not available")

        if hasattr(self.llm_config, "client") and self.llm_config.model_type == "huggingface":
            print(" Using Hugging Face InferenceClient (cloud LLM)")
            self.llm = self.llm_config.client
            self.llm_type = "huggingface"
        else:
            self.llm = None
            self.llm_type = None

        print("=" * 60)
        print(" Model initialization complete\n")

    def understand_context(self, data: Dict) -> Dict:
        """Identify entities and task type from the data"""
        context = {
            'job_title': None,
            'company': None,
            'data_type': None,
            'task_type': 'ranking',
            'target_variable': None,
            'features': [],
            'entities_found': []
        }

        if 'job_description' in data:
            job_desc = data['job_description']
            context['job_title'] = job_desc.get('title', 'Unknown')
            context['company'] = job_desc.get('company', 'Unknown')
            context['entities_found'].append('job_description')

        if 'candidates' in data or 'resumes' in data:
            context['data_type'] = 'resume_ranking'
            context['entities_found'].append('candidates')

        sample_candidate = None
        if 'candidates' in data and len(data['candidates']) > 0:
            sample_candidate = data['candidates'][0]
        elif 'resumes' in data and len(data['resumes']) > 0:
            sample_candidate = data['resumes'][0]

        if sample_candidate:
            if 'match_score' in sample_candidate:
                context['target_variable'] = 'match_score'
            elif 'score' in sample_candidate:
                context['target_variable'] = 'score'
            else:
                context['target_variable'] = 'computed_score'

            context['features'] = list(sample_candidate.keys())

        self.context = context
        print(f"\n Context Understanding Complete:")
        print(f"   • Job Title: {context['job_title']}")
        print(f"   • Task Type: {context['task_type']}")
        print(f"   • Target Variable: {context['target_variable']}")

        return context

    def interpret_data_structure(self, data: Dict) -> Dict:
        """Parse data structure and identify feature-label separation"""
        schema = {
            'num_candidates': 0,
            'score_field': None,
            'feature_fields': [],
            'label_field': None,
            'data_quality': {},
            'field_types': {}
        }

        candidates = data.get('candidates', data.get('resumes', []))
        schema['num_candidates'] = len(candidates)

        if candidates:
            sample = candidates[0]

            for field, value in sample.items():
                if isinstance(value, (int, float)):
                    schema['field_types'][field] = 'numeric'
                    if 'score' in field.lower() or 'match' in field.lower():
                        schema['score_field'] = field
                        schema['label_field'] = field
                elif isinstance(value, list):
                    schema['field_types'][field] = 'list'
                    schema['feature_fields'].append(field)
                elif isinstance(value, str):
                    schema['field_types'][field] = 'text'
                    schema['feature_fields'].append(field)

            schema['data_quality'] = {
                'has_scores': schema['score_field'] is not None,
                'has_features': len(schema['feature_fields']) > 0,
                'completeness': self._check_data_completeness(candidates),
                'consistency': self._check_data_consistency(candidates)
            }

        self.data_schema = schema
        print(f"\n Data Interpretation Complete:")
        print(f"   • Candidates: {schema['num_candidates']}")
        print(f"   • Score Field: {schema['score_field']}")
        print(f"   • Feature Fields: {len(schema['feature_fields'])}")
        print(f"   • Data Quality: {schema['data_quality']['completeness']:.1f}% complete")

        return schema

    def select_ranking_algorithm(self, candidates: List[Dict], job_desc: Dict) -> str:
        """Choose the optimal ranking algorithm based on data characteristics"""

        if self.data_schema.get('score_field'):
            algorithm = 'deterministic_sort'
            print(f"\n Algorithm Selection: {algorithm}")
            print(f"   • Reason: Pre-computed scores found")
        else:
            feature_count = len(self.data_schema.get('feature_fields', []))

            if self.llm:
                algorithm = 'llm_enhanced_scoring'
                print(f"\n Algorithm Selection: {algorithm}")
                print(f"   • Reason: LLM available for enhanced analysis")
            elif feature_count > 10:
                algorithm = 'ml_based_scoring'
                print(f"\n Algorithm Selection: {algorithm}")
                print(f"   • Reason: Rich feature set ({feature_count} features)")
            else:
                algorithm = 'rule_based_scoring'
                print(f"\n Algorithm Selection: {algorithm}")
                print(f"   • Reason: Limited features, using rules")

        return algorithm
    
    def get_llm_analysis(self, candidate: Dict, job_desc: Dict) -> Dict:
        """Get LLM-based analysis and scoring"""
        if not self.llm:
            return {
                'score': 50,
                'strengths': [],
                'weaknesses': [],
                'analysis': 'LLM not available',
                'confidence': 0
            }

        candidate_text = self._create_candidate_text(candidate)[:800]
        job_text = self._create_job_text(job_desc)[:800]
        prompt = self._create_analysis_prompt(candidate_text, job_text)

        try:
            response = self._get_mistral_response(prompt)
            if response:
                return self._parse_llm_response(response)
        except Exception as e:
            print(f"    LLM analysis error: {str(e)[:60]}")

        return {
            'score': 50,
            'strengths': [],
            'weaknesses': [],
            'analysis': 'Analysis failed',
            'confidence': 0
        }

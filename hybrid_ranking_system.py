# app/hybrid_ranking_system.py

import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from app.config.config import LLMConfig
from huggingface_hub import InferenceClient
import json
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
import time
from collections import Counter
import pandas as pd
import gc

# Set device for model loading
device = "cuda" if torch.cuda.is_available() else "cpu"

def clear_memory():
    """Clear memory to prevent OOM errors"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Your HybridRankingSystem class implementation here...
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
        print(f"   - Job Title: {context['job_title']}")
        print(f"   - Task Type: {context['task_type']}")
        print(f"   - Target Variable: {context['target_variable']}")

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
        print(f"   - Candidates: {schema['num_candidates']}")
        print(f"   - Score Field: {schema['score_field']}")
        print(f"   - Feature Fields: {len(schema['feature_fields'])}")
        print(f"   - Data Quality: {schema['data_quality']['completeness']:.1f}% complete")

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
                print(f"   - Reason: LLM available for enhanced analysis")
            elif feature_count > 10:
                algorithm = 'ml_based_scoring'
                print(f"\n Algorithm Selection: {algorithm}")
                print(f"   - Reason: Rich feature set ({feature_count} features)")
            else:
                algorithm = 'rule_based_scoring'
                print(f"\n Algorithm Selection: {algorithm}")
                print(f"   - Reason: Limited features, using rules")

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

    def _get_mistral_response(self, prompt: str) -> str:
        """Handle both Hugging Face and local Mistral inference"""
        if self.llm_type == "huggingface" and hasattr(self.llm, "generate"):
            return self.llm.generate(prompt, max_tokens=self.llm_config.max_tokens)

        elif self.llm_type == "mistral":
            response = self.llm(
                prompt,
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature,
                stop=["</s>", "[INST]"],
                echo=False
            )
            return response['choices'][0]['text'].strip()

        return ""

    def _create_analysis_prompt(self, candidate_text: str, job_text: str) -> str:
        """Create a prompt for LLM analysis"""
        prompt = (
            f"Analyze the following job description and candidate profile. "
            f"Provide a match score (0-100), key strengths, key weaknesses, "
            f"and a short recommendation for the candidate. "
            f"Structure your response as a JSON object with keys: "
            f"'score' (int), 'strengths' (list of str), "
            f"'weaknesses' (list of str), 'recommendation' (str), 'confidence' (int 0-100)."
            f"\n\nJOB DESCRIPTION:\n{job_text}"
            f"\n\nCANDIDATE PROFILE:\n{candidate_text}"
            f"\n\nJSON Output:"
        )
        return prompt

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response into a structured dictionary"""
        try:
            # Try to find a JSON block in the response
            match = re.search(r'```json\n({.*?})\n```', response, re.DOTALL)
            if not match:
                match = re.search(r'({.*?})', response, re.DOTALL) # Fallback to find any JSON
            if match:
                json_str = match.group(1)
                parsed_json = json.loads(json_str)

                # Ensure required keys exist, provide defaults
                return {
                    'score': parsed_json.get('score', 50),
                    'strengths': parsed_json.get('strengths', []),
                    'weaknesses': parsed_json.get('weaknesses', []),
                    'recommendation': parsed_json.get('recommendation', 'No specific recommendation.'),
                    'confidence': parsed_json.get('confidence', 75) # Default confidence
                }
            else:
                # If no JSON found, try to extract information heuristically or return default
                score_match = re.search(r'score:\s*(\d+)', response, re.IGNORECASE)
                score = int(score_match.group(1)) if score_match else 50
                return {
                    'score': score,
                    'strengths': [],
                    'weaknesses': [],
                    'recommendation': response[:100], # Take first 100 chars as recommendation
                    'confidence': 50 # Lower confidence for heuristic parsing
                }
        except json.JSONDecodeError as e:
            print(f"    JSON parsing error: {e}")
            # Fallback for malformed JSON
            score_match = re.search(r'score:\s*(\d+)', response, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 50
            return {
                'score': score,
                'strengths': [],
                'weaknesses': [],
                'recommendation': response[:100],
                'confidence': 50
            }
        except Exception as e:
            print(f"    General parsing error: {e}")
            return {
                'score': 50,
                'strengths': [],
                'weaknesses': [],
                'recommendation': 'Failed to parse LLM response.',
                'confidence': 0
            }
        
    
    def _compute_and_rank(self, candidates: List[Dict], job_desc: Dict) -> List[Dict]:
        """Enhanced scoring with LLM integration"""

        print("\n Computing candidate scores...")

        # First pass: Basic scoring for all candidates
        for i, candidate in enumerate(candidates):
            # Initialize score components
            score_components = {}

            # 1. Skills matching (30% weight)
            if 'skills' in candidate and 'required_skills' in job_desc:
                required = set(job_desc.get('required_skills', []))
                candidate_skills = set(candidate.get('skills', []))
                skill_match = len(required & candidate_skills) / len(required) if required else 0
                score_components['skills'] = skill_match * 100
            else:
                score_components['skills'] = 50

            # 2. Experience scoring (25% weight)
            exp_score = self._score_experience(candidate, job_desc)
            score_components['experience'] = exp_score

            # 3. Education scoring (15% weight)
            edu_score = self._score_education(candidate, job_desc)
            score_components['education'] = edu_score

            # 4. Semantic similarity (20% weight)
            if self.embedder:
                semantic_score = self._compute_semantic_score(candidate, job_desc)
                score_components['semantic'] = semantic_score
            else:
                score_components['semantic'] = 50

            # 5. Cross-encoder reranking (10% weight if available)
            if self.cross_encoder:
                ce_score = self._compute_cross_encoder_score(candidate, job_desc)
                score_components['cross_encoder'] = ce_score
            else:
                score_components['cross_encoder'] = score_components['semantic']

            # Calculate initial weighted score (without LLM)
            weights = {
                'skills': 0.30,
                'experience': 0.25,
                'education': 0.15,
                'semantic': 0.20,
                'cross_encoder': 0.10
            }

            initial_score = sum(score_components[k] * weights[k] for k in weights)

            candidate['score_components'] = score_components
            candidate['initial_score'] = initial_score
            candidate['total_score'] = initial_score  # Will be updated with LLM

            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(candidates)} candidates...")

        # Sort by initial score
        candidates_sorted = sorted(candidates, key=lambda x: x['initial_score'], reverse=True)

        # Second pass: LLM analysis for top N candidates
        if self.llm and self.llm_config.use_llm_for_top_n > 0:
            print(f"\n[AI] Running LLM analysis for top {self.llm_config.use_llm_for_top_n} candidates...")

            for i, candidate in enumerate(candidates_sorted[:self.llm_config.use_llm_for_top_n]):
                print(f"   Analyzing candidate {i + 1}/{min(len(candidates_sorted), self.llm_config.use_llm_for_top_n)}...")

                llm_result = self.get_llm_analysis(candidate, job_desc)
                candidate['llm_analysis'] = llm_result

                # Update total score with LLM component (30% weight for top candidates)
                if llm_result['confidence'] > 0:
                    candidate['total_score'] = (
                        candidate['initial_score'] * 0.7 +
                        llm_result['score'] * 0.3
                    )
                    candidate['score_components']['llm'] = llm_result['score']

                # Add insights
                candidate['ai_insights'] = {
                    'strengths': llm_result.get('strengths', []),
                    'weaknesses': llm_result.get('weaknesses', []),
                    'recommendation': llm_result.get('recommendation', '')
                }

                # Small delay to avoid overloading
                time.sleep(0.1)

                # Clear memory periodically
                if (i + 1) % 5 == 0 and torch.cuda.is_available():
                    clear_memory()

        # Final sort by total score
        ranked = sorted(candidates_sorted, key=lambda x: x['total_score'], reverse=True)

        # Add rank
        for i, candidate in enumerate(ranked):
            candidate['rank'] = i + 1
            candidate['percentile'] = (1 - (i / len(ranked))) * 100

        print(f" Scoring complete for {len(candidates)} candidates")

        return ranked

    def _score_experience(self, candidate: Dict, job_desc: Dict) -> float:
        """Score candidate's experience"""
        score = 50  # Base score

        if 'experience' in candidate:
            exp_text = str(candidate['experience']).lower()

            # Extract years of experience
            years_pattern = re.findall(r'(\d+)\s*(?:years?|yrs?)', exp_text)
            if years_pattern:
                years = int(years_pattern[0])
                required_years = job_desc.get('experience_years', 5)

                if years >= required_years:
                    score = 100
                elif years >= required_years * 0.8:
                    score = 80 + (years / required_years) * 20
                else:
                    score = (years / required_years) * 80

            # Check for leadership experience
            if any(word in exp_text for word in ['lead', 'manager', 'senior', 'principal']):
                score = min(score + 10, 100)

        return score

    def _score_education(self, candidate: Dict, job_desc: Dict) -> float:
        """Score candidate's education"""
        score = 50  # Base score

        if 'education' in candidate:
            edu_text = str(candidate['education']).lower()

            # Degree scoring
            if 'phd' in edu_text or 'doctorate' in edu_text:
                score = 100
            elif 'master' in edu_text or 'msc' in edu_text or 'mba' in edu_text:
                score = 85
            elif 'bachelor' in edu_text or 'bsc' in edu_text or 'ba' in edu_text:
                score = 70

            # Field relevance
            relevant_fields = ['computer science', 'software', 'engineering', 'data science', 'mathematics']
            if any(field in edu_text for field in relevant_fields):
                score = min(score + 15, 100)

        return score

    def _compute_semantic_score(self, candidate: Dict, job_desc: Dict) -> float:
        """Compute semantic similarity score"""
        candidate_text = self._create_candidate_text(candidate)[:1000]
        job_text = self._create_job_text(job_desc)[:1000]

        if not candidate_text or not job_text:
            return 50

        try:
            cand_embedding = self.embedder.encode(candidate_text, convert_to_tensor=True, show_progress_bar=False)
            job_embedding = self.embedder.encode(job_text, convert_to_tensor=True, show_progress_bar=False)
            similarity = float(util.pytorch_cos_sim(cand_embedding, job_embedding)[0][0])
            return similarity * 100
        except:
            return 50

    def _compute_cross_encoder_score(self, candidate: Dict, job_desc: Dict) -> float:
        """Compute cross-encoder reranking score"""
        if not self.cross_encoder:
            return 50

        candidate_text = self._create_candidate_text(candidate)[:256]
        job_text = self._create_job_text(job_desc)[:256]

        try:
            score = self.cross_encoder.predict([[candidate_text, job_text]])[0]
            # Convert to 0-100 scale
            return (score + 1) * 50  # Cross-encoder typically returns [-1, 1]
        except:
            return 50
        

    def validate_results(self, ranked_candidates: List[Dict]) -> Dict:
        """Enhanced validation including LLM consistency checks"""

        validation = {
            'numeric_validation': {},
            'ordering_validation': {},
            'statistical_validation': {},
            'anomaly_detection': {},
            'llm_consistency': {},
            'overall_valid': True
        }

        if not ranked_candidates:
            validation['overall_valid'] = False
            return validation

        scores = [c.get('total_score', 0) for c in ranked_candidates]

        # Standard validations
        validation['numeric_validation'] = {
            'all_in_range': all(0 <= s <= 100 for s in scores),
            'no_nulls': all(s is not None for s in scores),
            'no_negatives': all(s >= 0 for s in scores)
        }

        validation['ordering_validation'] = {
            'strictly_descending': all(scores[i] >= scores[i+1] for i in range(len(scores)-1)),
            'top_score_reasonable': scores[0] <= 100 if scores else True,
            'score_gaps_reasonable': self._check_score_gaps(scores)
        }

        # Statistical validation
        if scores:
            validation['statistical_validation'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'distribution_normal': self._check_distribution(scores),
                'outliers_detected': self._detect_outliers(scores)
            }

        # Anomaly detection
        validation['anomaly_detection'] = {
            'duplicate_scores': len(scores) != len(set(scores)),
            'suspicious_patterns': self._detect_suspicious_patterns(scores),
            'data_integrity': self._check_data_integrity(ranked_candidates)
        }

        # LLM consistency checks (if LLM was used)
        llm_candidates = [c for c in ranked_candidates if 'llm_analysis' in c]
        if llm_candidates:
            llm_scores = [c.get('llm_analysis', {}).get('score', 0) for c in llm_candidates]
            initial_scores = [c.get('initial_score', 0) for c in llm_candidates]

            if len(llm_scores) > 1:
                correlation = np.corrcoef(llm_scores, initial_scores)[0, 1]
                validation['llm_consistency'] = {
                    'correlation': correlation,
                    'correlation_positive': correlation > 0,
                    'significant_correlation': abs(correlation) > 0.3,
                    'llm_used_count': len(llm_candidates)
                }

        validation['overall_valid'] = (
            all(validation['numeric_validation'].values()) and
            validation['ordering_validation']['strictly_descending']
        )

        self.validation_results = validation

        print(f"\n Self-Validation Complete:")
        print(f"   • Numeric Valid: {all(validation['numeric_validation'].values())}")
        print(f"   • Ordering Valid: {validation['ordering_validation']['strictly_descending']}")
        print(f"   • No Anomalies: {not validation['anomaly_detection'].get('suspicious_patterns', False)}")
        print(f"   • Overall Valid: {validation['overall_valid']}")

        return validation

    def semantic_verification(self, ranked_candidates: List[Dict], job_desc: Dict) -> Dict:
        """Verify that top candidates semantically match the job requirements"""
        verification = {
            'skill_alignment': [],
            'context_relevance': [],
            'semantic_scores': [],
            'verification_passed': True
        }

        job_title = job_desc.get('title', '')
        required_skills = job_desc.get('required_skills', [])
        job_text = self._create_job_text(job_desc)

        for i, candidate in enumerate(ranked_candidates[:10]):
            candidate_text = self._create_candidate_text(candidate)
            candidate_skills = candidate.get('skills', [])

            skill_match = len(set(candidate_skills) & set(required_skills))
            skill_alignment = skill_match / len(required_skills) if required_skills else 0
            verification['skill_alignment'].append(skill_alignment)

            if self.embedder and job_text and candidate_text:
                job_embedding = self.embedder.encode(job_text, convert_to_tensor=True, show_progress_bar=False)
                candidate_embedding = self.embedder.encode(candidate_text, convert_to_tensor=True, show_progress_bar=False)
                semantic_score = float(util.pytorch_cos_sim(job_embedding, candidate_embedding)[0][0])
                verification['semantic_scores'].append(semantic_score)

            context_relevant = self._check_context_relevance(candidate, job_title)
            verification['context_relevance'].append(context_relevant)

        avg_skill_alignment = np.mean(verification['skill_alignment']) if verification['skill_alignment'] else 0
        avg_semantic_score = np.mean(verification['semantic_scores']) if verification['semantic_scores'] else 0

        verification['verification_passed'] = (
            avg_skill_alignment > 0.3 or avg_semantic_score > 0.5
        )

        print(f"\n Semantic Verification Complete:")
        print(f"   • Avg Skill Alignment: {avg_skill_alignment:.2%}")
        print(f"   • Avg Semantic Score: {avg_semantic_score:.2f}")
        print(f"   • Verification Passed: {verification['verification_passed']}")

        return verification
    
    def calculate_confidence(self, validation: Dict, verification: Dict) -> float:
        """Calculate overall confidence in the ranking results"""
        confidence_components = []

        if validation.get('numeric_validation'):
            numeric_conf = sum(validation['numeric_validation'].values()) / len(validation['numeric_validation'])
            confidence_components.append(('numeric', numeric_conf, 0.35))

        if validation.get('ordering_validation'):
            ordering_conf = 1.0 if validation['ordering_validation']['strictly_descending'] else 0.0
            confidence_components.append(('ordering', ordering_conf, 0.25))

        if verification.get('verification_passed'):
            semantic_conf = 1.0 if verification['verification_passed'] else 0.5
            confidence_components.append(('semantic', semantic_conf, 0.20))

        if self.data_schema.get('data_quality'):
            quality_conf = self.data_schema['data_quality']['completeness'] / 100
            confidence_components.append(('quality', quality_conf, 0.10))

        if self.llm and validation.get('llm_consistency'):
            llm_conf = 1.0 if validation['llm_consistency'].get('correlation_positive', False) else 0.5
            confidence_components.append(('llm', llm_conf, 0.10))

        total_confidence = sum(conf * weight for _, conf, weight in confidence_components)

        if total_confidence >= 0.9:
            confidence_level = "VERY HIGH (>90%)"
        elif total_confidence >= 0.75:
            confidence_level = "HIGH (75-90%)"
        elif total_confidence >= 0.6:
            confidence_level = "MODERATE (60-75%)"
        else:
            confidence_level = "LOW (<60%)"

        self.confidence_scores = {
            'overall_confidence': total_confidence,
            'confidence_level': confidence_level,
            'components': confidence_components
        }

        print(f"\n Confidence Calculation Complete:")
        print(f"   • Overall Confidence: {total_confidence:.2%}")
        print(f"   • Confidence Level: {confidence_level}")

        return total_confidence

    def _check_data_completeness(self, candidates: List[Dict]) -> float:
        """Check percentage of non-null fields"""
        if not candidates:
            return 0

        total_fields = 0
        non_null_fields = 0

        for candidate in candidates:
            for value in candidate.values():
                total_fields += 1
                if value is not None and value != "":
                    non_null_fields += 1

        return (non_null_fields / total_fields * 100) if total_fields > 0 else 0

    def _check_data_consistency(self, candidates: List[Dict]) -> bool:
        """Check if all candidates have same fields"""
        if not candidates:
            return True

        first_keys = set(candidates[0].keys())
        return all(set(c.keys()) == first_keys for c in candidates)

    def _check_score_gaps(self, scores: List[float]) -> bool:
        """Check if score gaps are reasonable"""
        if len(scores) < 2:
            return True

        gaps = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        max_gap = max(gaps) if gaps else 0
        score_range = max(scores) - min(scores) if scores else 1

        return max_gap <= score_range * 0.5

    def _check_distribution(self, scores: List[float]) -> str:
        """Check score distribution pattern"""
        if not scores:
            return "unknown"

        mean = np.mean(scores)
        std = np.std(scores)

        if std < 5:
            return "tight"
        elif std < 15:
            return "normal"
        else:
            return "wide"

    def _detect_outliers(self, scores: List[float]) -> List[int]:
        """Detect outlier indices using IQR method"""
        if len(scores) < 4:
            return []

        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [i for i, s in enumerate(scores)
                   if s < lower_bound or s > upper_bound]

        return outliers

    def _detect_suspicious_patterns(self, scores: List[float]) -> bool:
        """Detect suspicious scoring patterns"""
        if not scores:
            return False

        score_counts = Counter(scores)
        max_count = max(score_counts.values())

        if max_count > len(scores) * 0.3:
            return True

        if all(s % 10 == 0 for s in scores):
            return True

        return False

    def _check_data_integrity(self, candidates: List[Dict]) -> bool:
        """Check overall data integrity"""
        if not candidates:
            return False

        required_fields = ['name', 'email']
        for candidate in candidates[:10]:
            if not all(field in candidate for field in required_fields if field in candidate):
                return False

        return True

    def _create_job_text(self, job_desc: Dict) -> str:
        """Create text representation of job"""
        sections = []
        for field in ['title', 'description', 'requirements', 'required_skills']:
            if field in job_desc:
                value = job_desc[field]
                if isinstance(value, list):
                    sections.append(' '.join(map(str, value)))
                else:
                    sections.append(str(value))
        return ' '.join(sections)[:1000]

    def _create_candidate_text(self, candidate: Dict) -> str:
        """Create text representation of candidate"""
        sections = []
        for field in ['summary', 'experience', 'skills', 'education']:
            if field in candidate:
                value = candidate[field]
                if isinstance(value, list):
                    sections.append(' '.join(map(str, value)))
                else:
                    sections.append(str(value))
        return ' '.join(sections)[:1000]

    def _check_context_relevance(self, candidate: Dict, job_title: str) -> bool:
        """Check if candidate is relevant to job context"""
        job_keywords = job_title.lower().split()
        candidate_text = self._create_candidate_text(candidate).lower()

        matches = sum(1 for keyword in job_keywords if keyword in candidate_text)
        return matches > 0
    

    def rank_candidates(self, data: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Complete ranking pipeline with LLM enhancement"""

        print("="*60)
        print(" HYBRID RANKING SYSTEM WITH LLM - MULTI-PHASE VALIDATION")
        print("="*60)

        # Display LLM status
        if self.llm:
            print(f" LLM Mode: {self.llm_type.upper()}")
            print(f"   - Will analyze top {self.llm_config.use_llm_for_top_n} candidates")
        else:
            print(" LLM Mode: DISABLED (using embeddings only)")

        # Step A: Context Understanding
        context = self.understand_context(data)

        # Step B: Data Interpretation
        schema = self.interpret_data_structure(data)

        # Extract data
        candidates = data.get('candidates', data.get('resumes', []))
        job_desc = data.get('job_description', {})

        # Step C: Algorithmic Decision Making
        algorithm = self.select_ranking_algorithm(candidates, job_desc)

        # Step D: Perform ranking with LLM enhancement
        if algorithm == 'deterministic_sort' and schema['score_field']:
            # Use existing scores
            ranked_candidates = sorted(
                candidates,
                key=lambda x: x.get(schema['score_field'], 0),
                reverse=True
            )
            for i, c in enumerate(ranked_candidates):
                c['rank'] = i + 1
                c['total_score'] = c.get(schema['score_field'], 0)
        else:
            # Compute scores with LLM
            ranked_candidates = self._compute_and_rank(candidates, job_desc)

        # Step E: Self-Validation
        validation = self.validate_results(ranked_candidates)

        # Step F: Semantic Cross-Verification
        verification = self.semantic_verification(ranked_candidates, job_desc)

        # Step G: Confidence Weighting
        confidence = self.calculate_confidence(validation, verification)

        # Create DataFrame
        df_results = pd.DataFrame(ranked_candidates)

        # Add metadata
        metadata = {
            'context': context,
            'schema': schema,
            'algorithm_used': algorithm,
            'validation': validation,
            'verification': verification,
            'confidence': self.confidence_scores,
            'llm_status': {
                'enabled': self.llm is not None,
                'type': self.llm_type if self.llm else 'none',
                'candidates_analyzed': len([c for c in ranked_candidates if 'llm_analysis' in c])
            }
        }

        print("\n" + "="*60)
        print(" RANKING COMPLETE WITH FULL VALIDATION")
        print("="*60)

        return df_results, metadata
    
 
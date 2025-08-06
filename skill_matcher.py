from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Load the model globally so it's reused across all comparisons
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess(text):
    """Clean up input for embedding."""
    return re.sub(r'\s+', ' ', text.strip().lower())

def score_resume_against_job_keywords(resume_phrases, job_keywords, threshold=0.6):
    """
    Compares resume phrases against job keywords using semantic similarity.

    Args:
        resume_phrases (List[str]): words/phrases from the resume
        job_keywords (List[str]): required skills from the job posting
        threshold (float): similarity cutoff for considering a match

    Returns:
        Tuple[float, List[Dict]]: percentage match score, list of matched pairs
    """
    clean_resume = [preprocess(p) for p in resume_phrases]
    clean_keywords = [preprocess(k) for k in job_keywords]

    resume_vectors = model.encode(clean_resume)
    keyword_vectors = model.encode(clean_keywords)

    matched = 0
    matches = []

    for i, kw_vec in enumerate(keyword_vectors):
        sims = cosine_similarity([kw_vec], resume_vectors)[0]
        max_sim_idx = np.argmax(sims)
        max_sim = sims[max_sim_idx]

        if max_sim >= threshold:
            matched += 1
            matches.append({
                "keyword": job_keywords[i],
                "matched_phrase": resume_phrases[max_sim_idx],
                "similarity": round(float(max_sim), 2)
            })

    score = round((matched / len(job_keywords)) * 100, 1)
    return score, matches

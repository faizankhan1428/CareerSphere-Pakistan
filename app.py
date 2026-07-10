"""
CareerSphere Pakistan - AI Job Recommendation System
Production-ready Flask application with TF-IDF + Cosine Similarity algorithmic pipeline
Engineered by Muhammad Faizan
"""

import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model components
df = None
vectorizer = None
tfidf_matrix = None

# Premium companies and sectors for fallback sorting
PREMIUM_COMPANIES = [
    'Bank Alfalah', 'JS Bank', 'Habib Metropolitan Bank', 'Meezan Bank',
    'Telenor Pakistan', 'Jazz', 'Zong', 'PTCL', 'Nayatel',
    'Unilever Pakistan', 'P&G Pakistan', 'Nestlé Pakistan', 'Engro Corporation',
    '10Pearls', 'Tkxel', 'Systems Limited', 'NetSol Technologies', 'TRG Pakistan'
]

PREMIUM_SECTORS = [
    'IT & Technology', 'Banking & Finance', 'Telecom', 'FMCG & Consumer Goods',
    'Healthcare', 'Education', 'Engineering', 'Consulting'
]

def load_and_preprocess_data():
    """
    Load dataset and perform feature engineering with TF-IDF vectorization
    """
    global df, vectorizer, tfidf_matrix
    
    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), 'pakistan_job_market_dataset.csv')
    df = pd.read_csv(dataset_path)
    
    # Feature Engineering: Combine Job Title and Required Skills
    df['combined_features'] = df['Job Title'].fillna('') + " " + df['Required Skills'].fillna('')
    
    # Text Normalization: Convert to lowercase and clean whitespace
    df['combined_features'] = df['combined_features'].str.lower().str.strip()
    
    # Initialize TF-IDF Vectorizer with specified parameters
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        token_pattern=r'(?u)\b\w+\b'
    )
    
    # Fit and transform the combined features
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    
    print(f"✓ CareerSphere Pakistan: Dataset loaded with {len(df)} job listings")
    print(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")

def get_recommendations(user_query, top_n=10):
    """
    Generate job recommendations using Cosine Similarity
    
    Args:
        user_query (str): User's skills/job preferences
        top_n (int): Number of recommendations to return
    
    Returns:
        list: Top N job recommendations with metadata
    """
    global df, vectorizer, tfidf_matrix
    
    # Preprocess user query - handle comma-separated inputs
    query_processed = user_query.lower().strip().replace(',', ' ')
    
    # Transform user query using the fitted vectorizer
    query_vector = vectorizer.transform([query_processed])
    
    # Calculate cosine similarity between query and all job listings
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get indices of top N similar jobs
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    # Prepare recommendations with similarity scores
    recommendations = []
    for idx in top_indices:
        job_data = df.iloc[idx].to_dict()
        job_data['similarity_score'] = float(similarity_scores[idx])
        recommendations.append(job_data)
    
    # Fallback: If max similarity is 0.0, sort by premium companies/sectors
    max_similarity = max(similarity_scores) if len(similarity_scores) > 0 else 0.0
    if max_similarity == 0.0:
        df['premium_score'] = df.apply(
            lambda row: (
                (2 if row['Company'] in PREMIUM_COMPANIES else 0) +
                (1 if row['Sector'] in PREMIUM_SECTORS else 0)
            ),
            axis=1
        )
        fallback_indices = df['premium_score'].argsort()[-top_n:][::-1]
        recommendations = []
        for idx in fallback_indices:
            job_data = df.iloc[idx].to_dict()
            job_data['similarity_score'] = 0.01  # Small non-zero score for fallback
            recommendations.append(job_data)
    
    return recommendations

@app.route('/')
def index():
    """Render the main application interface"""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    API endpoint to receive user query and return job recommendations
    
    Expected JSON payload:
    {
        "skills": "python machine learning data science",
        "top_n": 10
    }
    """
    try:
        data = request.get_json()
        user_skills = data.get('skills', '')
        top_n = int(data.get('top_n', 10))
        
        if not user_skills:
            return jsonify({'error': 'Please provide your skills or job preferences'}), 400
        
        # Get recommendations
        recommendations = get_recommendations(user_skills, top_n)
        
        # Format response with relevant job details
        formatted_recommendations = []
        for job in recommendations:
            formatted_recommendations.append({
                'job_title': job.get('Job Title', ''),
                'company': job.get('Company', ''),
                'city': job.get('City', ''),
                'sector': job.get('Sector', ''),
                'salary_range': job.get('Salary Range (PKR)', ''),
                'required_skills': job.get('Required Skills', ''),
                'experience_required': job.get('Experience Required', ''),
                'job_type': job.get('Job Type', ''),
                'education_level': job.get('Education Level', ''),
                'similarity_score': round(job.get('similarity_score', 0) * 100, 2)
            })
        
        return jsonify({
            'success': True,
            'recommendations': formatted_recommendations,
            'total_results': len(formatted_recommendations)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Return dataset statistics for the frontend"""
    global df
    
    if df is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    stats = {
        'total_jobs': len(df),
        'unique_companies': df['Company'].nunique(),
        'unique_cities': df['City'].nunique(),
        'unique_sectors': df['Sector'].nunique(),
        'job_types': df['Job Type'].value_counts().to_dict(),
        'top_sectors': df['Sector'].value_counts().head(5).to_dict()
    }
    
    return jsonify(stats)

# Initialize data within Flask application context for Flask 3.x compatibility
with app.app_context():
    load_and_preprocess_data()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

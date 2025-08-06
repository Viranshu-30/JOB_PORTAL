# Candidate Recommendation Engine

## Project Overview

A production-ready AI-powered candidate recommendation system that matches job descriptions with candidate resumes using semantic similarity and machine learning. This system implements the core requirements specified in the SproutsAI assignment while providing a complete recruitment platform architecture.

## Assignment Requirements Fulfillment

The system addresses all specified requirements:

- **Job Description Input**: Accepts detailed job descriptions through web interface
- **Resume Processing**: Handles multiple resume formats (PDF, DOCX, TXT) via file upload
- **Embedding Generation**: Utilizes sentence-transformers library for semantic embeddings
- **Cosine Similarity Computation**: Implements sklearn-based similarity calculations
- **Top Candidate Display**: Presents ranked candidates with similarity scores
- **AI-Generated Summaries**: Integrates OpenAI GPT-3.5 for candidate fit analysis (Bonus requirement)

  
## Demonstration

**Login Credentials**: admin / admin123 (HR Dashboard)

## Technical Architecture

### Core Matching Algorithm

The recommendation engine (`skill_matcher.py`) implements semantic similarity using transformer-based embeddings:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def score_resume_against_job_keywords(resume_phrases, job_keywords, threshold=0.6):
    # Semantic embedding generation
    # Cosine similarity calculation
    # Threshold-based matching
    return similarity_score, matched_keywords
```

### Technology Stack

**Backend Framework**: Flask 2.x with Python 3.10
**Database**: MongoDB for document storage and scalability
**Machine Learning**: 
- sentence-transformers (all-MiniLM-L6-v2 model)
- scikit-learn for similarity calculations
**AI Integration**: OpenAI GPT-3.5-turbo for candidate analysis
**Document Processing**: PyPDF2, python-docx for multi-format support
**Deployment**: Waitress WSGI server

### System Components

- **Matching Engine**: Semantic similarity computation with configurable thresholds
- **Document Parser**: Multi-format resume text extraction
- **AI Analyzer**: Context-aware candidate summaries
- **Web Interface**: Bootstrap-based responsive UI
- **Data Layer**: MongoDB document storage with indexing

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- MongoDB instance (cloud)
- OpenAI API key

### Environment Configuration

Create a `.env` file with required variables:

```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/job_portal
OPENAI_API_KEY=sk-your-openai-api-key
```

### Installation Steps

```bash
git clone [repository-url]
cd candidate-recommendation-engine
pip install -r requirements.txt
python app.py
```

### Dependencies

```
Flask
pymongo[srv]
scikit-learn
numpy
PyPDF2
python-docx
Werkzeug
openai==0.28.1
requests
python-dotenv
sentence-transformers
waitress
```

## Application Usage

### For Recruiters

1. **Authentication**: Login with HR credentials (admin/admin123)
2. **Job Posting**: Create job descriptions with detailed requirements
3. **Application Review**: View candidates ranked by AI-computed similarity scores
4. **AI Analysis**: Generate contextual summaries for candidate evaluation
5. **Data Export**: Export candidate data in CSV format

### For Candidates

1. **Job Discovery**: Browse available positions
2. **Application Submission**: Upload resumes in supported formats
3. **Automatic Processing**: System extracts and analyzes resume content
4. **Status Tracking**: View application status and timestamps

## Algorithm Details

### Semantic Matching Process

1. **Text Preprocessing**: Clean and normalize job requirements and resume content
2. **Embedding Generation**: Convert text to high-dimensional vectors using transformer models
3. **Similarity Computation**: Calculate cosine similarity between job and resume embeddings
4. **Score Normalization**: Convert similarity values to percentage-based scores
5. **Ranking Algorithm**: Sort candidates by relevance score with tie-breaking logic

### Performance Characteristics

- **Processing Speed**: Sub-second similarity computation for typical resume lengths
- **Accuracy**: Semantic understanding beyond keyword matching
- **Scalability**: Batch processing capability for large candidate pools
- **Reliability**: Error handling for malformed documents and API failures

## System Architecture

```
├── app.py                     # Main Flask application
├── skill_matcher.py           # Core recommendation algorithm
├── requirements.txt           # Python dependencies
├── runtime.txt               # Python version specification
├── templates/                # HTML templates
│   ├── enhanced_hr_dashboard.html
│   ├── job_applicants.html
│   ├── candidate_dashboard.html
│   └── apply_job.html
└── uploads/                  # Resume file storage
```

## API Endpoints

- `GET /` - Landing page
- `GET /candidate` - Job listings for candidates
- `POST /apply/<job_id>` - Resume submission
- `GET /hr` - HR dashboard with candidate rankings
- `GET /api/generate_summary/<app_id>` - AI summary generation

## Deployment

The application is configured for production deployment with:

- **WSGI Server**: Waitress for production-grade serving
- **Environment Management**: python-dotenv for configuration
- **Error Handling**: Comprehensive exception management
- **Security**: Input validation and file upload restrictions
- **Monitoring**: Logging and performance tracking capabilities

## Technical Decisions

**Semantic Similarity over Keyword Matching**: Provides better understanding of candidate-job fit through contextual analysis rather than simple term frequency.

**MongoDB Document Storage**: Optimized for unstructured resume data and rapid scaling requirements typical in recruitment platforms.

**Transformer-based Embeddings**: all-MiniLM-L6-v2 model provides optimal balance between accuracy and computational efficiency.

**Flask Framework**: Lightweight architecture suitable for MVP development with clear extension paths for enterprise features.


## Future Enhancements

- REST API for programmatic access
- Advanced filtering capabilities (experience level, location, skills)
- Integration with external job boards and ATS systems
- Enhanced ML models for improved matching accuracy
- Real-time candidate scoring updates

---


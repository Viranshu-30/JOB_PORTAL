from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, Response, session
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
from functools import wraps
import pymongo
import os
from dotenv import load_dotenv
import re
import PyPDF2
import docx
import openai
import csv
from io import StringIO
from datetime import datetime
from skill_matcher import score_resume_against_job_keywords

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace for production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
  # Debug: Check if secret key is loaded
app.config['MONGODB_URI'] = os.getenv('MONGODB_URI')
#app.config['DEBUG'] = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# ✅ OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
print(os.getenv("OPENAI_API_KEY"))  # Debug: Check if key is loaded
# Sample HR credentials (replace with DB & hashing in production)
HR_CREDENTIALS = {"admin": "admin123"}

# MongoDB
client = pymongo.MongoClient("" + os.getenv("MONGODB_URI"))
db = client["job_portal"]
jobs_collection = db["jobs"]
applications_collection = db["applications"]

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# --- Auth Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('hr_logged_in'):
            flash("Login required", "warning")
            return redirect(url_for('hr_login'))
        return f(*args, **kwargs)
    return decorated_function

# --- HR Login Routes ---
@app.route('/hr/login', methods=['GET', 'POST'])
def hr_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if HR_CREDENTIALS.get(username) == password:
            session['hr_logged_in'] = True
            flash("Logged in successfully!", "success")
            return redirect(url_for('hr_dashboard'))
        else:
            flash("Invalid username or password", "danger")
    return render_template('hr_login.html')

@app.route('/hr/logout')
def hr_logout():
    session.pop('hr_logged_in', None)
    return redirect(url_for('hr_login'))


# --- Utility Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path, filename):
    try:
        if filename.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        elif filename.endswith('.docx'):
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        elif filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Text extraction error: {e}")
        return ""

def extract_resume_phrases(text):
    tokens = re.findall(r'[\w\-\.\+]+', text)
    return list(set(tokens))

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/candidate')
def candidate_dashboard():
    jobs = list(jobs_collection.find({"status": "Active"}).sort("posted_date", -1))
    for job in jobs:
        job['_id'] = str(job['_id'])
    return render_template('candidate_dashboard.html', jobs=jobs)

@app.route('/apply/<job_id>', methods=['GET', 'POST'])
def apply_job(job_id):
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        flash("Job not found", 'error')
        return redirect(url_for('candidate_dashboard'))

    if job.get("status", "Active") != "Active":
        flash("This job is no longer accepting applications.", "warning")
        return redirect(url_for('candidate_dashboard'))

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email', '').strip().lower()
        phone = request.form.get('phone')
        file = request.files.get('resume')

        # Check for duplicates with normalized lowercase email
        existing_application = applications_collection.find_one({
            "job_id": ObjectId(job_id),
            "email": email
        })

        if existing_application:
            flash("You have already applied to this job using this email.", "warning")
            return redirect(url_for('candidate_dashboard'))

        if not file or file.filename == '':
            flash('Resume file is required', 'error')
            return redirect(request.url)

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename = timestamp + filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            resume_text = extract_text_from_file(file_path, filename)

            application = {
                "job_id": ObjectId(job_id),
                "job_title": job['title'],
                "name": name,
                "email": email,  
                "phone": phone,
                "resume_filename": filename,
                "resume_text": resume_text,
                "applied_date": datetime.now(),
                "status": "Applied"
            }

            applications_collection.insert_one(application)
            flash('Application submitted successfully!', 'success')
            return redirect(url_for('candidate_dashboard'))
        else:
            flash('Invalid file type.', 'error')

    job['_id'] = str(job['_id'])
    return render_template('apply_job.html', job=job)


@app.route('/hr')
@login_required
def hr_dashboard():
    jobs = list(jobs_collection.find().sort("posted_date", -1))
    for job in jobs:
        job['_id'] = str(job['_id'])
        job['application_count'] = applications_collection.count_documents({"job_id": ObjectId(job['_id'])})

        applications = list(applications_collection.find({"job_id": ObjectId(job['_id'])}))
        job_keywords = [kw.strip() for kw in job.get('requirements', '').split(',') if kw.strip()]
        top_candidates = []
        for app in applications:
            resume_phrases = extract_resume_phrases(app.get('resume_text', ''))
            score, _ = score_resume_against_job_keywords(resume_phrases, job_keywords)
            app['similarity_score'] = round(score / 100, 2)
            app['match_percentage'] = score
            app['_id'] = str(app['_id'])
            app['job_id'] = str(app['job_id'])
            top_candidates.append(app)

        top_candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        job['top_applicants'] = top_candidates[:5]
    return render_template('enhanced_hr_dashboard.html', jobs=jobs)

@app.route('/hr/job/<job_id>/applicants')
@login_required
def view_job_applicants(job_id):
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        flash("Job not found", 'error')
        return redirect(url_for('hr_dashboard'))

    sort = request.args.get('sort', 'match')

    applications = list(applications_collection.find({"job_id": ObjectId(job_id)}))
    job_keywords = [kw.strip() for kw in job.get('requirements', '').split(',') if kw.strip()]

    for app in applications:
        resume_phrases = extract_resume_phrases(app.get('resume_text', ''))
        score, _ = score_resume_against_job_keywords(resume_phrases, job_keywords)
        app['similarity_score'] = round(score / 100, 2)
        app['match_percentage'] = score
        app['_id'] = str(app['_id'])
        app['job_id'] = str(app['job_id'])

    # Sort applicants
    if sort == 'recent':
        applications.sort(key=lambda x: x['applied_date'], reverse=True)
    else:  # default to best match
        applications.sort(key=lambda x: x['similarity_score'], reverse=True)

    job['_id'] = str(job['_id'])
    return render_template('job_applicants.html', job=job, applicants=applications, sort=sort)


@app.route('/api/generate_summary/<app_id>')
@login_required
def generate_summary_api(app_id):
    try:
        print(f"[DEBUG] Generating summary for application: {app_id}")

        application = applications_collection.find_one({"_id": ObjectId(app_id)})
        if not application:
            return jsonify({"error": "Application not found"}), 404

        job = jobs_collection.find_one({"_id": application["job_id"]})
        if not job:
            return jsonify({"error": "Job not found"}), 404

        keywords = job.get("requirements", "")
        resume_text = application.get("resume_text", "")
        similarity_score = application.get("similarity_score", 0)

        prompt = f"""You're an expert recruiter reviewing resumes for this role.

Given the required skills:
{keywords}

And this candidate's resume:
{resume_text[:1000]}

With a similarity score of {similarity_score:.1%}, write a short 2-line summary:
- Mention if the candidate is a strong, fair, or weak fit
- Highlight key relevant skills only
"""

        print("[DEBUG] Sending request to OpenAI...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3
        )
        summary = response.choices[0].message.content.strip()
        print("[DEBUG] Summary received from OpenAI.")
        return jsonify({"summary": summary})
    except Exception as e:
        print(f"[ERROR] OpenAI summary generation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/hr/post_job', methods=['GET', 'POST'])
@login_required
def post_job():
    if request.method == 'POST':
        try:
            job = {
                "title": request.form.get('title'),
                "company": request.form.get('company'),
                "location": request.form.get('location'),
                "description": request.form.get('description'),
                "requirements": request.form.get('requirements'),
                "salary_range": request.form.get('salary_range'),
                "posted_date": datetime.now(),
                "status": "Active"
            }
            jobs_collection.insert_one(job)
            flash('Job posted successfully!', 'success')
            return redirect(url_for('hr_dashboard'))
        except Exception as e:
            flash(f"Error posting job: {e}", 'error')
    return render_template('post_job.html')

@app.route('/hr/close_job/<job_id>', methods=['POST'])
@login_required
def close_job(job_id):
    jobs_collection.update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {"status": "Closed"}}
    )
    flash("Job has been marked as Closed.", "info")
    return redirect(url_for('hr_dashboard'))

@app.route('/hr/reopen_job/<job_id>', methods=['POST'])
@login_required
def reopen_job(job_id):
    jobs_collection.update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {"status": "Active"}}
    )
    flash("Job has been reopened.", "success")
    return redirect(url_for('hr_dashboard'))

@app.route('/hr/export_csv/<job_id>')
@login_required
def export_applicants_csv(job_id):
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        return "Job not found", 404

    applications = list(applications_collection.find({"job_id": ObjectId(job_id)}))
    job_keywords = [kw.strip() for kw in job.get('requirements', '').split(',') if kw.strip()]

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Email", "Match Score", "Job Title"])

    for app in applications:
        resume_phrases = extract_resume_phrases(app.get("resume_text", ""))
        score, _ = score_resume_against_job_keywords(resume_phrases, job_keywords)
        writer.writerow([
            app.get("name", ""),
            app.get("email", ""),
            f"{score:.1f}%",
            job.get("title", "")
        ])

    output.seek(0)
    return Response(output, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment;filename=applicants_{job_id}.csv"})

@app.route('/hr/resume/<filename>')
@login_required
def view_resume(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        flash(f"Error loading resume: {e}", 'error')
        return redirect(url_for('hr_dashboard'))

if __name__ == "__main__":
    
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)

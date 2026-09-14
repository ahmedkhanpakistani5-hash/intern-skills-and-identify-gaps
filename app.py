import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Optional Groq AI layer
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

st.set_page_config(
    page_title="Intern Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Dark Green UI
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #061a12 0%, #09291d 50%, #03110c 100%);
        color: #ecfdf5;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: #05150e;
        border-right: 1px solid #164e3a;
    }

    h1, h2, h3 {
        color: #86efac !important;
    }

    p, label, .stMarkdown {
        color: #d1fae5 !important;
    }

    .hero {
        padding: 28px;
        border-radius: 20px;
        background: linear-gradient(135deg, #0b3b28, #062418);
        border: 1px solid #166534;
        margin-bottom: 24px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    }

    .hero h1 {
        margin-bottom: 6px;
        font-size: 2.4rem;
    }

    .hero p {
        font-size: 1.05rem;
        color: #bbf7d0 !important;
    }

    div[data-testid="stMetric"] {
        background: #09251a;
        border: 1px solid #166534;
        padding: 12px;
        border-radius: 14px;
    }

    .gap-card {
        padding: 14px 18px;
        border-radius: 12px;
        background: #0a2d20;
        border-left: 4px solid #22c55e;
        margin: 8px 0;
    }

    .stButton > button {
        background: linear-gradient(90deg, #15803d, #16a34a);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 18px;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #16a34a, #22c55e);
        color: white;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox div {
        background-color: #071f15 !important;
        color: #ecfdf5 !important;
        border-color: #166534 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #09251a;
        border-radius: 8px;
        color: #bbf7d0;
    }

    .stTabs [aria-selected="true"] {
        background: #166534 !important;
        color: white !important;
    }

    .small-note {
        color: #86efac;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9+#.\-/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def find_column(df, candidates):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None

def prepare_interns(df):
    id_col = find_column(df, ["intern_id", "id", "employee_id"])
    name_col = find_column(df, ["name", "intern_name", "student_name"])
    skills_col = find_column(df, ["skills", "skill", "intern_skills", "technical_skills"])

    if skills_col is None:
        # Support multiple skill columns such as Python, SQL, Git, etc.
        excluded = {x for x in [id_col, name_col] if x is not None}
        possible = [c for c in df.columns if c not in excluded]
        if possible:
            df = df.copy()
            df["skills"] = df[possible].astype(str).agg(" ".join, axis=1)
            skills_col = "skills"
        else:
            raise ValueError("Intern CSV needs a 'skills' column.")

    result = pd.DataFrame()
    result["intern_id"] = (
        df[id_col].astype(str) if id_col else pd.Series(range(1, len(df) + 1)).astype(str)
    )
    result["name"] = (
        df[name_col].astype(str) if name_col else result["intern_id"].map(lambda x: f"Intern {x}")
    )
    result["skills"] = df[skills_col].fillna("").astype(str)
    result["clean_skills"] = result["skills"].map(clean_text)
    return result

def prepare_jobs(df):
    id_col = find_column(df, ["job_id", "id", "jobid"])
    title_col = find_column(df, ["title", "job_title", "role", "position"])
    desc_col = find_column(df, ["description", "job_description", "job_desc", "requirements", "skills"])

    if desc_col is None:
        raise ValueError("Job CSV needs a 'description' or 'requirements' column.")

    result = pd.DataFrame()
    result["job_id"] = (
        df[id_col].astype(str) if id_col else pd.Series(range(1, len(df) + 1)).astype(str)
    )
    result["title"] = (
        df[title_col].astype(str) if title_col else result["job_id"].map(lambda x: f"Job {x}")
    )
    result["description"] = df[desc_col].fillna("").astype(str)
    result["clean_description"] = result["description"].map(clean_text)
    return result

def get_top_terms(vectorizer, matrix, n=20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    order = scores.argsort()[::-1]
    return [(feature_names[i], float(scores[i])) for i in order[:n]]

def analyze_jobs(jobs, n_clusters):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(jobs["clean_description"])

    actual_clusters = max(1, min(n_clusters, len(jobs)))
    if len(jobs) == 1:
        labels = np.array([0])
    else:
        model = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)

    jobs = jobs.copy()
    jobs["cluster"] = labels
    return vectorizer, matrix, jobs

def extract_skill_gaps(intern_skills, job_text, vectorizer, job_vector):
    intern_clean = clean_text(intern_skills)
    intern_terms = set(intern_clean.split())

    job_scores = np.asarray(job_vector.toarray()).ravel()
    features = np.array(vectorizer.get_feature_names_out())

    # Strong TF-IDF terms in the selected job
    order = job_scores.argsort()[::-1]
    candidates = []
    for idx in order:
        term = features[idx]
        if len(term) < 2:
            continue
        # Treat multi-word terms and single tokens as skill candidates.
        parts = set(term.split())
        if term not in intern_clean and not parts.issubset(intern_terms):
            candidates.append((term, float(job_scores[idx])))
        if len(candidates) >= 12:
            break

    # Remove terms that are obviously generic
    generic = {
        "experience", "work", "working", "team", "skills", "skill",
        "ability", "strong", "knowledge", "role", "job", "using"
    }
    candidates = [(x, s) for x, s in candidates if x not in generic]
    return candidates

def groq_recommendation(intern_name, intern_skills, job_title, gaps):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not GROQ_AVAILABLE:
        return None

    gap_text = ", ".join([g for g, _ in gaps]) if gaps else "No major gap detected"

    prompt = f"""
You are an AI career coach and internship skills analyst.

Intern: {intern_name}
Current skills: {intern_skills}
Target role: {job_title}
Potential skill gaps from TF-IDF analysis: {gap_text}

Give a practical training plan.
Return:
1. Top 3 skill gaps
2. Why each matters for the target role
3. A 30-day learning plan
4. One mini-project for the intern
5. Recommended tools/technologies

Keep the answer concise, realistic for a beginner/intermediate software engineering intern,
and do not invent certifications or job requirements.
"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise AI career coach and skills-gap analyst."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_completion_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"Groq AI could not generate the recommendation: {exc}"

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🎯 Intern Skill Gap Analyzer</h1>
    <p>Analyze intern skills against industry job descriptions using NLP, TF-IDF and K-Means clustering — then generate an AI-powered training plan.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Project Settings")
    n_clusters = st.slider("Number of Job Clusters", 2, 8, 3)
    st.markdown("---")
    st.subheader("🤖 Groq AI")
    if os.getenv("GROQ_API_KEY"):
        st.success("GROQ_API_KEY detected")
    else:
        st.warning("GROQ_API_KEY not found")
        st.caption("Add it in Streamlit Cloud → Settings → Secrets.")
    st.markdown("---")
    st.caption("Model: TF-IDF + K-Means + Groq")
    st.caption("Groq model: openai/gpt-oss-20b")

# -----------------------------
# Data upload
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Upload Data", "🔎 Analyze Intern", "📈 Job Clusters"])

with tab1:
    st.subheader("Upload your datasets")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👨‍💻 Intern Skills CSV")
        intern_file = st.file_uploader(
            "Upload intern skills",
            type=["csv"],
            key="intern_csv",
            help="Recommended columns: intern_id, name, skills"
        )
        st.caption("Example: intern_id,name,skills")

    with col2:
        st.markdown("### 💼 Industry Jobs CSV")
        job_file = st.file_uploader(
            "Upload job descriptions",
            type=["csv"],
            key="job_csv",
            help="Recommended columns: job_id, title, description"
        )
        st.caption("Example: job_id,title,description")

    if intern_file and job_file:
        try:
            raw_interns = pd.read_csv(intern_file)
            raw_jobs = pd.read_csv(job_file)

            interns = prepare_interns(raw_interns)
            jobs = prepare_jobs(raw_jobs)

            st.session_state["interns"] = interns
            st.session_state["jobs"] = jobs

            c1, c2, c3 = st.columns(3)
            c1.metric("Interns", len(interns))
            c2.metric("Job Descriptions", len(jobs))
            c3.metric("Job Clusters", min(n_clusters, len(jobs)))

            st.success("Datasets loaded successfully.")

            st.markdown("### Intern Data Preview")
            st.dataframe(interns[["intern_id", "name", "skills"]], use_container_width=True)

            st.markdown("### Job Data Preview")
            st.dataframe(jobs[["job_id", "title", "description"]], use_container_width=True)

        except Exception as e:
            st.error(f"Could not process the CSV files: {e}")

    else:
        st.info("Upload both CSV files to start the analysis.")

with tab2:
    st.subheader("Analyze an intern against an industry role")

    if "interns" not in st.session_state or "jobs" not in st.session_state:
        st.info("First upload both datasets in the 'Upload Data' tab.")
    else:
        interns = st.session_state["interns"]
        jobs = st.session_state["jobs"]

        try:
            vectorizer, job_matrix, clustered_jobs = analyze_jobs(jobs, n_clusters)
            st.session_state["vectorizer"] = vectorizer
            st.session_state["job_matrix"] = job_matrix
            st.session_state["clustered_jobs"] = clustered_jobs

            intern_index = st.selectbox(
                "Select Intern",
                range(len(interns)),
                format_func=lambda i: f"{interns.iloc[i]['name']} — {interns.iloc[i]['intern_id']}"
            )

            selected_intern = interns.iloc[intern_index]

            job_index = st.selectbox(
                "Select Target Job",
                range(len(clustered_jobs)),
                format_func=lambda i: f"{clustered_jobs.iloc[i]['title']} — Cluster {clustered_jobs.iloc[i]['cluster'] + 1}"
            )

            selected_job = clustered_jobs.iloc[job_index]
            selected_job_vector = job_matrix[job_index]

            gaps = extract_skill_gaps(
                selected_intern["skills"],
                selected_job["description"],
                vectorizer,
                selected_job_vector
            )

            # Similarity score
            intern_vector = vectorizer.transform([selected_intern["clean_skills"]])
            similarity = float(cosine_similarity(intern_vector, selected_job_vector)[0][0])
            readiness = round(similarity * 100, 1)

            c1, c2, c3 = st.columns(3)
            c1.metric("Industry Match", f"{readiness}%")
            c2.metric("Potential Gaps", len(gaps))
            c3.metric("Job Cluster", f"Cluster {int(selected_job['cluster']) + 1}")

            st.markdown("### 🧠 Detected Skill Gaps")

            if gaps:
                gap_df = pd.DataFrame(
                    [{"Potential Gap": g, "TF-IDF Importance": round(score, 4)} for g, score in gaps]
                )
                st.dataframe(gap_df, use_container_width=True)

                st.markdown("### 📚 Training Priorities")
                for rank, (gap, score) in enumerate(gaps[:6], start=1):
                    st.markdown(
                        f'<div class="gap-card"><b>#{rank} {gap}</b><br>'
                        f'<span class="small-note">Priority score: {score:.4f}</span></div>',
                        unsafe_allow_html=True
                    )
            else:
                st.success("No major TF-IDF skill gaps were detected for this role.")

            st.markdown("### 🤖 AI Training Recommendation")

            if st.button("Generate AI Training Plan", type="primary"):
                with st.spinner("Analyzing skill gaps with Groq..."):
                    result = groq_recommendation(
                        selected_intern["name"],
                        selected_intern["skills"],
                        selected_job["title"],
                        gaps
                    )

                if result:
                    st.markdown(result)
                else:
                    st.warning(
                        "Groq is not configured. Add GROQ_API_KEY to Streamlit Secrets and try again."
                    )

            with st.expander("View selected job description"):
                st.write(selected_job["description"])

            with st.expander("View intern skills"):
                st.write(selected_intern["skills"])

        except Exception as e:
            st.error(f"Analysis error: {e}")

with tab3:
    st.subheader("Industry Job Clustering")

    if "clustered_jobs" not in st.session_state:
        st.info("Upload your job descriptions first.")
    else:
        clustered_jobs = st.session_state["clustered_jobs"]

        st.write(
            "K-Means groups similar industry job descriptions based on their TF-IDF representation."
        )

        cluster_counts = clustered_jobs["cluster"].value_counts().sort_index()
        chart_df = pd.DataFrame({
            "Cluster": [f"Cluster {i + 1}" for i in cluster_counts.index],
            "Number of Jobs": cluster_counts.values
        }).set_index("Cluster")

        st.bar_chart(chart_df)

        for cluster_id in sorted(clustered_jobs["cluster"].unique()):
            cluster_jobs = clustered_jobs[clustered_jobs["cluster"] == cluster_id]
            with st.expander(f"Cluster {cluster_id + 1} — {len(cluster_jobs)} jobs"):
                for _, row in cluster_jobs.iterrows():
                    st.markdown(f"**{row['title']}**")
                    st.caption(row["description"][:350] + ("..." if len(row["description"]) > 350 else ""))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Intern Skill Gap Analyzer • NLP + TF-IDF + K-Means + Groq AI")

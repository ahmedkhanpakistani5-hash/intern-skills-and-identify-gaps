import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


st.set_page_config(
    page_title="Intern Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DARK GREEN DESIGN
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #03150d 0%, #073b27 52%, #02100a 100%);
    color: #ecfdf5;
}
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stSidebar"] {
    background: #03120c;
    border-right: 1px solid #14532d;
}
h1, h2, h3 {
    color: #86efac !important;
}
p, label, .stMarkdown {
    color: #d1fae5 !important;
}
.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(135deg, #0b402b, #062417);
    border: 1px solid #15803d;
    margin-bottom: 24px;
    box-shadow: 0 15px 45px rgba(0,0,0,.25);
}
.hero h1 {
    margin: 0 0 8px 0;
    font-size: 2.5rem;
}
.hero p {
    font-size: 1.05rem;
    color: #bbf7d0 !important;
}
div[data-testid="stMetric"] {
    background: #08271a;
    border: 1px solid #166534;
    border-radius: 14px;
}
.stButton > button {
    background: linear-gradient(90deg, #15803d, #16a34a);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #16a34a, #22c55e);
}
.gap-card {
    padding: 14px 18px;
    border-radius: 12px;
    background: #092d1e;
    border-left: 4px solid #22c55e;
    margin: 8px 0;
}
.info-card {
    padding: 16px;
    border-radius: 14px;
    background: #08271a;
    border: 1px solid #14532d;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SAMPLE INDUSTRY JOB DATA
# This means Job Clusters and Intern Analysis work even
# before the user uploads their own job descriptions.
# =========================================================
DEFAULT_JOBS = pd.DataFrame([
    {
        "job_id": "J001",
        "title": "AI / Machine Learning Engineer",
        "description": (
            "Python machine learning artificial intelligence scikit-learn "
            "pandas numpy deep learning tensorflow pytorch SQL Git data analysis "
            "model evaluation feature engineering statistics"
        ),
    },
    {
        "job_id": "J002",
        "title": "Data Analyst",
        "description": (
            "Python SQL Excel Power BI Tableau pandas numpy statistics "
            "data visualization data cleaning reporting dashboards "
            "business intelligence analytical thinking"
        ),
    },
    {
        "job_id": "J003",
        "title": "Software Engineer",
        "description": (
            "Python Java C++ object oriented programming data structures "
            "algorithms SQL Git REST API Docker software development "
            "testing debugging databases"
        ),
    },
    {
        "job_id": "J004",
        "title": "Frontend Developer",
        "description": (
            "HTML CSS JavaScript React responsive design UI UX Git "
            "REST API frontend development TypeScript web development"
        ),
    },
    {
        "job_id": "J005",
        "title": "Backend Developer",
        "description": (
            "Python Java Node.js REST API SQL PostgreSQL MongoDB Docker "
            "Git backend development authentication databases cloud deployment"
        ),
    },
    {
        "job_id": "J006",
        "title": "DevOps / Cloud Engineer",
        "description": (
            "Linux Git Docker Kubernetes AWS Azure CI CD cloud computing "
            "Python networking automation infrastructure deployment monitoring"
        ),
    },
    {
        "job_id": "J007",
        "title": "Computer Vision Engineer",
        "description": (
            "Python OpenCV computer vision image processing deep learning "
            "PyTorch TensorFlow CNN object detection image classification "
            "numpy machine learning"
        ),
    },
    {
        "job_id": "J008",
        "title": "NLP Engineer",
        "description": (
            "Python natural language processing NLP transformers BERT "
            "machine learning deep learning text classification tokenization "
            "TF-IDF Hugging Face PyTorch data preprocessing"
        ),
    },
])


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9+#.\-/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_pdf_text(uploaded_file):
    if PdfReader is None:
        raise RuntimeError("pypdf is missing. Add pypdf to requirements.txt.")

    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages).strip()


def make_intern_from_pdf(uploaded_file):
    text = extract_pdf_text(uploaded_file)

    if not text:
        raise ValueError(
            "No selectable text was found in this PDF. "
            "Please upload a text-based PDF resume."
        )

    name = os.path.splitext(uploaded_file.name)[0]

    return pd.DataFrame([{
        "intern_id": "INTERN-001",
        "name": name,
        "skills": text,
        "clean_skills": clean_text(text),
    }])


def prepare_jobs(jobs):
    jobs = jobs.copy()
    jobs["clean_description"] = jobs["description"].map(clean_text)
    return jobs


def jobs_from_uploaded_pdfs(files):
    rows = []

    for i, uploaded_file in enumerate(files, start=1):
        text = extract_pdf_text(uploaded_file)

        if text:
            rows.append({
                "job_id": f"PDF-JOB-{i}",
                "title": os.path.splitext(uploaded_file.name)[0],
                "description": text,
            })

    if not rows:
        return pd.DataFrame()

    return prepare_jobs(pd.DataFrame(rows))


def run_clustering(jobs, number_of_clusters):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(jobs["clean_description"])

    actual_clusters = min(number_of_clusters, len(jobs))
    actual_clusters = max(1, actual_clusters)

    if actual_clusters == 1:
        labels = np.zeros(len(jobs), dtype=int)
    else:
        model = KMeans(
            n_clusters=actual_clusters,
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(matrix)

    result = jobs.copy()
    result["cluster"] = labels

    return vectorizer, matrix, result


def detect_gaps(intern_text, job_vector, vectorizer):
    intern_clean = clean_text(intern_text)
    intern_words = set(intern_clean.split())

    scores = np.asarray(job_vector.toarray()).ravel()
    terms = np.array(vectorizer.get_feature_names_out())

    order = scores.argsort()[::-1]

    generic_words = {
        "experience", "work", "working", "team", "skill", "skills",
        "ability", "strong", "knowledge", "role", "job", "using",
        "required", "requirements", "development", "good", "year",
        "years", "including"
    }

    gaps = []
    seen = set()

    for index in order:
        term = terms[index]
        normalized = term.replace("-", " ")

        if term in generic_words or len(term) < 2:
            continue

        if term in seen:
            continue

        # Check whether the skill/term is already present in the resume.
        if term not in intern_clean:
            parts = set(normalized.split())

            if not parts.issubset(intern_words):
                gaps.append((term, float(scores[index])))
                seen.add(term)

        if len(gaps) == 10:
            break

    return gaps


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🎯 Intern Skill Gap Analyzer</h1>
    <p>
        Compare an intern's resume with industry job requirements using
        NLP, TF-IDF and K-Means clustering to identify skill gaps and
        training priorities.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# NO GROQ API / NO GROQ SECTION
# =========================================================
with st.sidebar:
    st.header("⚙️ Project Settings")

    number_of_clusters = st.slider(
        "Number of Job Clusters",
        min_value=2,
        max_value=8,
        value=4,
    )

    st.markdown("---")

    st.markdown(
        "**ML Pipeline**  \n"
        "Resume → NLP → TF-IDF → Similarity → Skill Gaps  \n"
        "Industry Jobs → TF-IDF → K-Means → Job Clusters"
    )

    st.markdown("---")
    st.caption("AI Skill Gap Analysis")
    st.caption("NLP + TF-IDF + K-Means")


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Upload Data",
    "🔎 Analyze Intern",
    "📈 Job Clusters",
])


# =========================================================
# TAB 1
# =========================================================
with tab1:
    st.subheader("Upload your datasets")

    st.markdown("""
    <div class="info-card">
        <b>Intern:</b> Upload one or more resume PDFs.<br>
        <b>Industry:</b> You can use the built-in industry job dataset,
        or upload job-description PDFs to analyze your own industry data.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👨‍💻 Intern Resume")

        intern_file = st.file_uploader(
            "Upload Intern Resume PDF",
            type=["pdf"],
            key="intern_resume",
        )

        if intern_file:
            try:
                intern_df = make_intern_from_pdf(intern_file)
                st.session_state["interns"] = intern_df
                st.success("Resume loaded successfully.")
            except Exception as e:
                st.error(str(e))

    with col2:
        st.markdown("### 💼 Industry Jobs")

        st.markdown(
            "Built-in industry job dataset is active by default."
        )

        job_files = st.file_uploader(
            "Upload Job Description PDF(s) — Optional",
            type=["pdf"],
            accept_multiple_files=True,
            key="job_description_pdfs",
        )

        if job_files:
            try:
                uploaded_jobs = jobs_from_uploaded_pdfs(job_files)

                if not uploaded_jobs.empty:
                    st.session_state["jobs"] = uploaded_jobs
                    st.success(
                        f"{len(uploaded_jobs)} job description(s) loaded."
                    )
            except Exception as e:
                st.error(str(e))
        else:
            st.session_state["jobs"] = prepare_jobs(DEFAULT_JOBS)

    if "interns" in st.session_state:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Intern Resume",
            "Loaded"
        )

        c2.metric(
            "Industry Jobs",
            len(st.session_state["jobs"])
        )

        c3.metric(
            "Clusters",
            min(
                number_of_clusters,
                len(st.session_state["jobs"])
            )
        )

        st.success(
            "Your data is ready. Open the **Analyze Intern** tab."
        )

        with st.expander("View extracted resume text"):
            st.write(
                st.session_state["interns"].iloc[0]["skills"]
            )

    else:
        st.info(
            "Upload your Intern Resume PDF to start the analysis."
        )


# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.subheader("🔎 Intern Skill Gap Analysis")

    if "interns" not in st.session_state:
        st.warning(
            "Please upload an Intern Resume PDF in the Upload Data tab first."
        )
    else:
        interns = st.session_state["interns"]

        # Always make sure jobs exist.
        if "jobs" not in st.session_state:
            st.session_state["jobs"] = prepare_jobs(DEFAULT_JOBS)

        jobs = st.session_state["jobs"]

        try:
            vectorizer, job_matrix, clustered_jobs = run_clustering(
                jobs,
                number_of_clusters,
            )

            st.session_state["clustered_jobs"] = clustered_jobs

            selected_intern = interns.iloc[0]

            selected_job_index = st.selectbox(
                "Select Industry Role",
                range(len(clustered_jobs)),
                format_func=lambda i:
                    f"{clustered_jobs.iloc[i]['title']} — "
                    f"Cluster {clustered_jobs.iloc[i]['cluster'] + 1}",
            )

            selected_job = clustered_jobs.iloc[selected_job_index]
            selected_job_vector = job_matrix[selected_job_index]

            intern_vector = vectorizer.transform(
                [selected_intern["clean_skills"]]
            )

            similarity = float(
                cosine_similarity(
                    intern_vector,
                    selected_job_vector,
                )[0][0]
            )

            match_percentage = round(similarity * 100, 1)

            gaps = detect_gaps(
                selected_intern["skills"],
                selected_job_vector,
                vectorizer,
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Industry Match",
                f"{match_percentage}%"
            )

            c2.metric(
                "Skill Gaps",
                len(gaps)
            )

            c3.metric(
                "Job Cluster",
                f"{int(selected_job['cluster']) + 1}"
            )

            st.markdown("### 🎯 Skill Gap Results")

            if gaps:
                gap_data = []

                for rank, (skill, score) in enumerate(
                    gaps,
                    start=1
                ):
                    gap_data.append({
                        "Priority": rank,
                        "Potential Missing Skill": skill,
                        "Importance": round(score, 4),
                    })

                gap_df = pd.DataFrame(gap_data)

                st.dataframe(
                    gap_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("### 📚 Training Priorities")

                for rank, (skill, score) in enumerate(
                    gaps[:6],
                    start=1
                ):
                    st.markdown(
                        f"""
                        <div class="gap-card">
                            <b>#{rank} — {skill}</b><br>
                            TF-IDF importance: {score:.4f}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success(
                    "No major skill gaps were detected for this role."
                )

            st.markdown("### 💼 Selected Industry Role")
            st.info(selected_job["title"])

            with st.expander("View industry job requirements"):
                st.write(selected_job["description"])

            with st.expander("View extracted intern resume"):
                st.write(selected_intern["skills"])

        except Exception as e:
            st.error(
                f"Analysis could not be completed: {e}"
            )


# =========================================================
# TAB 3
# =========================================================
with tab3:
    st.subheader("📈 Industry Job Clusters")

    if "jobs" not in st.session_state:
        st.session_state["jobs"] = prepare_jobs(DEFAULT_JOBS)

    jobs = st.session_state["jobs"]

    try:
        _, _, clustered_jobs = run_clustering(
            jobs,
            number_of_clusters,
        )

        st.session_state["clustered_jobs"] = clustered_jobs

        st.success(
            f"K-Means successfully grouped {len(jobs)} industry roles."
        )

        counts = (
            clustered_jobs["cluster"]
            .value_counts()
            .sort_index()
        )

        chart_df = pd.DataFrame({
            "Cluster": [
                f"Cluster {i + 1}"
                for i in counts.index
            ],
            "Number of Jobs": counts.values,
        }).set_index("Cluster")

        st.bar_chart(chart_df)

        st.markdown("### Cluster Details")

        for cluster_id in sorted(
            clustered_jobs["cluster"].unique()
        ):
            cluster_jobs = clustered_jobs[
                clustered_jobs["cluster"] == cluster_id
            ]

            with st.expander(
                f"Cluster {cluster_id + 1} — "
                f"{len(cluster_jobs)} job(s)"
            ):
                for _, row in cluster_jobs.iterrows():
                    st.markdown(
                        f"**{row['title']}**"
                    )

                    st.caption(
                        row["description"][:400]
                        + (
                            "..."
                            if len(row["description"]) > 400
                            else ""
                        )
                    )

    except Exception as e:
        st.error(
            f"Clustering could not be completed: {e}"
        )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Intern Skill Gap Analyzer • NLP + TF-IDF + K-Means"
)

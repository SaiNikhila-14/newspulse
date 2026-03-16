"""
NewsPulse — Flask Backend
Authentication : MySQL + bcrypt password hashing
New features   : Bookmarks, Dark/Light mode (pref stored), User profile,
                 Sentiment trend over time, Search history, User CSV export
Run            : python app.py
Open           : http://localhost:5000
"""

from dotenv import load_dotenv
load_dotenv()

from flask import (Flask, render_template, request, jsonify,
                   session, send_file, redirect, url_for)
import pandas as pd
import numpy as np
import re, io, base64, os
from collections import Counter
from datetime import datetime
from functools import wraps

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import bcrypt
import mysql.connector
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "newspulse_dev_secret_change_me")

# ══════════════════════════════════════════════════════════════════
# MYSQL CONFIG
# ══════════════════════════════════════════════════════════════════

DB_CONFIG = {
    "host":     os.environ.get("DB_HOST",     "localhost"),
    "port":     int(os.environ.get("DB_PORT", 3306)),
    "user":     os.environ.get("DB_USER",     "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME",     "newspulse"),
}

# ══════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════

def get_db():
    return mysql.connector.connect(**DB_CONFIG)

def init_db():
    cfg  = {k: v for k, v in DB_CONFIG.items() if k != "database"}
    conn = mysql.connector.connect(**cfg)
    cur  = conn.cursor()

    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    cur.execute(f"USE `{DB_CONFIG['database']}`")

    # ── users ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INT          AUTO_INCREMENT PRIMARY KEY,
            username   VARCHAR(80)  UNIQUE NOT NULL,
            email      VARCHAR(120) UNIQUE NOT NULL,
            password   VARCHAR(255) NOT NULL,
            role       ENUM('user','admin') NOT NULL DEFAULT 'user',
            theme      ENUM('dark','light') NOT NULL DEFAULT 'dark',
            created_at DATETIME     DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── bookmarks ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bookmarks (
            id          INT          AUTO_INCREMENT PRIMARY KEY,
            user_id     INT          NOT NULL,
            title       TEXT         NOT NULL,
            description TEXT,
            source      VARCHAR(255),
            published   VARCHAR(100),
            sentiment   VARCHAR(20),
            created_at  DATETIME     DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # ── search history ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id         INT          AUTO_INCREMENT PRIMARY KEY,
            user_id    INT          NOT NULL,
            query      VARCHAR(255) NOT NULL,
            searched_at DATETIME    DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    conn.commit()

    # Seed default admin
    cur.execute("SELECT id FROM users WHERE role='admin' LIMIT 1")
    if not cur.fetchone():
        hashed = bcrypt.hashpw(b"admin123", bcrypt.gensalt()).decode()
        cur.execute(
            "INSERT INTO users (username, email, password, role) VALUES (%s,%s,%s,%s)",
            ("admin", "admin@newspulse.com", hashed, "admin")
        )
        conn.commit()
        print("✅  Default admin seeded  →  username: admin  |  password: admin123")

    cur.close(); conn.close()

init_db()

# ══════════════════════════════════════════════════════════════════
# AUTH DECORATORS
# ══════════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Unauthorized"}), 403
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user_id") or session.get("role") != "admin":
            return jsonify({"error": "Unauthorized"}), 403
        return f(*args, **kwargs)
    return wrapper

# ══════════════════════════════════════════════════════════════════
# NLP / ML SETUP
# ══════════════════════════════════════════════════════════════════

POS_KEYWORDS = [
    'win','success','growth','develop','achiev','innovat','launch','boost',
    'improve','benefit','profit','advance','excel','strong','progress',
    'celebrat','record','best','great','top','award','invest','rise',
    'gain','increas','expand','lead','new','upgrad','breakthrough',
    'renew','sustain','commit','promot','support','partner','collabor',
    'inaugurat','open','approv','fund','reform','opportun','thrive',
    'restor','empower','recover','posit','honor','promis','relief',
    'accord','deal','partnership','alli','reinforc','capac','competit',
    'summit','buy','pick','tech','ai','digital','smart','efficient'
]
NEG_KEYWORDS = [
    'war','conflict','attack','kill','death','crisis','loss','fail',
    'declin','drop','fall','risk','threat','violenc','disast',
    'protest','arrest','terror','ban','fraud','corrupt',
    'pollut','accident','injur','concern','alarm','critical','problem',
    'controversi','tension','unrest','crackdown','seiz','deni',
    'strike','flood','earthquake','famin','recession','unemploy',
    'inflat','debt','deficit','breach','hack','exploit','victim',
    'casualt','destruct','collaps','resign','impeach','shoot',
    'murder','rape','abus','disput','miss','outag','replac','warn',
    'skip','sanction','suspend','shut','close','fine','penalt'
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def compute_score(text):
    tl = text.lower()
    return (sum(1 for k in POS_KEYWORDS if k in tl)
          - sum(1 for k in NEG_KEYWORDS if k in tl))

class RichKeywordFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        rows = []
        for text in X:
            tl  = text.lower()
            pos = sum(1 for k in POS_KEYWORDS if k in tl)
            neg = sum(1 for k in NEG_KEYWORDS if k in tl)
            net = pos - neg
            rows.append([pos, neg, net, max(net-1,0), max(-1-net,0),
                         float(pos)/max(neg,1), float(net**2)])
        return csr_matrix(np.array(rows, dtype=float))

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                dpi=110, facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64

def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "gnews_data_cleaned.csv")
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().dropna(subset=["Title", "Description"])
    df = df.drop_duplicates(subset=["Title"])
    df["full_text"]    = df["Title"] + " " + df["Description"]
    df["cleaned_text"] = df["full_text"].apply(clean_text)
    df["kw_score"]     = df["cleaned_text"].apply(compute_score)
    p25 = df["kw_score"].quantile(0.25)
    p60 = df["kw_score"].quantile(0.60)
    def label(s):
        if s >= p60: return "Positive"
        if s <= p25: return "Negative"
        return "Neutral"
    df["sentiment_label"] = df["kw_score"].apply(label)
    # Parse date for trend chart
    if "Published Date" in df.columns:
        df["pub_date"] = pd.to_datetime(df["Published Date"], errors="coerce")
    return df, p25, p60

DF, P25, P60  = load_data()
TFIDF_VIZ     = TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                                 sublinear_tf=True, min_df=1, max_df=0.95)
TFIDF_MATRIX  = TFIDF_VIZ.fit_transform(DF["cleaned_text"])
FEATURE_NAMES = TFIDF_VIZ.get_feature_names_out()
MODEL_STORE   = {}

# ══════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/register")
def register_page():
    if session.get("user_id"):
        return redirect(url_for("user_dashboard" if session.get("role") == "user"
                                else "admin_dashboard"))
    return render_template("register.html")

@app.route("/user/login")
def user_login_page():
    if session.get("user_id") and session.get("role") == "user":
        return redirect(url_for("user_dashboard"))
    return render_template("user_login.html")

@app.route("/user/dashboard")
def user_dashboard():
    if not session.get("user_id"):
        return redirect(url_for("user_login_page"))
    return render_template("index.html", portal="user")

@app.route("/admin/login")
def admin_login_page():
    if session.get("user_id") and session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    return render_template("admin_login.html")

@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("user_id") or session.get("role") != "admin":
        return redirect(url_for("admin_login_page"))
    return render_template("index.html", portal="admin")

# ══════════════════════════════════════════════════════════════════
# AUTH API
# ══════════════════════════════════════════════════════════════════

@app.route("/api/register", methods=["POST"])
def register():
    data     = request.json or {}
    username = data.get("username", "").strip()
    email    = data.get("email",    "").strip().lower()
    password = data.get("password", "")

    if not username or not email or not password:
        return jsonify({"success": False, "error": "All fields are required."}), 400
    if len(username) < 3:
        return jsonify({"success": False, "error": "Username must be at least 3 characters."}), 400
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        return jsonify({"success": False, "error": "Invalid email address."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters."}), 400

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password, role) VALUES (%s,%s,%s,'user')",
            (username, email, hashed))
        conn.commit()
    except mysql.connector.IntegrityError as e:
        field = "Username" if "username" in str(e).lower() else "Email"
        return jsonify({"success": False, "error": f"{field} already taken."}), 409
    finally:
        cur.close(); conn.close()

    return jsonify({"success": True, "message": "Account created! Please log in.",
                    "redirect": url_for("user_login_page")})


@app.route("/api/user/login", methods=["POST"])
def user_login_api():
    data     = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"success": False, "error": "All fields are required."}), 400

    conn = get_db(); cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE username=%s AND role='user'", (username,))
    user = cur.fetchone(); cur.close(); conn.close()

    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return jsonify({"success": False, "error": "Invalid username or password."}), 401

    session["user_id"]  = user["id"]
    session["username"] = user["username"]
    session["role"]     = user["role"]
    session["theme"]    = user.get("theme", "dark")
    return jsonify({"success": True, "redirect": url_for("user_dashboard")})


@app.route("/api/admin/login", methods=["POST"])
def admin_login_api():
    data     = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"success": False, "error": "All fields are required."}), 400

    conn = get_db(); cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE username=%s AND role='admin'", (username,))
    user = cur.fetchone(); cur.close(); conn.close()

    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return jsonify({"success": False, "error": "Invalid admin credentials."}), 401

    session["user_id"]  = user["id"]
    session["username"] = user["username"]
    session["role"]     = "admin"
    session["theme"]    = user.get("theme", "dark")
    return jsonify({"success": True, "redirect": url_for("admin_dashboard")})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True, "redirect": url_for("landing")})


@app.route("/api/session")
def get_session():
    return jsonify({
        "user_id":  session.get("user_id"),
        "username": session.get("username", ""),
        "role":     session.get("role", ""),
        "theme":    session.get("theme", "dark"),
        "is_user":  session.get("role") == "user",
        "is_admin": session.get("role") == "admin",
    })

# ══════════════════════════════════════════════════════════════════
# USER API — general
# ══════════════════════════════════════════════════════════════════

@app.route("/api/stats")
@login_required
def stats():
    sc = DF["sentiment_label"].value_counts()
    return jsonify({
        "total":    int(len(DF)),
        "positive": int(sc.get("Positive", 0)),
        "negative": int(sc.get("Negative", 0)),
        "neutral":  int(sc.get("Neutral",  0)),
        "sources":  int(DF["Source"].nunique()) if "Source" in DF.columns else 0,
    })

@app.route("/api/articles")
@login_required
def articles():
    sentiment = request.args.get("sentiment", "All")
    source    = request.args.get("source",    "All")
    search    = request.args.get("search",    "").strip()
    page      = int(request.args.get("page",     1))
    per_page  = int(request.args.get("per_page", 20))

    # Save search to history (non-empty, deduplicate recent)
    if search and session.get("user_id"):
        try:
            conn = get_db(); cur = conn.cursor()
            # keep only last 20 unique queries
            cur.execute(
                "DELETE FROM search_history WHERE user_id=%s AND query=%s",
                (session["user_id"], search))
            cur.execute(
                "INSERT INTO search_history (user_id, query) VALUES (%s,%s)",
                (session["user_id"], search))
            # trim to 20
            cur.execute("""
                DELETE FROM search_history WHERE user_id=%s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id FROM search_history WHERE user_id=%s
                        ORDER BY searched_at DESC LIMIT 20
                    ) t
                )
            """, (session["user_id"], session["user_id"]))
            conn.commit(); cur.close(); conn.close()
        except Exception:
            pass

    df = DF.copy()
    if sentiment != "All":
        df = df[df["sentiment_label"] == sentiment]
    if source != "All" and "Source" in df.columns:
        df = df[df["Source"] == source]
    if search:
        mask = (df["Title"].str.contains(search, case=False, na=False) |
                df["Description"].str.contains(search, case=False, na=False))
        df = df[mask]

    total  = len(df)
    start  = (page - 1) * per_page
    subset = df.iloc[start:start + per_page]
    cols   = [c for c in ["Title","Description","Source","Published Date","sentiment_label"]
              if c in subset.columns]
    return jsonify({
        "total": total, "page": page, "per_page": per_page,
        "articles": subset[cols].fillna("").to_dict(orient="records")
    })

@app.route("/api/sources")
@login_required
def sources():
    if "Source" not in DF.columns:
        return jsonify([])
    return jsonify(sorted(DF["Source"].dropna().unique().tolist()))

@app.route("/api/trending")
@login_required
def trending():
    freq = Counter(" ".join(DF["cleaned_text"]).split())
    return jsonify([{"word": w, "count": c} for w, c in freq.most_common(20)])

@app.route("/api/sentiment_chart")
@login_required
def sentiment_chart():
    sc = DF["sentiment_label"].value_counts()
    return jsonify({"labels": sc.index.tolist(), "values": sc.values.tolist()})

# ── NEW: Sentiment trend over time ────────────────────────────────
@app.route("/api/sentiment_trend")
@login_required
def sentiment_trend():
    if "pub_date" not in DF.columns:
        return jsonify({"labels": [], "positive": [], "negative": [], "neutral": []})

    df = DF.dropna(subset=["pub_date"]).copy()
    df["month"] = df["pub_date"].dt.to_period("M").astype(str)
    grp = df.groupby(["month","sentiment_label"]).size().unstack(fill_value=0).sort_index()

    labels = grp.index.tolist()
    return jsonify({
        "labels":   labels,
        "positive": [int(grp.loc[m, "Positive"]) if "Positive" in grp.columns else 0 for m in labels],
        "negative": [int(grp.loc[m, "Negative"]) if "Negative" in grp.columns else 0 for m in labels],
        "neutral":  [int(grp.loc[m, "Neutral"])  if "Neutral"  in grp.columns else 0 for m in labels],
    })

# ── NEW: User CSV export (filtered articles) ─────────────────────
@app.route("/api/user/export")
@login_required
def user_export():
    sentiment = request.args.get("sentiment", "All")
    source    = request.args.get("source",    "All")
    search    = request.args.get("search",    "").strip()

    df = DF.copy()
    if sentiment != "All":
        df = df[df["sentiment_label"] == sentiment]
    if source != "All" and "Source" in df.columns:
        df = df[df["Source"] == source]
    if search:
        mask = (df["Title"].str.contains(search, case=False, na=False) |
                df["Description"].str.contains(search, case=False, na=False))
        df = df[mask]

    cols = [c for c in ["Title","Description","Source","Published Date","sentiment_label"]
            if c in df.columns]
    buf = io.StringIO()
    df[cols].fillna("").to_csv(buf, index=False)
    buf.seek(0)
    fname = f"newspulse_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(io.BytesIO(buf.getvalue().encode()),
                     mimetype="text/csv", as_attachment=True, download_name=fname)

# ══════════════════════════════════════════════════════════════════
# USER PROFILE API
# ══════════════════════════════════════════════════════════════════

@app.route("/api/profile")
@login_required
def get_profile():
    conn = get_db(); cur = conn.cursor(dictionary=True)
    cur.execute(
    "SELECT id, username, email, role, theme, created_at FROM users WHERE id=%s",
    (session["user_id"],)
)

    user = cur.fetchone()
    cur.close()
    conn.close()
    if not user:
        return jsonify({"error": "User not found"}), 404
    user["created_at"] = str(user["created_at"])

    # bookmark count
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM bookmarks WHERE user_id=%s", (session["user_id"],))
    bcount = cur.fetchone()[0]; cur.close(); conn.close()

    user["bookmark_count"] = bcount
    return jsonify(user)


@app.route("/api/profile/update", methods=["POST"])
@login_required
def update_profile():
    data     = request.json or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if email and not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        return jsonify({"success": False, "error": "Invalid email address."}), 400
    if password and len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters."}), 400

    conn = get_db(); cur = conn.cursor()
    if email:
        try:
            cur.execute("UPDATE users SET email=%s WHERE id=%s",
                        (email, session["user_id"]))
        except mysql.connector.IntegrityError:
            return jsonify({"success": False, "error": "Email already in use."}), 409
    if password:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        cur.execute("UPDATE users SET password=%s WHERE id=%s",
                    (hashed, session["user_id"]))
    conn.commit(); cur.close(); conn.close()
    return jsonify({"success": True, "message": "Profile updated."})


@app.route("/api/profile/theme", methods=["POST"])
@login_required
def update_theme():
    theme = request.json.get("theme", "dark")
    if theme not in ("dark", "light"):
        return jsonify({"success": False, "error": "Invalid theme."}), 400
    conn = get_db(); cur = conn.cursor()
    cur.execute("UPDATE users SET theme=%s WHERE id=%s", (theme, session["user_id"]))
    conn.commit(); cur.close(); conn.close()
    session["theme"] = theme
    return jsonify({"success": True, "theme": theme})

# ══════════════════════════════════════════════════════════════════
# BOOKMARKS API
# ══════════════════════════════════════════════════════════════════

@app.route("/api/bookmarks")
@login_required
def get_bookmarks():
    conn = get_db(); cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM bookmarks WHERE user_id=%s ORDER BY created_at DESC",
        (session["user_id"],))
    rows = cur.fetchall(); cur.close(); conn.close()
    for r in rows:
        r["created_at"] = str(r["created_at"])
    return jsonify(rows)


@app.route("/api/bookmarks/add", methods=["POST"])
@login_required
def add_bookmark():
    data = request.json or {}
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"success": False, "error": "Title required."}), 400

    # Prevent duplicate bookmarks for same user+title
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT id FROM bookmarks WHERE user_id=%s AND title=%s",
                (session["user_id"], title))
    if cur.fetchone():
        cur.close(); conn.close()
        return jsonify({"success": False, "error": "Already bookmarked."}), 409

    cur.execute(
        "INSERT INTO bookmarks (user_id,title,description,source,published,sentiment) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        (session["user_id"], title,
         data.get("description",""), data.get("source",""),
         data.get("published",""),  data.get("sentiment","")))
    conn.commit()
    new_id = cur.lastrowid
    cur.close(); conn.close()
    return jsonify({"success": True, "id": new_id})


@app.route("/api/bookmarks/remove", methods=["POST"])
@login_required
def remove_bookmark():
    title = (request.json or {}).get("title", "")
    conn = get_db(); cur = conn.cursor()
    cur.execute("DELETE FROM bookmarks WHERE user_id=%s AND title=%s",
                (session["user_id"], title))
    conn.commit(); cur.close(); conn.close()
    return jsonify({"success": True})

# ══════════════════════════════════════════════════════════════════
# SEARCH HISTORY API
# ══════════════════════════════════════════════════════════════════

@app.route("/api/search_history")
@login_required
def get_search_history():
    conn = get_db(); cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT id, query, searched_at FROM search_history "
        "WHERE user_id=%s ORDER BY searched_at DESC LIMIT 20",
        (session["user_id"],))
    rows = cur.fetchall(); cur.close(); conn.close()
    for r in rows:
        r["searched_at"] = str(r["searched_at"])
    return jsonify(rows)


@app.route("/api/search_history/clear", methods=["POST"])
@login_required
def clear_search_history():
    conn = get_db(); cur = conn.cursor()
    cur.execute("DELETE FROM search_history WHERE user_id=%s", (session["user_id"],))
    conn.commit(); cur.close(); conn.close()
    return jsonify({"success": True})

# ══════════════════════════════════════════════════════════════════
# ADMIN API
# ══════════════════════════════════════════════════════════════════

@app.route("/api/admin/wordcloud")
@admin_required
def wordcloud_img():
    wc = WordCloud(width=1200, height=400, background_color="#0e1117",
                   colormap="cool", max_words=120).generate(" ".join(DF["cleaned_text"]))
    fig, ax = plt.subplots(figsize=(13, 4), facecolor="#0e1117")
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    return jsonify({"image": fig_to_base64(fig)})

@app.route("/api/admin/tfidf")
@admin_required
def tfidf_chart():
    mean_scores = np.asarray(TFIDF_MATRIX.mean(axis=0)).flatten()
    top_idx     = mean_scores.argsort()[-12:][::-1]
    return jsonify({
        "words":  [FEATURE_NAMES[i] for i in top_idx],
        "scores": [round(float(mean_scores[i]), 5) for i in top_idx],
    })

@app.route("/api/admin/lda")
@admin_required
def lda_topics():
    lda = LatentDirichletAllocation(n_components=3, random_state=42, max_iter=10)
    lda.fit(TFIDF_MATRIX)
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[-10:][::-1]
        topics.append({
            "id": idx + 1,
            "words":  [FEATURE_NAMES[i] for i in top_idx],
            "scores": [round(float(topic[i]), 4) for i in top_idx],
        })
    return jsonify({"topics": topics,
                    "perplexity": round(lda.perplexity(TFIDF_MATRIX), 2)})

@app.route("/api/admin/sentiment_by_source")
@admin_required
def sentiment_by_source():
    if "Source" not in DF.columns:
        return jsonify([])
    grp = DF.groupby(["Source","sentiment_label"]).size().unstack(fill_value=0)
    result = []
    for src in grp.index[:15]:
        row = {"source": src}
        for col in ["Positive","Negative","Neutral"]:
            row[col.lower()] = int(grp.loc[src, col]) if col in grp.columns else 0
        result.append(row)
    return jsonify(result)

@app.route("/api/admin/train", methods=["POST"])
@admin_required
def train_model():
    X = DF["cleaned_text"]; y = DF["sentiment_label"]
    if y.value_counts().min() < 3:
        return jsonify({"error": "Insufficient samples in one class."}), 400

    kf_ext = RichKeywordFeatures()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    tfidf_f = TfidfVectorizer(max_features=3000, ngram_range=(1,2),
                              sublinear_tf=True, min_df=1, max_df=0.95)
    X_tr = hstack([tfidf_f.fit_transform(X_train), kf_ext.transform(X_train)*3])
    X_te = hstack([tfidf_f.transform(X_test),      kf_ext.transform(X_test)*3])
    tfidf_cv = TfidfVectorizer(max_features=3000, ngram_range=(1,2),
                               sublinear_tf=True, min_df=1, max_df=0.95)
    X_all = hstack([tfidf_cv.fit_transform(X), kf_ext.transform(X)*3])

    clf = CalibratedClassifierCV(
        LinearSVC(max_iter=10000, C=1.0, class_weight="balanced"), cv=3)
    cv_scores = cross_val_score(
        clf, X_all, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy", n_jobs=-1)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_test, y_pred)
    lbls   = sorted(y.unique())
    cm     = confusion_matrix(y_test, y_pred, labels=lbls)
    report = classification_report(y_test, y_pred, labels=lbls, output_dict=True)
    MODEL_STORE.update({"clf": clf, "tfidf": tfidf_f, "kf": kf_ext, "labels": lbls})

    return jsonify({
        "accuracy":    round(acc*100, 2),
        "cv_mean":     round(float(cv_scores.mean())*100, 2),
        "cv_std":      round(float(cv_scores.std())*100,  2),
        "cv_scores":   [round(s*100, 2) for s in cv_scores],
        "conf_matrix": cm.tolist(), "conf_labels": lbls,
        "report": [
            {"class": cls,
             "precision": round(report[cls]["precision"]*100, 1),
             "recall":    round(report[cls]["recall"]*100,    1),
             "f1":        round(report[cls]["f1-score"]*100,  1),
             "support":   int(report[cls]["support"])}
            for cls in lbls],
    })

@app.route("/api/admin/predict", methods=["POST"])
@admin_required
def predict():
    if "clf" not in MODEL_STORE:
        return jsonify({"error": "Model not trained yet."}), 400
    text    = request.json.get("text", "")
    cleaned = clean_text(text)
    clf, tf, kf = MODEL_STORE["clf"], MODEL_STORE["tfidf"], MODEL_STORE["kf"]
    X_in  = hstack([tf.transform([cleaned]), kf.transform([cleaned])*5])
    pred  = clf.predict(X_in)[0]
    probs = clf.predict_proba(X_in)[0]
    tl    = text.lower()
    return jsonify({
        "prediction":    pred,
        "confidence":    round(float(probs.max())*100, 2),
        "probabilities": [{"label": c, "prob": round(float(p)*100, 2)}
                          for c, p in zip(clf.classes_, probs)],
        "pos_keywords":  [k for k in POS_KEYWORDS if k in tl],
        "neg_keywords":  [k for k in NEG_KEYWORDS if k in tl],
        "net_score":     (sum(1 for k in POS_KEYWORDS if k in tl)
                        - sum(1 for k in NEG_KEYWORDS if k in tl)),
    })

@app.route("/api/admin/export")
@admin_required
def export_csv():
    sentiment = request.args.get("sentiment", "All")
    source    = request.args.get("source",    "All")
    exp_type  = request.args.get("type",      "articles")

    if exp_type == "trends":
        freq   = Counter(" ".join(DF["cleaned_text"]).split())
        n      = int(request.args.get("n", 50))
        out_df = pd.DataFrame(freq.most_common(n), columns=["Keyword","Frequency"])
        out_df.insert(0, "Rank", range(1, n+1))
    elif exp_type == "sentiment":
        sc = DF["sentiment_label"].value_counts().reset_index()
        sc.columns = ["Sentiment","Count"]
        sc["Percentage"] = (sc["Count"] / len(DF) * 100).round(2)
        out_df = sc
    else:
        out_df = DF.copy()
        if sentiment != "All": out_df = out_df[out_df["sentiment_label"] == sentiment]
        if source != "All" and "Source" in out_df.columns:
            out_df = out_df[out_df["Source"] == source]
        cols = [c for c in ["Title","Description","Source","Published Date",
                             "sentiment_label","kw_score"] if c in out_df.columns]
        out_df = out_df[cols]

    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)
    fname = f"newspulse_{exp_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(io.BytesIO(buf.getvalue().encode()),
                     mimetype="text/csv", as_attachment=True, download_name=fname)

# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🚀 NewsPulse  →  http://localhost:5000\n")
    app.run(debug=True, port=5000, use_reloader=False)

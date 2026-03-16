# 📰 NewsPulse

A Flask-based news sentiment analysis dashboard with separate **User** and **Admin** portals, real authentication backed by **MySQL**, and an ML pipeline for sentiment classification.

---

## ✨ Features

| Portal | Capabilities |
|--------|-------------|
| **User** | Register & login · Browse/search/filter articles · Sentiment stats · Trending keywords |
| **Admin** | Everything above · Word cloud · TF-IDF · LDA topic modeling · Train ML model · Live prediction · CSV export |

---

## 🛠️ Tech Stack

- **Backend** — Flask (Python)
- **Database** — MySQL (via `mysql-connector-python`)
- **Auth** — bcrypt password hashing + Flask sessions
- **ML** — scikit-learn (TF-IDF · LinearSVC · LDA)
- **Viz** — Chart.js · Matplotlib · WordCloud

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/newspulse.git
cd newspulse
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up MySQL

Open **MySQL Workbench** and run:
```sql
CREATE DATABASE newspulse CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```
That's it — the app creates the `users` table automatically on first run.

### 5. Configure your database password

Open `app.py` and find the `DB_CONFIG` block near the top:
```python
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "",   # ← put your MySQL root password here
    "database": "newspulse",
}
```
Or set environment variables instead (recommended for production):
```bash
set DB_PASSWORD=yourpassword   # Windows
export DB_PASSWORD=yourpassword  # Mac/Linux
```

### 6. Add the dataset
Place `gnews_data_cleaned.csv` in the project root (same folder as `app.py`).

### 7. Run
```bash
python app.py
```
Open **http://localhost:5000**

---

## 🔐 Default Credentials

A default admin account is seeded automatically on first run:

| Field    | Value                   |
|----------|-------------------------|
| Username | `admin`                 |
| Password | `admin123`              |
| Email    | `admin@newspulse.com`   |

> ⚠️ **Change the admin password** after first login in production.

Regular users register at `/register`.

---

## 📁 Project Structure

```
newspulse/
├── app.py                    # Flask app — routes, auth, ML, APIs
├── requirements.txt          # Python dependencies
├── .gitignore
├── gnews_data_cleaned.csv    # Dataset (not tracked by git)
└── templates/
    ├── landing.html          # Home — choose User or Admin
    ├── register.html         # User registration
    ├── user_login.html       # User login
    ├── admin_login.html      # Admin login
    └── index.html            # Main SPA dashboard (user + admin)
```

---

## 🌐 API Reference

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/register` | Register new user |
| POST | `/api/user/login` | User login |
| POST | `/api/admin/login` | Admin login |
| POST | `/api/logout` | Logout |
| GET  | `/api/session` | Current session info |

### User (requires login)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stats` | Article counts by sentiment |
| GET | `/api/articles` | Paginated/filtered articles |
| GET | `/api/sources` | List of news sources |
| GET | `/api/trending` | Top 20 keywords |
| GET | `/api/sentiment_chart` | Chart data |

### Admin only
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/wordcloud` | Word cloud image |
| GET | `/api/admin/tfidf` | Top TF-IDF features |
| GET | `/api/admin/lda` | LDA topic modeling |
| GET | `/api/admin/sentiment_by_source` | Per-source breakdown |
| POST | `/api/admin/train` | Train ML model |
| POST | `/api/admin/predict` | Predict sentiment |
| GET | `/api/admin/export` | Download CSV |

---

## 📜 License

MIT

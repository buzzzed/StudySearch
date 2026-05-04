"""
StudySearch Backend v4
BM25 retrieval + Claude AI + User Accounts
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import sqlite3, json, os, re, math, uuid, zipfile, io, time, httpx, random
import hashlib, secrets
from pathlib import Path
from contextlib import asynccontextmanager
from xml.etree import ElementTree as ET
import pypdf
import libsql_experimental as libsql
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN   = os.getenv("TURSO_AUTH_TOKEN", "")
DB_PATH            = "studysearch.db"
CHAT_MODEL        = "claude-sonnet-4-20250514"
CHUNK_SIZE_WORDS  = 80
MAX_RESULTS       = 8
K1, B             = 1.5, 0.75
TOKEN_DAYS        = 30

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,https://studysearch.onrender.com").split(",")

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ─── AP CLASSES ──────────────────────────────────────────────────────────────

AP_CLASSES = [
    "AP African American Studies",
    "AP Art History",
    "AP Biology",
    "AP Calculus AB",
    "AP Calculus BC",
    "AP Chemistry",
    "AP Chinese Language and Culture",
    "AP Comparative Government and Politics",
    "AP Computer Science A",
    "AP Computer Science Principles",
    "AP English Language and Composition",
    "AP English Literature and Composition",
    "AP Environmental Science",
    "AP European History",
    "AP French Language and Culture",
    "AP German Language and Culture",
    "AP Human Geography",
    "AP Italian Language and Culture",
    "AP Japanese Language and Culture",
    "AP Latin",
    "AP Macroeconomics",
    "AP Microeconomics",
    "AP Music Theory",
    "AP Physics 1: Algebra-Based",
    "AP Physics 2: Algebra-Based",
    "AP Physics C: Electricity and Magnetism",
    "AP Physics C: Mechanics",
    "AP Precalculus",
    "AP Psychology",
    "AP Research",
    "AP Seminar",
    "AP Spanish Language and Culture",
    "AP Spanish Literature and Culture",
    "AP Statistics",
    "AP Studio Art: 2-D Design",
    "AP Studio Art: 3-D Design",
    "AP Studio Art: Drawing",
    "AP United States Government and Politics",
    "AP United States History",
    "AP World History: Modern",
]

# ─── DB ───────────────────────────────────────────────────────────────────────

class DictCursor:
    """Wraps a libsql cursor so fetchone/fetchall return dict-like rows."""
    def __init__(self, cursor):
        self._cursor = cursor
    @property
    def description(self):
        return self._cursor.description
    @property
    def lastrowid(self):
        return self._cursor.lastrowid
    def _to_dict(self, row):
        if row is None:
            return None
        cols = [d[0] for d in self._cursor.description]
        return dict(zip(cols, row))
    def fetchone(self):
        return self._to_dict(self._cursor.fetchone())
    def fetchall(self):
        return [self._to_dict(r) for r in self._cursor.fetchall()]
    def __iter__(self):
        return (self._to_dict(r) for r in self._cursor)

class TursoConnection:
    """Wraps a libsql connection to return dict rows like sqlite3.Row."""
    def __init__(self, conn):
        self._conn = conn
    def execute(self, sql, params=()):
        return DictCursor(self._conn.execute(sql, params))
    def executescript(self, sql):
        """Execute multiple statements one by one (executescript is unreliable on remote libsql)."""
        statements = [s.strip() for s in sql.split(';') if s.strip()]
        for stmt in statements:
            try:
                self._conn.execute(stmt)
            except Exception:
                pass
        self._conn.commit()
        self._sync()
    def commit(self):
        self._conn.commit()
        self._sync()
    def _sync(self):
        """Sync to remote if using embedded replica mode."""
        if hasattr(self._conn, 'sync'):
            try:
                self._conn.sync()
            except Exception:
                pass
    def close(self):
        self._conn.close()

def get_db():
    if TURSO_DATABASE_URL:
        try:
            # For remote-only Turso, pass URL as database param
            conn = libsql.connect(database=TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)
            # Sync if embedded replica mode
            if hasattr(conn, 'sync'):
                try:
                    conn.sync()
                except Exception:
                    pass
            return TursoConnection(conn)
        except Exception as e:
            import logging
            logging.getLogger("studysearch").error(f"[DB] Turso connection failed: {e}, falling back to local SQLite")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at REAL NOT NULL,
            bio TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS auth_tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT 'My Notebook',
            created_at REAL NOT NULL, last_used REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, filename TEXT NOT NULL,
            file_type TEXT NOT NULL, slide_count INTEGER NOT NULL DEFAULT 0, uploaded_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, session_id TEXT NOT NULL,
            slide_num INTEGER NOT NULL, text TEXT NOT NULL, tokens TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            created_at REAL NOT NULL, title TEXT
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, sources TEXT, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS community_posts (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            username TEXT NOT NULL,
            ap_class TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            unit TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS community_replies (
            id TEXT PRIMARY KEY,
            post_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            username TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_votes (
            user_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (user_id, post_id)
        );
        CREATE TABLE IF NOT EXISTS study_progress (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            activity_type TEXT NOT NULL,
            topic TEXT NOT NULL DEFAULT '',
            score REAL,
            total REAL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            used INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS share_links (
            token TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            created_at REAL NOT NULL
        );
    """)
    # Migrations — add columns if they don't exist yet
    migrations = [
        "ALTER TABLE sessions ADD COLUMN user_id TEXT",
        "ALTER TABLE users ADD COLUMN bio TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE community_posts ADD COLUMN unit TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE community_posts ADD COLUMN upvotes INTEGER NOT NULL DEFAULT 0",
    ]
    for m in migrations:
        try:
            conn.execute(m); conn.commit()
        except Exception:
            pass
    conn.commit()
    # Verify DB is working by counting users
    try:
        count = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()
        import logging
        logging.getLogger("studysearch").info(f"[DB] Verified: {count['c'] if count else 0} users in database")
    except Exception as e:
        import logging
        logging.getLogger("studysearch").error(f"[DB] Verification failed: {e}")
    conn.close()

# ─── AUTH HELPERS ─────────────────────────────────────────────────────────────

def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100_000).hex()

def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    return hash_password(password, salt) == stored_hash

def require_auth(authorization: Optional[str] = None) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    token = authorization[7:]
    conn = get_db()
    row = conn.execute(
        "SELECT u.id, u.username, u.email FROM auth_tokens t "
        "JOIN users u ON t.user_id = u.id "
        "WHERE t.token = ? AND t.expires_at > ?",
        (token, time.time())
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(401, "Invalid or expired session — please log in again")
    return dict(row)

# ─── BM25 ─────────────────────────────────────────────────────────────────────

STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with','is',
    'are','was','were','be','been','have','has','had','do','does','did','will',
    'would','could','should','may','might','this','that','these','those','it',
    'its','they','their','we','our','you','your','i','my','he','she','his',
    'her','from','by','as','if','so','not','no','can','all','also','about','slide','section'
}

def tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

def bm25_score(q_tokens, doc_tokens, df, N, avgdl) -> float:
    dl = len(doc_tokens)
    tf = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1
    score = 0.0
    for term in set(q_tokens):
        if term not in tf: continue
        f   = tf[term]
        dft = df.get(term, 0)
        idf = math.log((N - dft + 0.5) / (dft + 0.5) + 1)
        score += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / avgdl))
    return score

def search_bm25(query: str, chunks: List[dict], top_k=MAX_RESULTS) -> List[dict]:
    if not chunks: return []
    q_tokens = tokenize(query)
    N        = len(chunks)
    df       = {}
    total    = 0
    for c in chunks:
        total += len(c["tokens"])
        seen = set()
        for t in c["tokens"]:
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)
    avgdl = total / N if N else 1
    scored = [(bm25_score(q_tokens, c["tokens"], df, N, avgdl), c) for c in chunks]
    scored = [(s, c) for s, c in scored if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{**c, "score": round(s, 4)} for s, c in scored[:top_k]]

def highlight(text: str, query: str) -> str:
    terms = set(tokenize(query)) - STOPWORDS
    def rep(m):
        w = m.group(0)
        return f"<mark>{w}</mark>" if w.lower() in terms else w
    return re.sub(r'\b\w+\b', rep, text)

# ─── PARSERS ──────────────────────────────────────────────────────────────────
def parse_pdf(data: bytes) -> List[dict]:
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ''
            text = text.strip()
            if text:
                pages.append({'slide_num': i, 'text': text})
        return pages
    except Exception:
        return []
def parse_pptx(data: bytes) -> List[dict]:
    slides = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        sfiles = sorted(
            [n for n in z.namelist() if re.match(r'ppt/slides/slide\d+\.xml', n)],
            key=lambda x: int(re.search(r'\d+', x).group())
        )
        for i, sf in enumerate(sfiles, 1):
            xml = z.read(sf).decode('utf-8', errors='ignore')
            try:
                root  = ET.fromstring(xml)
                texts = [(e.text or '').strip() for e in root.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t')]
            except ET.ParseError:
                texts = re.findall(r'<a:t[^>]*>([^<]+)</a:t>', xml)
            text = ' '.join(t for t in texts if t)
            if text.strip():
                slides.append({'slide_num': i, 'text': text.strip()})
    return slides

def parse_txt(data: bytes) -> List[dict]:
    text  = data.decode('utf-8', errors='ignore')
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 20]
    chunks, buf, idx = [], '', 1
    for p in paras:
        buf += p + '\n'
        if len(buf.split()) >= CHUNK_SIZE_WORDS:
            chunks.append({'slide_num': idx, 'text': buf.strip()})
            buf, idx = '', idx + 1
    if buf.strip(): chunks.append({'slide_num': idx, 'text': buf.strip()})
    return chunks

def parse_docx(data: bytes) -> List[dict]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml   = z.read('word/document.xml').decode('utf-8', errors='ignore')
            texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', xml)
            return parse_txt(' '.join(texts).encode())
    except Exception: return []

def make_chunks(raw, doc_id, session_id, filename) -> List[dict]:
    is_pptx = filename.lower().endswith('.pptx')
    result  = []
    for r in raw:
        label = f"[Slide {r['slide_num']}]" if is_pptx else f"[Section {r['slide_num']}]"
        text  = f"{label} {r['text']}"
        result.append({
            'id': str(uuid.uuid4()), 'doc_id': doc_id,
            'session_id': session_id, 'slide_num': r['slide_num'],
            'text': text, 'tokens': tokenize(text), 'doc_name': filename,
        })
    return result

# ─── HELPERS FOR CHUNK FETCHING ──────────────────────────────────────────────

def _get_session_chunks(conn, sid: str, topic: str = "") -> List[dict]:
    """Fetch chunks for a session, optionally filtered by topic via BM25."""
    rows = conn.execute(
        "SELECT c.id, c.doc_id, c.slide_num, c.text, c.tokens, d.filename "
        "FROM chunks c JOIN documents d ON c.doc_id=d.id WHERE c.session_id=?", (sid,)
    ).fetchall()
    if not rows:
        return []
    chunks = [{"id": r["id"], "doc_id": r["doc_id"], "slide_num": r["slide_num"],
               "text": r["text"], "tokens": json.loads(r["tokens"]), "doc_name": r["filename"]} for r in rows]
    if topic and topic.strip():
        results = search_bm25(topic, chunks, top_k=MAX_RESULTS)
        if results:
            return results
    return chunks[:MAX_RESULTS]

# ─── APP ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    import logging
    logger = logging.getLogger("studysearch")
    if TURSO_DATABASE_URL:
        logger.info(f"[DB] Using Turso remote database: {TURSO_DATABASE_URL[:40]}...")
    else:
        logger.info(f"[DB] Using local SQLite: {DB_PATH}")
    init_db()
    logger.info("[DB] Schema initialized successfully")
    yield

app = FastAPI(title="StudySearch API v3", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

# ─── AUTH ENDPOINTS ───────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/api/auth/signup")
@limiter.limit("5/minute")
def signup(request: Request, body: SignupRequest):
    if len(body.username.strip()) < 2:
        raise HTTPException(400, "Username must be at least 2 characters")
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if "@" not in body.email:
        raise HTTPException(400, "Invalid email address")
    conn = get_db()
    if conn.execute("SELECT id FROM users WHERE email=?", (body.email.lower(),)).fetchone():
        conn.close(); raise HTTPException(400, "An account with that email already exists")
    uid   = str(uuid.uuid4())
    salt  = secrets.token_hex(16)
    pw    = hash_password(body.password, salt)
    conn.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?)",
                 (uid, body.username.strip(), body.email.lower(), pw, salt, time.time(), ''))
    token   = secrets.token_urlsafe(32)
    expires = time.time() + TOKEN_DAYS * 86400
    conn.execute("INSERT INTO auth_tokens VALUES (?,?,?,?)", (token, uid, time.time(), expires))
    # Create a default session for this user
    sid = str(uuid.uuid4()); now = time.time()
    conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", (sid, "My Notebook", now, now, uid))
    conn.commit(); conn.close()
    return {"token": token, "user_id": uid, "username": body.username.strip(), "session_id": sid}

@app.post("/api/auth/login")
@limiter.limit("5/minute")
def login(request: Request, body: LoginRequest):
    conn = get_db()
    row  = conn.execute("SELECT * FROM users WHERE email=?", (body.email.lower(),)).fetchone()
    if not row or not verify_password(body.password, row["salt"], row["password_hash"]):
        conn.close(); raise HTTPException(401, "Incorrect email or password")
    token   = secrets.token_urlsafe(32)
    expires = time.time() + TOKEN_DAYS * 86400
    conn.execute("INSERT INTO auth_tokens VALUES (?,?,?,?)", (token, row["id"], time.time(), expires))
    conn.commit()
    # Get or create user's session
    sess = conn.execute("SELECT id FROM sessions WHERE user_id=? ORDER BY last_used DESC LIMIT 1",
                        (row["id"],)).fetchone()
    if not sess:
        sid = str(uuid.uuid4()); now = time.time()
        conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", (sid, "My Notebook", now, now, row["id"]))
        conn.commit()
    else:
        sid = sess["id"]
    conn.close()
    return {"token": token, "user_id": row["id"], "username": row["username"], "session_id": sid}

@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        conn = get_db()
        conn.execute("DELETE FROM auth_tokens WHERE token=?", (token,))
        conn.commit(); conn.close()
    return {"ok": True}

@app.get("/api/auth/me")
def get_me(authorization: Optional[str] = Header(None)):
    user = require_auth(authorization)
    conn = get_db()
    sess = conn.execute("SELECT id FROM sessions WHERE user_id=? ORDER BY last_used DESC LIMIT 1",
                        (user["id"],)).fetchone()
    conn.close()
    return {"user_id": user["id"], "username": user["username"], "email": user["email"],
            "session_id": sess["id"] if sess else None}

# ─── PASSWORD RESET ──────────────────────────────────────────────────────────

class ResetRequestBody(BaseModel):
    email: str

class ResetBody(BaseModel):
    token: str
    new_password: str

@app.post("/api/auth/reset-request")
def reset_request(body: ResetRequestBody):
    conn = get_db()
    user = conn.execute("SELECT id FROM users WHERE email=?", (body.email.lower(),)).fetchone()
    if not user:
        conn.close()
        # Return success even if user not found to prevent email enumeration
        return {"token": ""}
    token = secrets.token_urlsafe(32)
    now = time.time()
    expires = now + 3600  # 1 hour expiry
    conn.execute("INSERT INTO password_reset_tokens VALUES (?,?,?,?,?)",
                 (token, user["id"], now, expires, 0))
    conn.commit(); conn.close()
    return {"token": token}

@app.post("/api/auth/reset")
def reset_password(body: ResetBody):
    if len(body.new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM password_reset_tokens WHERE token=? AND used=0 AND expires_at > ?",
        (body.token, time.time())
    ).fetchone()
    if not row:
        conn.close(); raise HTTPException(400, "Invalid or expired reset token")
    salt = secrets.token_hex(16)
    pw = hash_password(body.new_password, salt)
    conn.execute("UPDATE users SET password_hash=?, salt=? WHERE id=?", (pw, salt, row["user_id"]))
    conn.execute("UPDATE password_reset_tokens SET used=1 WHERE token=?", (body.token,))
    conn.commit(); conn.close()
    return {"ok": True}

# ─── SESSIONS ─────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    name: Optional[str] = "My Notebook"

@app.post("/api/sessions")
def create_session(body: SessionCreate, authorization: Optional[str] = Header(None)):
    conn = get_db(); sid = str(uuid.uuid4()); now = time.time()
    user_id = None
    if authorization and authorization.startswith("Bearer "):
        try:
            user = require_auth(authorization)
            user_id = user["id"]
        except Exception:
            pass
    conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", (sid, body.name, now, now, user_id))
    conn.commit(); conn.close()
    return {"session_id": sid, "name": body.name}

@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    conn = get_db()
    row  = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    if not row: raise HTTPException(404, "Session not found")
    docs = conn.execute(
        "SELECT id, filename, file_type, slide_count, uploaded_at FROM documents WHERE session_id=? ORDER BY uploaded_at",
        (sid,)
    ).fetchall()
    conn.execute("UPDATE sessions SET last_used=? WHERE id=?", (time.time(), sid))
    conn.commit(); conn.close()
    return {"session_id": row["id"], "name": row["name"], "documents": [dict(d) for d in docs]}

# ─── DOCUMENTS ────────────────────────────────────────────────────────────────

@app.post("/api/sessions/{sid}/documents")
async def upload_document(sid: str, file: UploadFile = File(...)):
    conn = get_db()
    if not conn.execute("SELECT id FROM sessions WHERE id=?", (sid,)).fetchone():
        conn.close(); raise HTTPException(404, "Session not found")
    data = await file.read()
    ext  = Path(file.filename).suffix.lower().lstrip('.')
    if   ext == 'pptx':         raw = parse_pptx(data)
    elif ext in ('txt', 'md'):  raw = parse_txt(data)
    elif ext == 'docx':         raw = parse_docx(data)
    elif ext == 'pdf':          raw = parse_pdf(data)
    else:                       raise HTTPException(400, f"Unsupported: {ext}")
    if not raw: raise HTTPException(400, "No text extracted from file")
    doc_id = str(uuid.uuid4())
    conn.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)",
                 (doc_id, sid, file.filename, ext, len(raw), time.time()))
    for c in make_chunks(raw, doc_id, sid, file.filename):
        conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                     (c['id'], c['doc_id'], c['session_id'],
                      c['slide_num'], c['text'], json.dumps(c['tokens'])))
    conn.commit(); conn.close()
    return {"doc_id": doc_id, "filename": file.filename, "chunks": len(raw)}

@app.delete("/api/sessions/{sid}/documents/{doc_id}")
def delete_document(sid: str, doc_id: str):
    conn = get_db()
    conn.execute("DELETE FROM chunks WHERE doc_id=? AND session_id=?", (doc_id, sid))
    conn.execute("DELETE FROM documents WHERE id=? AND session_id=?", (doc_id, sid))
    conn.commit(); conn.close()
    return {"deleted": True}

# ─── BM25 SEARCH — NO AI ─────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str

@app.post("/api/sessions/{sid}/search")
def bm25_search_route(sid: str, body: SearchRequest):
    conn = get_db()
    rows = conn.execute(
        "SELECT c.id, c.doc_id, c.slide_num, c.text, c.tokens, d.filename "
        "FROM chunks c JOIN documents d ON c.doc_id=d.id WHERE c.session_id=?", (sid,)
    ).fetchall()
    conn.close()
    if not rows: raise HTTPException(400, "No documents uploaded yet")
    chunks  = [{"id": r["id"], "doc_id": r["doc_id"], "slide_num": r["slide_num"],
                "text": r["text"], "tokens": json.loads(r["tokens"]), "doc_name": r["filename"]} for r in rows]
    results = search_bm25(body.query, chunks)
    return {
        "query":   body.query,
        "results": [{"id": r["id"], "doc_name": r["doc_name"], "slide_num": r["slide_num"],
                     "text": r["text"], "highlighted": highlight(r["text"], body.query),
                     "score": r["score"]} for r in results]
    }

# ─── SMART SEARCH (QUERY EXPANSION) ─────────────────────────────────────────

@app.post("/api/sessions/{sid}/smart-search")
@limiter.limit("15/minute")
async def smart_search_route(request: Request, sid: str, body: SearchRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured on server")
    conn = get_db()
    rows = conn.execute(
        "SELECT c.id, c.doc_id, c.slide_num, c.text, c.tokens, d.filename "
        "FROM chunks c JOIN documents d ON c.doc_id=d.id WHERE c.session_id=?", (sid,)
    ).fetchall()
    if not rows:
        conn.close(); raise HTTPException(400, "No documents uploaded yet")
    conn.close()

    # Use Claude to expand the query
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": CHAT_MODEL, "max_tokens": 150,
                  "system": "Given the search query, generate 3-5 related search terms or synonyms. Return ONLY a comma-separated list of terms, nothing else.",
                  "messages": [{"role": "user", "content": body.query}]}
        )

    expanded_terms = ""
    try:
        data = resp.json()
        if "content" in data and data["content"]:
            expanded_terms = data["content"][0]["text"]
    except Exception:
        pass

    combined_query = body.query
    if expanded_terms:
        combined_query = f"{body.query} {expanded_terms.replace(',', ' ')}"

    chunks = [{"id": r["id"], "doc_id": r["doc_id"], "slide_num": r["slide_num"],
               "text": r["text"], "tokens": json.loads(r["tokens"]), "doc_name": r["filename"]} for r in rows]
    results = search_bm25(combined_query, chunks)
    return {
        "query": body.query,
        "expanded_terms": expanded_terms,
        "results": [{"id": r["id"], "doc_name": r["doc_name"], "slide_num": r["slide_num"],
                     "text": r["text"], "highlighted": highlight(r["text"], body.query),
                     "score": r["score"]} for r in results]
    }

# ─── AI EXPLAIN — OPTIONAL ────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    query: str
    chunk_ids: List[str]
    conversation_id: Optional[str] = None

@app.post("/api/sessions/{sid}/explain")
@limiter.limit("15/minute")
async def explain_route(request: Request, sid: str, body: ExplainRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured on server")
    conn = get_db()
    ph   = ','.join('?' * len(body.chunk_ids))
    rows = conn.execute(
        f"SELECT c.id, c.slide_num, c.text, d.filename FROM chunks c "
        f"JOIN documents d ON c.doc_id=d.id WHERE c.id IN ({ph}) AND c.session_id=?",
        (*body.chunk_ids, sid)
    ).fetchall()
    if not rows: conn.close(); raise HTTPException(404, "Chunks not found")

    conv_id = body.conversation_id
    if not conv_id:
        conv_id = str(uuid.uuid4())
        conn.execute("INSERT INTO conversations VALUES (?,?,?,?)",
                     (conv_id, sid, time.time(), body.query[:60]))

    history = [{"role": r["role"], "content": r["content"]} for r in conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at", (conv_id,)
    ).fetchall()]

    context = "\n\n".join(f"[{r['filename']} — Slide {r['slide_num']}]\n{r['text']}" for r in rows)
    sources = [{"doc": r["filename"], "slide": r["slide_num"]} for r in rows]

    messages = history + [{"role": "user",
        "content": f"Notes from slides:\n\n{context}\n\n---\nExplain: {body.query}"}]

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": CHAT_MODEL, "max_tokens": 1024,
                  "system": """You are StudySearch, a smart study assistant. Your job is to directly answer the student's question using ONLY the information in the provided slide notes.

Rules:
- Lead with a clear, direct answer to the question in the very first sentence — like a Google AI overview.
- Then give 2-4 sentences of the most important supporting detail.
- IGNORE any slide content that is not relevant to the question — do not mention it.
- Write in plain, clear prose. No bullet points, no headers, no lists, no markdown.
- If the notes don't contain enough information to answer fully, say so briefly at the end.
- For follow-up questions, use the conversation history for context.""",
                  "messages": messages}
        )

    data = resp.json()
    if "error" in data: conn.close(); raise HTTPException(500, data["error"]["message"])
    answer = data["content"][0]["text"]
    now    = time.time()
    conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)",
                 (str(uuid.uuid4()), conv_id, "user", body.query, "[]", now))
    conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)",
                 (str(uuid.uuid4()), conv_id, "assistant", answer, json.dumps(sources), now + 0.001))
    conn.commit(); conn.close()
    return {"conversation_id": conv_id, "answer": answer, "sources": sources}

# ─── SSE STREAMING AI EXPLAIN ────────────────────────────────────────────────

@app.post("/api/sessions/{sid}/explain-stream")
@limiter.limit("15/minute")
async def explain_stream_route(request: Request, sid: str, body: ExplainRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured on server")
    conn = get_db()
    ph   = ','.join('?' * len(body.chunk_ids))
    rows = conn.execute(
        f"SELECT c.id, c.slide_num, c.text, d.filename FROM chunks c "
        f"JOIN documents d ON c.doc_id=d.id WHERE c.id IN ({ph}) AND c.session_id=?",
        (*body.chunk_ids, sid)
    ).fetchall()
    if not rows:
        conn.close(); raise HTTPException(404, "Chunks not found")

    conv_id = body.conversation_id
    if not conv_id:
        conv_id = str(uuid.uuid4())
        conn.execute("INSERT INTO conversations VALUES (?,?,?,?)",
                     (conv_id, sid, time.time(), body.query[:60]))
        conn.commit()

    history = [{"role": r["role"], "content": r["content"]} for r in conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at", (conv_id,)
    ).fetchall()]

    context = "\n\n".join(f"[{r['filename']} — Slide {r['slide_num']}]\n{r['text']}" for r in rows)
    sources = [{"doc": r["filename"], "slide": r["slide_num"]} for r in rows]

    messages = history + [{"role": "user",
        "content": f"Notes from slides:\n\n{context}\n\n---\nExplain: {body.query}"}]

    conn.close()

    async def event_generator():
        full_text = ""
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": CHAT_MODEL, "max_tokens": 1024, "stream": True,
                      "system": """You are StudySearch, a smart study assistant. Your job is to directly answer the student's question using ONLY the information in the provided slide notes.

Rules:
- Lead with a clear, direct answer to the question in the very first sentence — like a Google AI overview.
- Then give 2-4 sentences of the most important supporting detail.
- IGNORE any slide content that is not relevant to the question — do not mention it.
- Write in plain, clear prose. No bullet points, no headers, no lists, no markdown.
- If the notes don't contain enough information to answer fully, say so briefly at the end.
- For follow-up questions, use the conversation history for context.""",
                      "messages": messages}
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    event_type = event.get("type", "")
                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            token = delta.get("text", "")
                            full_text += token
                            yield {"data": json.dumps({"token": token})}
                    elif event_type == "message_stop":
                        break

        # Save to DB
        db = get_db()
        now = time.time()
        db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)",
                   (str(uuid.uuid4()), conv_id, "user", body.query, "[]", now))
        db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)",
                   (str(uuid.uuid4()), conv_id, "assistant", full_text, json.dumps(sources), now + 0.001))
        db.commit(); db.close()

        yield {"data": json.dumps({"done": True, "answer": full_text, "conversation_id": conv_id, "sources": sources})}

    return EventSourceResponse(event_generator())

# ─── FLASHCARD GENERATION ────────────────────────────────────────────────────

class FlashcardRequest(BaseModel):
    topic: str = ""
    count: int = 10

@app.post("/api/sessions/{sid}/flashcards")
@limiter.limit("15/minute")
async def flashcard_route(request: Request, sid: str, body: FlashcardRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured on server")
    conn = get_db()
    chunks = _get_session_chunks(conn, sid, body.topic)
    conn.close()
    if not chunks:
        raise HTTPException(400, "No documents uploaded yet")

    context = "\n\n".join(c["text"] for c in chunks)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": CHAT_MODEL, "max_tokens": 2048,
                  "system": f"Generate exactly {body.count} flashcards from the provided study material. Return ONLY valid JSON array: [{{\"front\": \"question\", \"back\": \"answer\"}}]. No markdown, no explanation.",
                  "messages": [{"role": "user", "content": f"Study material:\n\n{context}"}]}
        )

    data = resp.json()
    if "error" in data:
        raise HTTPException(500, data["error"]["message"])

    raw_text = data["content"][0]["text"]
    try:
        flashcards = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to extract JSON array from the response
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            try:
                flashcards = json.loads(match.group())
            except json.JSONDecodeError:
                raise HTTPException(500, "Failed to parse flashcard response from AI")
        else:
            raise HTTPException(500, "Failed to parse flashcard response from AI")

    return {"flashcards": flashcards, "topic": body.topic}

# ─── QUIZ GENERATION ─────────────────────────────────────────────────────────

class QuizRequest(BaseModel):
    topic: str = ""
    count: int = 5

@app.post("/api/sessions/{sid}/quiz")
@limiter.limit("15/minute")
async def quiz_route(request: Request, sid: str, body: QuizRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured on server")
    conn = get_db()
    chunks = _get_session_chunks(conn, sid, body.topic)
    conn.close()
    if not chunks:
        raise HTTPException(400, "No documents uploaded yet")

    context = "\n\n".join(c["text"] for c in chunks)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": CHAT_MODEL, "max_tokens": 2048,
                  "system": f"Generate exactly {body.count} multiple choice questions from the study material. Return ONLY valid JSON array: [{{\"question\": \"...\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"correct\": 0, \"explanation\": \"...\"}}]. The correct field is the 0-based index. No markdown.",
                  "messages": [{"role": "user", "content": f"Study material:\n\n{context}"}]}
        )

    data = resp.json()
    if "error" in data:
        raise HTTPException(500, data["error"]["message"])

    raw_text = data["content"][0]["text"]
    try:
        questions = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            try:
                questions = json.loads(match.group())
            except json.JSONDecodeError:
                raise HTTPException(500, "Failed to parse quiz response from AI")
        else:
            raise HTTPException(500, "Failed to parse quiz response from AI")

    return {"questions": questions, "topic": body.topic}

# ─── STUDY PROGRESS ──────────────────────────────────────────────────────────

class ProgressLog(BaseModel):
    activity_type: str
    topic: str = ""
    score: Optional[float] = None
    total: Optional[float] = None

@app.post("/api/sessions/{sid}/progress")
def log_progress(sid: str, body: ProgressLog, authorization: Optional[str] = Header(None)):
    user = require_auth(authorization)
    conn = get_db()
    if not conn.execute("SELECT id FROM sessions WHERE id=?", (sid,)).fetchone():
        conn.close(); raise HTTPException(404, "Session not found")
    pid = str(uuid.uuid4())
    now = time.time()
    conn.execute("INSERT INTO study_progress VALUES (?,?,?,?,?,?,?,?)",
                 (pid, user["id"], sid, body.activity_type, body.topic, body.score, body.total, now))
    conn.commit(); conn.close()
    return {"id": pid}

@app.get("/api/sessions/{sid}/progress")
def get_progress(sid: str, authorization: Optional[str] = Header(None)):
    user = require_auth(authorization)
    conn = get_db()
    activities = conn.execute(
        "SELECT * FROM study_progress WHERE user_id=? AND session_id=? ORDER BY created_at DESC LIMIT 50",
        (user["id"], sid)
    ).fetchall()

    # Calculate stats
    all_activities = conn.execute(
        "SELECT activity_type, score, total, created_at FROM study_progress WHERE user_id=? AND session_id=?",
        (user["id"], sid)
    ).fetchall()
    conn.close()

    total_searches = sum(1 for a in all_activities if a["activity_type"] == "search")
    total_quizzes = sum(1 for a in all_activities if a["activity_type"] == "quiz")
    total_flashcards = sum(1 for a in all_activities if a["activity_type"] == "flashcard")

    quiz_scores = [(a["score"], a["total"]) for a in all_activities
                   if a["activity_type"] == "quiz" and a["score"] is not None and a["total"] is not None and a["total"] > 0]
    avg_quiz_score = 0.0
    if quiz_scores:
        avg_quiz_score = round(sum(s / t for s, t in quiz_scores) / len(quiz_scores) * 100, 1)

    # Calculate streak days: consecutive days with activity looking back from today
    import datetime
    today = datetime.date.today()
    activity_dates = set()
    for a in all_activities:
        d = datetime.date.fromtimestamp(a["created_at"])
        activity_dates.add(d)

    streak_days = 0
    check_date = today
    while check_date in activity_dates:
        streak_days += 1
        check_date -= datetime.timedelta(days=1)

    return {
        "activities": [dict(a) for a in activities],
        "stats": {
            "total_searches": total_searches,
            "total_quizzes": total_quizzes,
            "total_flashcards": total_flashcards,
            "avg_quiz_score": avg_quiz_score,
            "streak_days": streak_days,
        }
    }

# ─── CONVERSATIONS ────────────────────────────────────────────────────────────

@app.get("/api/sessions/{sid}/conversations")
def list_conversations(sid: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, created_at FROM conversations WHERE session_id=? ORDER BY created_at DESC", (sid,)
    ).fetchall()
    conn.close(); return [dict(r) for r in rows]

@app.get("/api/sessions/{sid}/conversations/{conv_id}")
def get_conversation(sid: str, conv_id: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, sources FROM messages WHERE conversation_id=? ORDER BY created_at", (conv_id,)
    ).fetchall()
    conn.close()
    return [{**dict(r), "sources": json.loads(r["sources"] or "[]")} for r in rows]

# ─── SUGGESTIONS — NO AI ─────────────────────────────────────────────────────

@app.get("/api/sessions/{sid}/suggestions")
def get_suggestions(sid: str):
    conn  = get_db()
    rows  = conn.execute("SELECT tokens FROM chunks WHERE session_id=? LIMIT 30", (sid,)).fetchall()
    conn.close()
    if not rows: return {"suggestions": []}
    tf = {}
    for row in rows:
        for t in json.loads(row["tokens"]):
            if t not in STOPWORDS and len(t) > 4:
                tf[t] = tf.get(t, 0) + 1
    top = sorted(tf.items(), key=lambda x: x[1], reverse=True)[:8]
    templates = ["What is {}?", "How does {} work?", "Explain {}",
                 "What are the key points about {}?", "Summarize {} from the notes"]
    random.seed(42)
    return {"suggestions": [random.choice(templates).format(t) for t, _ in top[:5]]}

# ─── COMMUNITY ───────────────────────────────────────────────────────────────

@app.get("/api/community/classes")
def list_ap_classes():
    return {"classes": AP_CLASSES}

class CommunityPostCreate(BaseModel):
    ap_class: str
    title: str
    body: str
    category: str = "general"  # general, study-guide, test, notes, quiz
    unit: str = ""  # e.g. "Unit 1: Chemistry of Life"

@app.get("/api/community/posts")
def list_community_posts(ap_class: str = "", category: str = "", unit: str = "", user_id: str = ""):
    conn = get_db()
    query = "SELECT * FROM community_posts WHERE 1=1"
    params = []
    if ap_class:
        query += " AND ap_class = ?"
        params.append(ap_class)
    if category:
        query += " AND category = ?"
        params.append(category)
    if unit:
        query += " AND unit = ?"
        params.append(unit)
    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    query += " ORDER BY created_at DESC LIMIT 100"
    rows = conn.execute(query, params).fetchall()
    posts = []
    for r in rows:
        d = dict(r)
        d["reply_count"] = conn.execute(
            "SELECT COUNT(*) as c FROM community_replies WHERE post_id=?", (r["id"],)
        ).fetchone()["c"]
        posts.append(d)
    conn.close()
    return {"posts": posts}

@app.post("/api/community/posts")
def create_community_post(body: CommunityPostCreate, authorization: Optional[str] = Header(None)):
    user = require_auth(authorization)
    if body.ap_class not in AP_CLASSES:
        raise HTTPException(400, "Invalid AP class")
    if not body.title.strip() or not body.body.strip():
        raise HTTPException(400, "Title and body are required")
    if body.category not in ("general", "study-guide", "test", "notes", "quiz"):
        raise HTTPException(400, "Invalid category")
    conn = get_db()
    pid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO community_posts VALUES (?,?,?,?,?,?,?,?,?,?)",
        (pid, user["id"], user["username"], body.ap_class, body.title.strip(),
         body.body.strip(), body.category, body.unit, time.time(), 0)
    )
    conn.commit(); conn.close()
    return {"id": pid}

@app.get("/api/community/posts/{post_id}")
def get_community_post(post_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM community_posts WHERE id=?", (post_id,)).fetchone()
    if not row:
        conn.close(); raise HTTPException(404, "Post not found")
    replies = conn.execute(
        "SELECT * FROM community_replies WHERE post_id=? ORDER BY created_at ASC", (post_id,)
    ).fetchall()
    conn.close()
    return {"post": dict(row), "replies": [dict(r) for r in replies]}

class CommunityReplyCreate(BaseModel):
    body: str

@app.post("/api/community/posts/{post_id}/replies")
def create_community_reply(post_id: str, body: CommunityReplyCreate, authorization: Optional[str] = Header(None)):
    user = require_auth(authorization)
    if not body.body.strip():
        raise HTTPException(400, "Reply cannot be empty")
    conn = get_db()
    if not conn.execute("SELECT id FROM community_posts WHERE id=?", (post_id,)).fetchone():
        conn.close(); raise HTTPException(404, "Post not found")
    rid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO community_replies VALUES (?,?,?,?,?,?)",
        (rid, post_id, user["id"], user["username"], body.body.strip(), time.time())
    )
    conn.commit(); conn.close()
    return {"id": rid}

@app.delete("/api/community/posts/{post_id}")
def delete_community_post(post_id: str, authorization: Optional[str] = Header(None)):
    user = require_auth(authorization)
    conn = get_db()
    row = conn.execute("SELECT user_id FROM community_posts WHERE id=?", (post_id,)).fetchone()
    if not row:
        conn.close(); raise HTTPException(404, "Post not found")
    if row["user_id"] != user["id"]:
        conn.close(); raise HTTPException(403, "You can only delete your own posts")
    conn.execute("DELETE FROM community_replies WHERE post_id=?", (post_id,))
    conn.execute("DELETE FROM community_posts WHERE id=?", (post_id,))
    conn.commit(); conn.close()
    return {"deleted": True}

# ─── PROFILES ────────────────────────────────────────────────────────────────

@app.get("/api/users/{user_id}/profile")
def get_user_profile(user_id: str):
    conn = get_db()
    user = conn.execute("SELECT id, username, email, bio, created_at FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close(); raise HTTPException(404, "User not found")
    post_count = conn.execute("SELECT COUNT(*) as c FROM community_posts WHERE user_id=?", (user_id,)).fetchone()["c"]
    reply_count = conn.execute("SELECT COUNT(*) as c FROM community_replies WHERE user_id=?", (user_id,)).fetchone()["c"]
    recent_posts = conn.execute(
        "SELECT * FROM community_posts WHERE user_id=? ORDER BY created_at DESC LIMIT 20", (user_id,)
    ).fetchall()
    posts = []
    for r in recent_posts:
        d = dict(r)
        d["reply_count"] = conn.execute(
            "SELECT COUNT(*) as c FROM community_replies WHERE post_id=?", (r["id"],)
        ).fetchone()["c"]
        posts.append(d)
    conn.close()
    return {
        "user": {"id": user["id"], "username": user["username"], "bio": user["bio"], "created_at": user["created_at"]},
        "stats": {"posts": post_count, "replies": reply_count},
        "recent_posts": posts,
    }

class BioUpdate(BaseModel):
    bio: str

@app.put("/api/users/me/bio")
def update_bio(body: BioUpdate, authorization: Optional[str] = Header(None)):
    user = require_auth(authorization)
    bio = body.bio.strip()[:300]  # max 300 chars
    conn = get_db()
    conn.execute("UPDATE users SET bio=? WHERE id=?", (bio, user["id"]))
    conn.commit(); conn.close()
    return {"bio": bio}

# ─── UNITS ───────────────────────────────────────────────────────────────────

@app.get("/api/community/units/{ap_class}")
def get_units_for_class(ap_class: str):
    try:
        from ap_units import AP_UNITS
        units = AP_UNITS.get(ap_class, [])
    except ImportError:
        units = []
    return {"ap_class": ap_class, "units": units}

# ─── UPVOTES (TOGGLE) ───────────────────────────────────────────────────────

@app.post("/api/community/posts/{post_id}/upvote")
def upvote_post(post_id: str, authorization: Optional[str] = Header(None)):
    user = require_auth(authorization)
    conn = get_db()
    row = conn.execute("SELECT id, upvotes FROM community_posts WHERE id=?", (post_id,)).fetchone()
    if not row:
        conn.close(); raise HTTPException(404, "Post not found")

    existing_vote = conn.execute(
        "SELECT user_id FROM user_votes WHERE user_id=? AND post_id=?",
        (user["id"], post_id)
    ).fetchone()

    if existing_vote:
        # Already voted — remove vote and decrement
        conn.execute("DELETE FROM user_votes WHERE user_id=? AND post_id=?", (user["id"], post_id))
        new_count = max((row["upvotes"] or 0) - 1, 0)
        conn.execute("UPDATE community_posts SET upvotes=? WHERE id=?", (new_count, post_id))
        conn.commit(); conn.close()
        return {"upvotes": new_count, "voted": False}
    else:
        # Not voted — add vote and increment
        conn.execute("INSERT INTO user_votes VALUES (?,?,?)", (user["id"], post_id, time.time()))
        new_count = (row["upvotes"] or 0) + 1
        conn.execute("UPDATE community_posts SET upvotes=? WHERE id=?", (new_count, post_id))
        conn.commit(); conn.close()
        return {"upvotes": new_count, "voted": True}

# ─── SHARING NOTEBOOKS ──────────────────────────────────────────────────────

@app.post("/api/sessions/{sid}/share")
def share_session(sid: str, authorization: Optional[str] = Header(None)):
    user = require_auth(authorization)
    conn = get_db()
    sess = conn.execute("SELECT id FROM sessions WHERE id=?", (sid,)).fetchone()
    if not sess:
        conn.close(); raise HTTPException(404, "Session not found")
    token = secrets.token_urlsafe(16)
    conn.execute("INSERT INTO share_links VALUES (?,?,?,?)", (token, sid, user["id"], time.time()))
    conn.commit(); conn.close()
    return {"share_url": f"/shared/{token}", "token": token}

@app.get("/api/shared/{token}")
def get_shared_session(token: str):
    conn = get_db()
    link = conn.execute("SELECT * FROM share_links WHERE token=?", (token,)).fetchone()
    if not link:
        conn.close(); raise HTTPException(404, "Share link not found")
    sess = conn.execute("SELECT id, name FROM sessions WHERE id=?", (link["session_id"],)).fetchone()
    if not sess:
        conn.close(); raise HTTPException(404, "Session not found")
    docs = conn.execute(
        "SELECT id, filename, file_type, slide_count, uploaded_at FROM documents WHERE session_id=? ORDER BY uploaded_at",
        (link["session_id"],)
    ).fetchall()
    user = conn.execute("SELECT username FROM users WHERE id=?", (link["user_id"],)).fetchone()
    conn.close()
    return {
        "session_name": sess["name"],
        "documents": [dict(d) for d in docs],
        "shared_by": user["username"] if user else "Unknown"
    }

# ─── HEALTH & LANDING ───────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    db_mode = "turso" if TURSO_DATABASE_URL else "local-sqlite"
    return {"status": "ok", "version": "4.1.0", "search": "BM25", "ai": "optional", "db": db_mode}

@app.get("/")
def landing_page():
    """Serve landing page if no auth, otherwise redirect to app"""
    return FileResponse(Path(__file__).parent / "landing.html", media_type="text/html")

@app.get("/app")
def serve_app():
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")

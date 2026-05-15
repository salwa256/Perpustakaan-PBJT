# =========================
# IMPORT
# =========================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from datetime import date, datetime
from pathlib import Path
from collections import Counter

import pandas as pd
import psycopg2
import mysql.connector
import shutil
import uuid
import os

from psycopg2.extras import RealDictCursor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# APP
# =========================

app = FastAPI(title="Sistem Perpustakaan API")

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# DATABASE
# =========================

DATABASE_URL = "postgresql://postgres.xlndpiygnorrgtupavmg:perpusbackend@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres?sslmode=require"

def get_db():

    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor
    )

# =========================
# MODELS
# =========================

class LoginModel(BaseModel):
    username: str
    password: str


class BookModel(BaseModel):
    id: str
    title: str
    author: str
    category: str


class BorrowModel(BaseModel):
    user_id: int
    book_id: str
    borrower_name: str
    member_id: str
    phone: str
    due_date: str


class ReturnModel(BaseModel):
    book_id: str


class ScanModel(BaseModel):
    image_base64: str


class MemberModel(BaseModel):
    member_code: str
    name: str
    nim: str
    major: str
    phone: str
    address: str

# =========================
# DATASET
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset_buku.csv"

IMAGES_DIR = BASE_DIR / "static/images"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# =========================
# LOAD DATASET
# =========================

if DATASET_PATH.exists():
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.DataFrame(columns=[
        "judul",
        "pengarang",
        "klasifikasi",
        "status",
        "genre_utama",
        "subgenre",
        "genre",
        "content",
        "image_url"
    ])

df = df.fillna("")

def normalize_filename(text):

    return text.lower().strip().replace(" ", "_")


def find_existing_image(judul):

    base_name = normalize_filename(judul)

    for ext in ["jpg", "jpeg", "png", "webp"]:

        file_path = IMAGES_DIR / f"{base_name}.{ext}"

        if file_path.exists():
            return f"/images/{base_name}.{ext}"

    return "/images/no_cover.png"

df["image_url"] = df["judul"].apply(find_existing_image)

# =========================
# TFIDF
# =========================

vectorizer = TfidfVectorizer(stop_words=None)

tfidf_matrix = None

def update_tfidf():

    global tfidf_matrix

    if len(df) == 0:
        tfidf_matrix = None
        return

    tfidf_matrix = vectorizer.fit_transform(df["content"])

update_tfidf()

search_log = Counter()

# =========================
# AUTH
# =========================

@app.post("/login")
def login(data: LoginModel):

    db = get_db()

    cur = db.cursor()

    cur.execute(
        """
        SELECT *
        FROM users
        WHERE username=%s
        AND password=%s
        """,
        (data.username, data.password)
    )

    user = cur.fetchone()

    db.close()

    if user:

        return {
            "user_id": user["id"],
            "username": user["username"],
            "role": user.get("role", "petugas")
        }

    raise HTTPException(
        status_code=401,
        detail="Username atau password salah"
    )

# =========================
# BOOKS
# =========================

@app.get("/books")
def get_books():

    db = get_db()

    cur = db.cursor()

    cur.execute("""
        SELECT *
        FROM books
        ORDER BY id
    """)

    books = cur.fetchall()

    db.close()

    return books


@app.get("/books/available")
def get_available_books():

    db = get_db()

    cur = db.cursor()

    cur.execute("""
        SELECT *
        FROM books
        WHERE status='tersedia'
        ORDER BY title
    """)

    books = cur.fetchall()

    db.close()

    return books


@app.get("/books/search")
def search_books(q: str = ""):

    db = get_db()

    cur = db.cursor()

    like = f"%{q}%"

    cur.execute("""
        SELECT *
        FROM books
        WHERE title ILIKE %s
        OR author ILIKE %s
        OR id ILIKE %s
    """, (like, like, like))

    books = cur.fetchall()

    db.close()

    return books

@app.get("/members/search")
def search_members(q: str = ""):

    db = get_db()

    cur = db.cursor()

    keyword = f"%{q}%"

    cur.execute("""
        SELECT *
        FROM members
        WHERE
            name ILIKE %s
            OR nim ILIKE %s
            OR member_code ILIKE %s
        ORDER BY id DESC
    """, (
        keyword,
        keyword,
        keyword
    ))

    members = cur.fetchall()

    db.close()

    return members

# =========================
# RECOMMENDATION
# =========================

class RequestRekomendasi(BaseModel):
    genre: str
    subgenre: str

@app.post("/recommend")
def recommend(data: RequestRekomendasi):

    if tfidf_matrix is None:

        return {
            "recommendations": [],
            "popular": []
        }

    query_text = f"{data.genre} {data.subgenre}".strip()

    query_vec = vectorizer.transform([query_text])

    similarity = cosine_similarity(
        query_vec,
        tfidf_matrix
    )[0]

    top_idx = similarity.argsort()[-5:][::-1]

    hasil = []

    for i in top_idx:

        row = df.iloc[i]

        hasil.append({
            "judul": row["judul"],
            "pengarang": row["pengarang"],
            "klasifikasi": row["klasifikasi"],
            "image_url": row["image_url"],
            "description": f"Buku karya {row['pengarang']}"
        })

        search_log[row["judul"]] += 1

    return {
        "recommendations": hasil,
        "popular": [j for j, _ in search_log.most_common(5)]
    }

# =========================
# ADD BOOK
# =========================

@app.post("/books")
def add_book(book: BookModel):

    db = get_db()

    cur = db.cursor()

    try:

        cur.execute("""
            INSERT INTO books (
                id,
                title,
                author,
                category,
                status
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                'tersedia'
            )
        """, (
            book.id,
            book.title,
            book.author,
            book.category
        ))

        db.commit()

    except mysql.connector.IntegrityError:

        db.close()

        raise HTTPException(
            status_code=400,
            detail="ID buku sudah ada"
        )

    db.close()

    return {
        "message":
        f"Buku '{book.title}' berhasil ditambahkan"
    }

# =========================
# DELETE BOOK
# =========================
@app.delete("/books/{book_id:path}")
def delete_book(book_id: str):

    db = get_db()
    cur = db.cursor()

    try:

        cur.execute("""
    SELECT *
    FROM books
    WHERE TRIM(id)=TRIM(%s)
""",(book_id,))

        book = cur.fetchone()

        if not book:
            raise HTTPException(
                status_code=404,
                detail="Buku tidak ditemukan"
            )

        cur.execute("""
            SELECT *
            FROM loans
            WHERE book_id=%s
            AND status IN(
                'dipinjam',
                'terlambat'
            )
        """,(book_id,))

        activeLoan = cur.fetchone()

        if activeLoan:
            raise HTTPException(
                status_code=400,
                detail="Buku masih dipinjam"
            )

        cur.execute("""
    DELETE
    FROM books
    WHERE TRIM(id)=TRIM(%s)
""",(book_id,))

        db.commit()

        return {
            "message":"Buku berhasil dihapus"
        }

    finally:
        db.close()
        
# =========================
# LOANS
# =========================

@app.get("/loans")
def get_loans():

    db = get_db()

    cur = db.cursor()

    cur.execute("""
SELECT
    l.*,
    b.title as book_title,
    b.author as book_author
FROM loans l
LEFT JOIN books b
ON l.book_id = b.id
ORDER BY l.borrow_date DESC
""")

    loans = cur.fetchall()

    db.close()

    for loan in loans:

        for k, v in loan.items():

            if isinstance(v, (date, datetime)):
                loan[k] = str(v)

    return loans


@app.get("/loans/active")
def get_active_loans():

    db = get_db()

    cur = db.cursor()

    cur.execute("""
        SELECT
            l.*,
            b.title as book_title,
            b.author as book_author
        FROM loans l
LEFT JOIN books b
ON l.book_id = b.id
        WHERE l.status IN ('dipinjam','terlambat')
        ORDER BY l.due_date ASC
    """)

    loans = cur.fetchall()

    db.close()

    for loan in loans:

        for k, v in loan.items():

            if isinstance(v, (date, datetime)):
                loan[k] = str(v)

    return loans

# =========================
# STATS
# =========================

@app.get("/stats")
def get_stats():

    db = get_db()

    cur = db.cursor()

    cur.execute("""
        SELECT COUNT(*) as total
        FROM books
    """)

    total = cur.fetchone()["total"]

    cur.execute("""
    SELECT COUNT(*) as c
    FROM loans
    WHERE status IN ('dipinjam', 'terlambat')
    """)

    borrowed = cur.fetchone()["c"]

    cur.execute("""
        SELECT COUNT(*) as c
        FROM books
        WHERE status='tersedia'
    """)

    available = cur.fetchone()["c"]

    cur.execute("""
        UPDATE loans
        SET status='terlambat'
        WHERE due_date < CURRENT_DATE
        AND status='dipinjam'
    """)

    db.commit()

    cur.execute("""
        SELECT COUNT(*) as c
        FROM loans
        WHERE status='terlambat'
    """)

    overdue = cur.fetchone()["c"]

    cur.execute("""
        SELECT COUNT(*) as c
        FROM members
    """)

    members = cur.fetchone()["c"]

    db.close()

    return {
        "total": total,
        "borrowed": borrowed,
        "available": available,
        "overdue": overdue,
        "members": members
    }

# =========================
# MEMBERS
# =========================

@app.get("/members")
def get_members():

    db = get_db()

    cur = db.cursor()

    cur.execute("""
        SELECT *
        FROM members
        ORDER BY id DESC
    """)

    members = cur.fetchall()

    db.close()

    return members


@app.post("/members")
def add_member(member: MemberModel):

    db = get_db()

    cur = db.cursor()

    try:

        cur.execute("""
            INSERT INTO members (
                member_code,
                name,
                nim,
                major,
                phone,
                address
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
        """, (
            member.member_code,
            member.name,
            member.nim,
            member.major,
            member.phone,
            member.address
        ))

        db.commit()

        return {
            "message": "Member berhasil ditambahkan"
        }

    except Exception as e:

        db.rollback()

        print("MEMBER ERROR:")
        print(e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        db.close()
# =========================
# MEMBER ACTIVE LOANS
# =========================

@app.get("/member-loans/{nim}")
def get_member_loans(nim: str):

    db = get_db()

    cur = db.cursor()

    cur.execute("""
        SELECT
            l.*,
            b.title as book_title,
            b.author as book_author
        FROM loans l
LEFT JOIN books b
ON l.book_id = b.id
        WHERE l.nim = %s
        AND l.status IN ('dipinjam', 'terlambat')
        ORDER BY l.borrow_date DESC
    """, (nim,))

    loans = cur.fetchall()

    db.close()

    for loan in loans:

        for k, v in loan.items():

            if isinstance(v, (date, datetime)):
                loan[k] = str(v)

    return loans

@app.post("/loans/borrow")
def borrow_book(data: BorrowModel):

    db = get_db()

    cur = db.cursor()

    try:

        # cek member
        cur.execute("""
            SELECT *
            FROM members
            WHERE member_code = %s
        """, (data.member_id,))

        member = cur.fetchone()

        if not member:

            raise HTTPException(
                status_code=404,
                detail="Member tidak ditemukan"
            )

        # cek buku
        cur.execute("""
            SELECT *
            FROM books
            WHERE id = %s
        """, (data.book_id,))

        book = cur.fetchone()

        if not book:

            raise HTTPException(
                status_code=404,
                detail="Buku tidak ditemukan"
            )

        if book["status"] == "dipinjam":

            raise HTTPException(
                status_code=400,
                detail="Buku sedang dipinjam"
            )

        today = date.today().isoformat()

        # insert loan
        cur.execute("""
            INSERT INTO loans (
                book_id,
                user_id,
                borrower_name,
                nim,
                member_id,
                phone,
                borrow_date,
                due_date,
                status
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                'dipinjam'
            )
        """, (
            data.book_id,
            data.user_id,
            data.borrower_name,
            member["nim"],
            data.member_id,
            data.phone,
            today,
            data.due_date
        ))

        # update status buku
        cur.execute("""
            UPDATE books
            SET status = 'dipinjam'
            WHERE id = %s
        """, (data.book_id,))

        db.commit()

        return {
            "message": "Peminjaman berhasil"
        }

    except Exception as e:

        db.rollback()

        print("BORROW ERROR:")
        print(e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        db.close()

# =========================
# RETURN BOOK
# =========================

@app.post("/loans/return")
def return_book(data: ReturnModel):

    db = get_db()

    cur = db.cursor()

    # cari peminjaman aktif
    cur.execute("""
        SELECT
            l.*,
            b.title as book_title
        FROM loans l
LEFT JOIN books b
ON l.book_id = b.id
        WHERE l.book_id = %s
        AND l.status IN ('dipinjam', 'terlambat')
        ORDER BY l.borrow_date DESC
        LIMIT 1
    """, (data.book_id,))

    loan = cur.fetchone()

    if not loan:

        db.close()

        raise HTTPException(
            status_code=404,
            detail="Tidak ada peminjaman aktif"
        )

    today = date.today().isoformat()

    # update loan
    cur.execute("""
        UPDATE loans
        SET
            status = 'dikembalikan',
            return_date = %s
        WHERE id = %s
    """, (
        today,
        loan["id"]
    ))

    # update status buku
    cur.execute("""
        UPDATE books
        SET status = 'tersedia'
        WHERE id = %s
    """, (data.book_id,))

    db.commit()

    db.close()

    return {
        "message":
        f"Buku '{loan['book_title']}' berhasil dikembalikan"
    }


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
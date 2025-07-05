import requests
import re
import csv
import psycopg2
from bs4 import BeautifulSoup
from crawler.config import API_KEY, BASE_URL, PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
import os
import random

# ----------------------------
# 🧹 Xử lý & Trích xuất
# ----------------------------

def clean_text(text):
    text = text.replace('\xa0', ' ')
    text = text.replace('', "'")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r' ([\.,;:])', r'\1', text)
    return text.strip()

def extract_tail_paragraph(text, min_len=200):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in reversed(paragraphs):
        if len(para) >= min_len and not re.search(r"\b(judgment\.|affirmed\.|reversed\.|dismissed\.)\b", para.lower()):
            return para
    return paragraphs[-1] if paragraphs else ""

def extract_outcome(text):
    match = re.search(
        r"\b(AFFIRMED\.|APPLIED\.|DISTINGUISHED\.|REFERRED TO\.|CONSIDERED\ |followed\.)",
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    text_lower = text.lower()

    if "cited" in text_lower:
        return "CITED"
    elif "applied" in text_lower:
        return "APPLIED"
    elif "distinguished" in text_lower:
        return "DISTINGUISHED"
    elif "referred to" in text_lower:
        return "REFERRED TO"
    elif "considered" in text_lower:
        return "CONSIDERED"
    elif "followed" in text_lower:
        return "FOLLOWED"
    return "UNKNOWN"
    
    # fallback_outcomes = [
    #     "cited", "applied", "distinguished",
    #     "referred to", "considered", "followed"
    # ]
    # return random.choice(fallback_outcomes)

def fetch_case_name(cluster_url, headers):
    try:
        r = requests.get(cluster_url, headers=headers)
        if r.status_code == 200:
            raw_name = r.json().get("case_name", "")
            name = re.sub(r'\s+', ' ', raw_name.strip())
            return name if len(name) >= 5 else None
    except:
        pass
    return None

# ----------------------------
# 💾 Lưu vào PostgreSQL
# ----------------------------

def connect_pg():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        database=PG_DATABASE
    ) 

def init_table():
    conn = connect_pg()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id INTEGER PRIMARY KEY,
            case_name TEXT,
            case_text TEXT,
            case_outcome TEXT
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

def is_text_exist(text):
    conn = connect_pg()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM cases WHERE case_text = %s;", (text,))
    exists = cursor.fetchone() is not None
    cursor.close()
    conn.close()
    return exists

def store_case(case):
    conn = connect_pg()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO cases (case_id, case_name, case_text, case_outcome)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (case_id) DO NOTHING;
    """, (
        case["case_id"],
        case["case_name"],
        case["case_text"],
        case["case_outcome"]
    ))
    conn.commit()
    cursor.close()
    conn.close()

# ----------------------------
# 🚀 Pipeline chính
# ----------------------------

def run_pipeline(pages):
    headers = {"Authorization": f"Token {API_KEY}"}
    seen_texts = set()
    collected_cases = []

    init_table()

    for page in range(pages, pages + 100):
        print(f"\n📄 Crawling page {page}...")
        resp = requests.get(BASE_URL, headers=headers, params={"page": page})
        if resp.status_code != 200:
            print(f"❌ Lỗi trang {page}: {resp.status_code}")
            continue

        data = resp.json()
        for item in data["results"]:
            case_id = item["id"]
            raw_text = item.get("plain_text", "")
            if not raw_text or len(raw_text) < 50:
                continue
            
            tail = extract_tail_paragraph(raw_text)
            cleaned = clean_text(tail)
            if cleaned in seen_texts or is_text_exist(cleaned):
                continue
            
            seen_texts.add(cleaned)

            cluster_url = item.get("cluster")
            case_name = item.get("case_name")
            if not case_name and cluster_url:
                case_name = fetch_case_name(cluster_url, headers)
            if not case_name:
                case_name = "Unknown Case"

            outcome = extract_outcome(raw_text)

            case = {
                "case_id": case_id,
                "case_name": case_name,
                "case_text": cleaned,
                "case_outcome": outcome
            }

            store_case(case)
            collected_cases.append(case)
            print(f"✅ Đã lưu case {case_id}")

    return collected_cases

# ----------------------------
# 🔁 Gọi pipeline
# ----------------------------

if __name__ == "__main__":
    cases = run_pipeline(1) 

import os
import pdfplumber
import docx
import logging
logging.basicConfig(level=logging.INFO)
import re
import nltk
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression as LR
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup as BS
import shutil
from pdf2image import convert_from_path as pdf2img
import pytesseract as pyt
import joblib
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

app = FastAPI()

SVC_MODEL = joblib.load("./mdo/SVC_model.pkl")
NB_MODEL = joblib.load("./mdo/GaussianNB_model.pkl")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

genai.configure(api_key='apki api key')
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=5)
count_vectorizer = CountVectorizer()

def pdf2txt(path):
    txt = ''
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            txt += p.extract_text()
    return txt

def docx2txt(path):
    doc = docx.Document(path)
    return '\n'.join([p.text for p in doc.paragraphs])

def clean_txt(txt):
    return re.sub(r'[^a-zA-Z0-9\s]', '', re.sub(r'\s+', ' ', txt)).lower()

def extract_kw(jd):
    return [kw.lower() for kw in jd.split(', ')]

def tfidf_sim(txt, kw):
    tfidf_vec = tfidf_vectorizer.fit_transform([txt] + kw)
    vecs = tfidf_vec.toarray()
    return (sum([cosine_similarity([vecs[0]], [v])[0][0] for v in vecs[1:]]) / len(vecs[1:])) * 100

def bert_similarity(resume_txt, jd_kw):
    resume_embedding = bert_model.encode(resume_txt)
    jd_embedding = bert_model.encode(jd_kw)
    return cosine_similarity([resume_embedding], [jd_embedding]).mean() * 100

def sec_extract(txt):
    txt = txt.lower()
    return {
        "skills": re.findall(r'(skills)(.*?)\n', txt),
        "edu": re.findall(r'(education)(.*?)\n', txt),
        "exp": re.findall(r'(experience)(.*?)\n', txt)
    }

def chk_links(txt):
    links = re.findall(r'(https?://[^\s]+)', txt)
    res = {}
    for link in links:
        try:
            r = requests.get(link, timeout=5)
            res[link] = {"valid": True, "content": scrape_site(r.text)} if r.status_code == 200 else {"valid": False, "content": None}
        except:
            res[link] = {"valid": False, "content": None}
    return res

def scrape_site(html):
    soup = BS(html, "html.parser")
    return clean_txt(soup.get_text())

def eval_link_content(content, kw):
    if not content:
        return 0
    vec = TfidfVectorizer().fit_transform([content] + kw)
    return cosine_similarity([vec.toarray()[0]], [vec.toarray()[1]])[0][0] * 100

def layout_chk(path):
    imgs = pdf2img(path)
    score = 0
    for img in imgs:
        txt = pyt.image_to_string(img)
        if "skills" in txt.lower() and "experience" in txt.lower() and "education" in txt.lower():
            score += 1
    return score / len(imgs) * 100

def svm_score(txt, kw):
    vec = tfidf_vectorizer.fit([txt] + kw)
    X = vec.transform([txt] + kw).toarray()
    return SVC_MODEL.predict_proba([X[0]])[0][1] * 100

def nb_score(txt, kw):
    vec = tfidf_vectorizer.fit([txt] + kw)
    X = vec.transform([txt] + kw).toarray()
    return NB_MODEL.predict_proba([X[0]])[0][1] * 100

def lr_score(txt, kw):
    if len(kw) < 1:
        return 0
    vec = count_vectorizer.fit([txt] + kw)
    X = vec.transform([txt] + kw).toarray()
    if len(X[1:]) > 1:
        labels = [1] + [0] * (len(X[1:]) - 1)
    else:
        return 0
    lr = LR()
    lr.fit(X[1:], labels)
    return lr.predict_proba([X[0]])[0][1] * 100

def gemini_generate_content(resume_txt, jd_txt):
    content = (
        f"Match this resume: {resume_txt} with the following job description: {jd_txt}. "
        f"Give me the skills, education, experience, links, links score, layout score, and the overall score "
        f"on a scale of 100 as a float. I need all of this in JSON format, with no extra text."
    )

    try:
        response = gemini_model.generate_content(content)
        if response.candidates and response.candidates[0].content.parts:
            gemini_json = json.loads(response.candidates[0].content.parts[0].text.strip('```json').strip('```'))
            logging.info(f"Gemini API response: {gemini_json}")
            return gemini_json
        else:
            logging.error("Gemini API error: Invalid response structure")
            return {"error": "Gemini API error: Invalid response structure"}

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse Gemini API response: {str(e)}")
        return {"error": "Failed to parse Gemini API response"}

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {str(e)}")
        return {"error": "An unexpected error occurred"}

@app.post("/upload/")
async def upload(file: UploadFile = File(...), jd: str = Form(...)):
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    loc = f"{temp_dir}/{file.filename}"

    with open(loc, "wb+") as f:
        shutil.copyfileobj(file.file, f)

    try:
        if file.filename.endswith('.pdf'):
            resume_txt = pdf2txt(loc)
        elif file.filename.endswith('.docx'):
            resume_txt = docx2txt(loc)
        else:
            return JSONResponse(status_code=400, content={"msg": "Invalid format. Only PDF and DOCX supported."})

        clean_res = clean_txt(resume_txt)
        kw = extract_kw(jd)

        ats = tfidf_sim(clean_res, kw)
        bert_score = bert_similarity(clean_res, ' '.join(kw))
        svm = svm_score(clean_res, kw)
        nb = nb_score(clean_res, kw)
        lr = lr_score(clean_res, kw)
        gemini_ats = gemini_generate_content(clean_res, jd)
        sections = {
            "skills": gemini_ats.get("skills", []),
            "exp": gemini_ats.get("experience", [])
        }
        links = gemini_ats.get("links", [])
        link_scores = gemini_ats.get("links_score", 0)
        layout_score = gemini_ats.get("layout_score", 0)
        return {
            "ATS Score (TF-IDF)": f"{ats:.2f}%",
            "BERT Score": f"{bert_score:.2f}%",
            "SVM Score": f"{svm:.2f}%",
            "NB Score": f"{nb:.2f}%",
            "LR Score": f"{lr:.2f}%",
            "ATS Score": gemini_ats,
            "Sections": sections,
            "Links": links,
            "Link Scores": link_scores,
            "Layout Score": f"{layout_score:.2f}%"
        }

    finally:
        if os.path.exists(loc):
            os.remove(loc)

@app.get("/")
def read_root():
    return {"msg": "Welcome to Dwos ATS"}

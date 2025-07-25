import streamlit as st
import os
import tempfile
import json
import sqlite3
import re
#import spacy
#from spacy import displacy
#from pdfminer.high_level import extract_text
#import pytesseract
from PIL import Image
#from spacy.matcher import Matcher
import pandas as pd

# --- Database Setup ---
DATABASE_NAME = 'cv_analysis.db'

def create_database_and_tables():
    """Connects to SQLite DB and creates tables if they don't exist."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                contact TEXT,
                job_role_svm TEXT,
                job_role_bert TEXT,
                experience_level_svm TEXT,
                experience_level_bert TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS education (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER,
                degree TEXT NOT NULL,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER,
                skill TEXT NOT NULL,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER,
                organization TEXT NOT NULL,
                position TEXT,
                duration TEXT,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER,
                relation_type TEXT,
                entity TEXT,
                context TEXT,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error during creation: {e}")
    finally:
        if conn:
            conn.close()

def save_analysis_to_db(model):
    """Saves the CV analysis results into the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        # Insert candidate information
        c = model['candidat']
        cursor.execute("INSERT INTO candidates (name, contact, job_role_svm, job_role_bert, experience_level_svm, experience_level_bert) VALUES (?, ?, ?, ?, ?, ?)",
                       (c.get('nom'), c.get('contact'), c.get('job_role_svm'), c.get('job_role_bert'), c.get('experience_level_svm'), c.get('experience_level_bert')))
        candidate_id = cursor.lastrowid
        # Insert other details...
        for degree in c['education']:
            cursor.execute("INSERT INTO education (candidate_id, degree) VALUES (?, ?)", (candidate_id, degree))
        for skill in c['competences']:
            cursor.execute("INSERT INTO skills (candidate_id, skill) VALUES (?, ?)", (candidate_id, skill))
        for exp in c['experiences']:
            cursor.execute("INSERT INTO experiences (candidate_id, organization, position, duration) VALUES (?, ?, ?, ?)",
                           (candidate_id, exp.get('organisation'), exp.get('poste'), exp.get('duree')))
        for rel in model['relations']:
            cursor.execute("INSERT INTO relations (candidate_id, relation_type, entity, context) VALUES (?, ?, ?, ?)",
                           (candidate_id, rel.get('relation'), rel.get('entite'), rel.get('contexte')))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error while saving analysis: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# --- Analysis Functions ---
def extract_data(file_path):
    try:
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img, lang='fra')
        elif file_path.lower().endswith('.pdf'):
            text = extract_text(file_path)
        else:
            raise ValueError("Format de fichier non supporté")
        return text
    except Exception as e:
        st.error(f"Erreur d'extraction: {str(e)}")
        return ""

def clean_data(text):
    text = re.sub(r'[^\w\s.,;:!?À-ÿ-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def extract_entities(text):
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "DATE": [], "LOC": [], "DIPLOME": [], "COMPETENCE": []}
    for ent in doc.ents:
        if ent.label_ in entities and ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    diplome_patterns = [
        [{"LOWER": {"IN": ["bac", "bts", "dut", "licence", "master", "doctorat"]}}],
        [{"LOWER": "bachelor"}, {"LOWER": "of"}, {"LOWER": "science"}],
        [{"LOWER": "ingénieur"}]]
    matcher = Matcher(nlp.vocab)
    matcher.add("DIPLOME", diplome_patterns)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        if span.text not in entities["DIPLOME"]:
            entities["DIPLOME"].append(span.text)
    return entities

def extract_relations(entities, text):
    relations = []
    doc = nlp(text)
    for sentence in doc.sents:
        if "expérience" in sentence.text:
            for competence in entities["COMPETENCE"]:
                if competence in sentence.text:
                    relations.append({"relation": "a_experience_avec", "entite": competence, "contexte": sentence.text.strip()})
    return relations

def build_model(entities, relations):
    model = {
        "candidat": {
            "nom": entities["PERSON"][0] if entities["PERSON"] else "Inconnu",
            "contact": "",
            "education": entities["DIPLOME"],
            "competences": entities["COMPETENCE"],
            "experiences": []
        },
        "relations": relations
    }
    for org in entities["ORG"]:
        model["candidat"]["experiences"].append({"organisation": org, "poste": "À déterminer", "duree": ""})
    return model

def classify_cv(text):
    # Placeholder for actual classification logic
    # In a real app, you would load your trained models here and use them
    return {
        "job_role_svm": "Not Implemented",
        "job_role_bert": "Not Implemented",
        "experience_level_svm": "Not Implemented",
        "experience_level_bert": "Not Implemented"
    }

def analyze_cv(file_path):
    st.text(f"Début de l'analyse du fichier: {os.path.basename(file_path)}")
    raw_text = extract_data(file_path)
    if not raw_text: return None
    cleaned_text = clean_data(raw_text)
    entities = extract_entities(cleaned_text)
    relations = extract_relations(entities, cleaned_text)
    model_data = build_model(entities, relations)
    classification_results = classify_cv(cleaned_text)
    model_data['candidat'].update(classification_results)
    save_analysis_to_db(model_data)
    return model_data

# --- Streamlit UI ---
def display_results(model):
    st.subheader(f"Résultats de l'analyse pour {model['candidat']['nom']}")
    st.markdown("#### Informations Candidat")
    st.write(f"**Nom:** {model['candidat'].get('nom', 'N/A')}")
    st.markdown("#### Formation")
    st.write(model['candidat']['education'] if model['candidat']['education'] else "Aucune information trouvée.")
    st.markdown("#### Compétences")
    st.write(model['candidat']['competences'] if model['candidat']['competences'] else "Aucune compétence trouvée.")
    st.markdown("#### Expériences Professionnelles")
    st.write(model['candidat']['experiences'] if model['candidat']['experiences'] else "Aucune expérience trouvée.")
    st.markdown("#### Classification Results")
    st.write(f"**Job Role (SVM):** {model['candidat'].get('job_role_svm', 'N/A')}")
    st.write(f"**Job Role (BERT):** {model['candidat'].get('job_role_bert', 'N/A')}")

def get_analysis_from_db(candidate_name):
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM candidates WHERE name = ?", (candidate_name,))
        candidate_info = cursor.fetchone()
        if candidate_info:
            # Reconstruct the model from DB data (simplified)
            # This part needs to be more robust in a real application
            model = {'candidat': {'nom': candidate_info[1]}} # Basic reconstruction
            return model
        return None
    except sqlite3.Error as e:
        st.error(f"Database error while retrieving analysis: {e}")
    finally:
        if conn: conn.close()
    return None


# --- Main App ---
create_database_and_tables()
st.title("Analyse de CV")

st.header("Analyser un nouveau CV")
uploaded_file = st.file_uploader("Choisissez un fichier PDF ou image", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    analysis_result = analyze_cv(tmp_file_path)
    os.unlink(tmp_file_path)

    if analysis_result:
        st.success("Analyse terminée avec succès!")
        display_results(analysis_result)

st.header("Voir les analyses précédentes")
conn = None
try:
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM candidates ORDER BY name")
    candidate_names = [row[0] for row in cursor.fetchall()]
    selected_candidate = st.selectbox("Sélectionnez un candidat:", ["-- Sélectionner --"] + candidate_names)
    if selected_candidate != "-- Sélectionner --":
        db_analysis_data = get_analysis_from_db(selected_candidate)
        if db_analysis_data:
            display_results(db_analysis_data)
        else:
            st.warning("Aucune analyse trouvée.")
except sqlite3.Error as e:
    st.error(f"Database error fetching candidates: {e}")
finally:
    if conn: conn.close()

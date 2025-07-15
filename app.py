import streamlit as st
import os
import tempfile
import json
import sqlite3

# Assuming DATABASE_NAME is defined in a previous cell
DATABASE_NAME = 'cv_analysis.db' # Ensure DATABASE_NAME is defined

# Ensure the previous cells defining the analysis functions (extract_data, clean_data,
# extract_entities, extract_relations, build_model, display_results, analyze_cv,
# save_analysis_to_db, create_database_and_tables, update_database_schema) have been executed.

# Configuration de l'OCR (Tesseract) - needs to be set if not already
# pytesseract.pytesseract.tesseract_cmd = r'<CHEMIN_VERS_TESSERACT_EXE>' # Replace with your Tesseract path


def get_analysis_from_db(candidate_name):
    """Queries the database for analysis results based on the candidate's name."""
    conn = None
    analysis_data = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # Get candidate ID and classification results
        cursor.execute("SELECT id, name, contact, job_role_svm, job_role_bert, experience_level_svm, experience_level_bert FROM candidates WHERE name = ?", (candidate_name,))
        candidate_info = cursor.fetchone()

        if candidate_info:
            candidate_id, name, contact, job_role_svm, job_role_bert, experience_level_svm, experience_level_bert = candidate_info

            analysis_data = {
                "candidat": {
                    "nom": name,
                    "contact": contact,
                    "education": [],
                    "competences": [],
                    "experiences": [],
                    "job_role_svm": job_role_svm, # Include classification results
                    "job_role_bert": job_role_bert,
                    "experience_level_svm": experience_level_svm,
                    "experience_level_bert": experience_level_bert
                },
                "relations": []
            }

            # Get education
            cursor.execute("SELECT degree FROM education WHERE candidate_id = ?", (candidate_id,))
            education_results = cursor.fetchall()
            analysis_data["candidat"]["education"] = [row[0] for row in education_results]

            # Get skills
            cursor.execute("SELECT skill FROM skills WHERE candidate_id = ?", (candidate_id,))
            skills_results = cursor.fetchall()
            analysis_data["candidat"]["competences"] = [row[0] for row in skills_results]

            # Get experiences
            cursor.execute("SELECT organization, position, duration FROM experiences WHERE candidate_id = ?", (candidate_id,))
            experiences_results = cursor.fetchall()
            analysis_data["candidat"]["experiences"] = [{"organisation": row[0], "poste": row[1], "duree": row[2]} for row in experiences_results]

            # Get relations
            cursor.execute("SELECT relation_type, entity, context FROM relations WHERE candidate_id = ?", (candidate_id,))
            relations_results = cursor.fetchall()
            analysis_data["relations"] = [{"relation": row[0], "entite": row[1], "contexte": row[2]} for row in relations_results]

    except sqlite3.Error as e:
        st.error(f"Database error while retrieving analysis: {e}")
    finally:
        if conn:
            conn.close()

    return analysis_data

# Create tables and update schema if they don't exist when the app starts
# Ensure create_database_and_tables and update_database_schema are defined elsewhere or included here
# For simplicity, let's assume they are defined in this script for standalone execution
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
                contact TEXT
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
        print(f"Database '{DATABASE_NAME}' and tables created successfully or already exist.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

def update_database_schema():
    """Adds columns for ML classification results to the candidates table."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE candidates ADD COLUMN job_role_svm TEXT")
        except sqlite3.OperationalError:
            pass # Column already exists
        try:
            cursor.execute("ALTER TABLE candidates ADD COLUMN job_role_bert TEXT")
        except sqlite3.OperationalError:
            pass # Column already exists
        try:
            cursor.execute("ALTER TABLE candidates ADD COLUMN experience_level_svm TEXT")
        except sqlite3.OperationalError:
            pass # Column already exists
        try:
            cursor.execute("ALTER TABLE candidates ADD COLUMN experience_level_bert TEXT")
        except sqlite3.OperationalError:
            pass # Column already exists
        conn.commit()
        print("Database schema update process completed.")

    except sqlite3.Error as e:
        print(f"Database error during schema update: {e}")
    finally:
        if conn:
            conn.close()

# Placeholder/dummy analysis functions for standalone app.
# In a real scenario, you'd import or define your actual analysis and ML functions here.
def extract_data(file_path):
    st.warning("Placeholder: Data extraction logic not implemented in this standalone app.")
    return "Placeholder text for analysis." # Return dummy text for the rest of the pipeline

def clean_data(text):
     st.warning("Placeholder: Data cleaning logic not implemented in this standalone app.")
     return text # Return text as is

def extract_entities(text):
    st.warning("Placeholder: Entity extraction logic not implemented in this standalone app.")
    # Return dummy entities for demonstration
    return {
        "PERSON": ["John Doe"],
        "ORG": ["Example Corp"],
        "DATE": ["2023-Present"],
        "LOC": ["New York"],
        "DIPLOME": ["Master's Degree"],
        "COMPETENCE": ["Python", "Communication"]
    }

def extract_relations(entities, text):
     st.warning("Placeholder: Relation extraction logic not implemented in this standalone app.")
     # Return dummy relations
     return [{"relation": "worked_at", "entite": "Example Corp", "contexte": "Worked at Example Corp"}]

def build_model(entities, relations):
    st.warning("Placeholder: Model building logic not fully implemented in this standalone app.")
    # Build a dummy model structure
    return {
        "candidat": {
            "nom": entities["PERSON"][0] if entities["PERSON"] else "Inconnu",
            "contact": "N/A", # Dummy contact
            "education": entities["DIPLOME"],
            "competences": entities["COMPETENCE"],
            "experiences": [{"organisation": org, "poste": "À déterminer", "duree": ""} for org in entities["ORG"]]
        },
        "relations": relations
    }

def save_analysis_to_db(model):
    st.warning("Placeholder: Saving to database logic not fully implemented in this standalone app.")
    # In a real app, you would implement the actual database saving here
    print("Placeholder: Saving model to DB:", model)
    pass # Dummy save

# Placeholder/dummy classification function
def classify_cv(text, svm_model=None, tfidf_vectorizer=None, bert_model=None, tokenizer=None, label_map=None):
    st.warning("Placeholder: ML classification logic not implemented in this standalone app.")
    # Return dummy classification results
    return {
        "job_role_svm": "Placeholder Role (SVM)",
        "job_role_bert": "Placeholder Role (BERT)",
        "experience_level_svm": "Placeholder Level (SVM)",
        "experience_level_bert": "Placeholder Level (BERT)"
    }

def analyze_cv(file_path, svm_model=None, tfidf_vectorizer=None, bert_model=None, tokenizer=None, label_map=None):
    st.text(f"Début de l'analyse du fichier: {os.path.basename(file_path)}")

    raw_text = extract_data(file_path)
    if not raw_text:
        return None

    cleaned_text = clean_data(raw_text)
    entities = extract_entities(cleaned_text)
    relations = extract_relations(entities, cleaned_text)
    model_data = build_model(entities, relations)

    # Placeholder classification call
    classification_results = classify_cv(cleaned_text, svm_model, tfidf_vectorizer, bert_model, tokenizer, label_map)
    model_data['candidat'].update(classification_results)

    save_analysis_to_db(model_data)

    # Return the model data to be displayed by the Streamlit app
    return model_data


def display_results(model):
    """Displays the analysis results in Streamlit."""
    st.markdown("#### Informations Candidat")
    st.write(f"**Nom:** {model['candidat']['nom']}")
    st.write(f"**Contact:** {model['candidat']['contact']}") # Display contact

    st.markdown("#### Formation")
    if model['candidat']['education']:
        for i, diplome in enumerate(model['candidat']['education'], 1):
            st.write(f"- {diplome}")
    else:
        st.write("Aucune information de formation trouvée.")

    st.markdown("#### Compétences")
    if model['candidat']['competences']:
        for i, competence in enumerate(model['candidat']['competences'], 1):
             st.write(f"- {competence}")
    else:
        st.write("Aucune compétence trouvée.")

    st.markdown("#### Expériences Professionnelles")
    if model['candidat']['experiences']:
        for i, exp in enumerate(model['candidat']['experiences'], 1):
            st.write(f"- **Organisation:** {exp.get('organisation', 'N/A')}")
            st.write(f"  **Poste:** {exp.get('poste', 'À déterminer')}")
            st.write(f"  **Durée:** {exp.get('duree', 'N/A')}")
        else:
             st.write("Aucune expérience professionnelle trouvée.")


    st.markdown("#### Relations identifiées")
    if model['relations']:
        for rel in model['relations']:
             st.write(f"- **Relation:** {rel.get('relation', 'N/A')}, **Entité:** {rel.get('entite', 'N/A')}, **Contexte:** {rel.get('contexte', 'N/A')}")
    else:
        st.write("Aucune relation identifiée.")

    st.markdown("#### Classification Results")
    st.write(f"**Job Role (SVM):** {model['candidat'].get('job_role_svm', 'N/A')}")
    st.write(f"**Job Role (BERT):** {model['candidat'].get('job_role_bert', 'N/A')}")
    st.write(f"**Experience Level (SVM):** {model['candidat'].get('experience_level_svm', 'N/A')}")
    st.write(f"**Experience Level (BERT):** {model['candidat'].get('experience_level_bert', 'N/A')}")


# --- Streamlit App Layout ---

# Create tables and update schema if they don't exist when the app starts
create_database_and_tables()
update_database_schema()


st.title("Analyse de CV")

# Section for uploading a new CV
st.header("Analyser un nouveau CV")
uploaded_file = st.file_uploader("Choisissez un fichier PDF ou image (PNG, JPG, JPEG)", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.info(f"Fichier téléchargé: {uploaded_file.name}")

    st.text("Analyse en cours...")
    # analyze_cv now saves to DB and includes classification (assuming models are available)
    # Note: The analyze_cv function now requires model objects.
    # For this Streamlit app to work, you would need to load the trained models here
    # or make them globally accessible. This is a simplification for demonstration.
    # Assuming svm_model, tfidf_vectorizer, model (bert_model), tokenizer, label_map
    # are available from previous cells' execution.
    try:
         # Pass the models and label_map to analyze_cv - using None as placeholders
         # for this standalone app where models are not loaded.
        analysis_result = analyze_cv(tmp_file_path, None, None, None, None, None)

         # Retrieve data from the database after saving
        if analysis_result and 'candidat' in analysis_result and 'nom' in analysis_result['candidat']:
            candidate_name = analysis_result['candidat']['nom']
            db_analysis_data = get_analysis_from_db(candidate_name)
            if db_analysis_data:
                st.success(f"Analyse terminée et sauvegardée pour {candidate_name}!")
                # Display data retrieved from the database
                st.subheader(f"Résultats de l'analyse pour {candidate_name}")
                # Modify display_results to handle and show classification
                # For now, we'll just display the classification results directly here
                display_results(db_analysis_data) # Reuse the original display function


            else:
                st.warning(f"Analyse terminée, mais les données pour {candidate_name} n'ont pas pu être récupérées de la base de données.")

        else:
            st.error("Échec de l'analyse du CV. Veuillez vérifier le file content or analysis logic.")

    except NameError as e:
        st.error(f"Error: Required models or variables are not defined. Please ensure all previous cells, including model training, have been executed. Details: {e}")
    except Exception as e:
         st.error(f"An unexpected error occurred during analysis: {e}")


    # Clean up the temporary file
    os.unlink(tmp_file_path)


# Section for viewing previously analyzed CVs
st.header("Voir les analyses précédentes")

# Get list of candidates from the database
conn = None
candidate_names = []
try:
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM candidates ORDER BY name")
    candidate_names = [row[0] for row in cursor.fetchall()]
except sqlite3.Error as e:
    st.error(f"Database error while fetching candidate list: {e}")
finally:
    if conn:
        conn.close()

if candidate_names:
    selected_candidate = st.selectbox("Sélectionnez un candidat:", ["-- Sélectionner --"] + candidate_names)

    if selected_candidate != "-- Sélectionner --":
        st.text(f"Récupération de l'analyse pour {selected_candidate}...")
        previous_analysis_data = get_analysis_from_db(selected_candidate)

        if previous_analysis_data:
            st.subheader(f"Résultats de l'analyse pour {selected_candidate}")
            display_results(previous_analysis_data) # Reuse the display function


        else:
            st.warning(f"Aucune analyse trouvée pour {selected_candidate}.")
else:
    st.info("Aucun CV analysé n'a été trouvé dans la base de données.")

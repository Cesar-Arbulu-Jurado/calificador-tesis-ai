import streamlit as st
import fitz  # PyMuPDF
import json
import os
import subprocess
from google import genai

# Configuración de página
st.set_page_config(page_title="Calificador Automático de Tesis", layout="wide")

st.title("Calificador Automático de Tesis (Map-Reduce)")
st.write("Sube el PDF, selecciona las rúbricas y obtén tu informe con citas literales.")

# --- SIDEBAR: Configuración Institucional y de Evaluación ---
with st.sidebar:
    st.header("1. Datos del Evaluador")
    evaluator_name = st.text_input("Nombre del Docente")
    evaluator_role = st.selectbox("Rol", ["Dictaminante", "Asesor", "Replicante"])
    university = st.text_input("Universidad")
    email = st.text_input("Correo Electrónico")
    
    st.header("2. Nivel de Rigurosidad")
    rigor_level = st.radio("Selecciona el rigor:", [
        "1 - Básico (Tolerante)", 
        "2 - Intermedio (Estándar)", 
        "3 - Avanzado (Estricto)"
    ], index=1)
    # Extraer el número del rigor
    rigor_val = int(rigor_level.split(" ")[0])

# --- MAIN ÁREA: Carga de PDF y Rúbricas ---
st.header("3. Subir Tesis")
uploaded_file = st.file_uploader("Selecciona el PDF de la tesis", type="pdf")

st.header("4. Rúbricas a Evaluar")
rubricas_disponibles = [
    "Cumplimiento de Normativa",
    "Pertinencia Institucional",
    "Perfil del Egresado",
    "Respuesta a Necesidades del Planeta (ODS)",
    "Respuesta a Necesidades del País (PEDN)",
    "Ajuste Epistemológico del Proyecto",
    "Formulación del Problema",
    "Justificaciones",
    "Hipótesis de Investigación",
    "Pregunta de Investigación" # Limitado aquí por concepto, agregar las 19
]
selected_rubrics = st.multiselect("Selecciona las secciones inalterables de LaTeX", rubricas_disponibles, default=rubricas_disponibles[:2])

# Inicializar cliente de Gemini usando la variable de entorno que inyectará Streamlit Cloud
def get_gemini_client():
    if "GEMINI_API_KEY" not in os.environ:
        st.error("Error: La llave GEMINI_API_KEY no está configurada en los Secrets de Streamlit.")
        st.stop()
    return genai.Client()

def extract_chunks(file_bytes, chunk_size=15):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunk_text = ""
    chunks = []
    for i in range(len(doc)):
        chunk_text += doc[i].get_text("text") + f"\n[Pág. {i+1}]\n"
        if (i + 1) % chunk_size == 0 or i == len(doc) - 1:
            chunks.append(chunk_text)
            chunk_text = ""
    return chunks

def map_phase(client, chunk_text, rubric, rigor):
    from google.genai import types
    prompt = f"""
    Actúa como evaluador experto de tesis. Rúbrica: {rubric}. Nivel de rigor: {rigor}.
    REGLA OBLIGATORIA: Si hallas un error, extrae la cita EXACTA enmarcada en comillas ("...").
    Si parafraseas fallarás. Si no hay evidencia, devuelve vacío [].
    Retorna JSON puro con un arreglo de objetos: [{{"error_description": "...", "exact_quote": "..."}}].
    
    Texto:
    {chunk_text}
    """
    
    modelos_map = ['gemini-2.5-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-flash-002', 'gemini-flash']
    error_msg = ""
    for m in modelos_map:
        try:
            res = client.models.generate_content(
                model=m, 
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(res.text) if res.text else []
        except Exception as e:
            error_msg = str(e)
            continue
            
    st.write(f"⚠️ Error en Map (Ningún modelo Flash funcionó): {error_msg}")
    return []

def reduce_phase(client, rubric, map_results, rigor):
    from google.genai import types
    evidences = json.dumps(map_results)
    prompt = f"""
    Eres Gemini Pro. Rúbrica: {rubric}. Rigor: {rigor}.
    Evidencias de la tesis: {evidences}.
    1. Delimita únicamente los 3 peores errores.
    2. Mantén las citas literales ("...") invariables.
    3. Redacta un sustento teórico APA (4-6 líneas).
    4. Asigna un puntaje (número entero).
    Devuelve EXACTAMENTE UN JSON PURO con las siguientes claves y tipos de dato:
    {{
      "top_3_errores": [{{"error_description": "...", "exact_quote": "..."}}],
      "sustento_teorico": "...",
      "puntaje": 0
    }}
    """
    
    modelos_reduce = ['gemini-2.5-pro', 'gemini-1.5-pro-latest', 'gemini-1.5-pro-002', 'gemini-pro']
    error_msg = ""
    for m in modelos_reduce:
        try:
            res = client.models.generate_content(
                model=m, 
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(res.text)
        except Exception as e:
            error_msg = str(e)
            continue
            
    return {"top_3_errores": [], "sustento_teorico": f"Error técnico de API crítico (Model Not Found): {error_msg}", "puntaje": 0}

if st.button("Iniciar Evaluación Completa", type="primary"):
    if not uploaded_file or not evaluator_name or not selected_rubrics:
        st.warning("Completa los datos del evaluador, seleccionarúbricas y sube el PDF.")
        st.stop()
        
    client = get_gemini_client()
    file_bytes = uploaded_file.read()
    
    st.info("Leyendo y particionando archivo PDF en memoria...")
    chunks = extract_chunks(file_bytes, chunk_size=15)
    
    informe_final = []
    
    # Barra de progreso principal
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(selected_rubrics)
    
    for idx, rubric in enumerate(selected_rubrics):
        status_text.text(f"Fase MAP [Gemini Flash]: Filtrando evidencias para '{rubric}' a lo largo del documento...")
        
        all_candidates = []
        for chunk in chunks:
            candidates = map_phase(client, chunk, rubric, rigor_val)
            if candidates:
                all_candidates.extend(candidates)
                
        status_text.text(f"Fase REDUCE [Gemini Pro]: Consolidando evaluación rigurosa para '{rubric}'...")
        rubric_result = reduce_phase(client, rubric, all_candidates, rigor_val)
        
        informe_final.append({
            "rubrica": rubric,
            "resultado": rubric_result
        })
        progress_bar.progress((idx + 1) / total_steps)
        
    status_text.success("¡Evaluación de Inteligencia Artificial completada!")
    
    # Renderizado en Pantalla
    st.balloons()
    st.header("Resultados de la Evaluación (Preview)")
    
    total_score = 0
    for item in informe_final:
        res = item['resultado']
        st.subheader(f"📌 {item['rubrica']} - Puntaje: {res.get('puntaje', 0)}")
        st.write(f"**Sustento Teórico (APA):** {res.get('sustento_teorico')}")
        for err in res.get('top_3_errores', []):
            st.error(f"**Observación:** {err.get('error_description')}\n\n**Evidencia Literal:** {err.get('exact_quote')}")
        total_score += int(res.get('puntaje', 0))
        st.divider()
        
    st.markdown(f"### Puntaje Total: **{total_score}**")
    
    st.info("En la nube, aquí se ejecuta 'pdflatex' para compilar el informe_final.tex y crear el botón de descarga del PDF.")

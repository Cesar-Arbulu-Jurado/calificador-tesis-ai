import streamlit as st
import fitz  # PyMuPDF
import json
import os
import subprocess
from google import genai
import smtplib
from email.message import EmailMessage
import re

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
    email = st.text_input("Tu Correo Electrónico (Opcional)")
    correo_destino = st.text_input("Correo electrónico del evaluado (Destino del PDF)")
    
    st.header("2. Nivel de Rigurosidad")
    max_observaciones = st.number_input("Número de Revisiones Deseadas por Criterio", min_value=2, max_value=10, value=3)
    
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
# Cargar la base de datos de matrices (compilada desde main.tex)
db_path = os.path.join(os.path.dirname(__file__), "rubricas_extraidas.json")
if os.path.exists(db_path):
    with open(db_path, "r", encoding="utf-8") as f:
        rubricas_db = json.load(f)
else:
    st.error("⚠️ Error Crítico: No se encontró 'rubricas_extraidas.json'. El parseador no se ejecutó.")
    st.stop()

rubricas_disponibles = list(rubricas_db.keys())

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

def map_phase(client, chunk_text, rubric_title, rubric_content, rigor):
    from google.genai import types
    prompt = f"""
    Actúa como evaluador experto de tesis.
    Dimensión a evaluar: {rubric_title}
    
    Nivel de rigor: {rigor}.
    
    CRITERIOS INALTERABLES (Matriz LaTeX original):
    {rubric_content}
    
    REGLA OBLIGATORIA: Si hallas un error acorde a la matriz, extrae la cita EXACTA enmarcada en comillas ("...").
    Si parafraseas fallarás. Si no hay evidencia, devuelve vacío [].
    Retorna JSON puro con un arreglo de objetos: [{{"error_description": "...", "exact_quote": "..."}}].
    
    Texto:
    {chunk_text}
    """
    
    modelos_map = ['gemini-3-flash-preview', 'gemini-3-flash', 'gemini-2.5-flash', 'gemini-1.5-flash-latest', 'gemini-flash']
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

def reduce_phase(client, rubric_title, rubric_content, map_results, rigor, max_errores):
    from google.genai import types
    evidences = json.dumps(map_results)
    prompt = f"""
    Eres Gemini Pro. Rigor: {rigor}.
    Dimensión a evaluar: {rubric_title}
    
    CRITERIOS INALTERABLES (Matriz LaTeX original):
    {rubric_content}
    
    Evidencias extraídas de la tesis: {evidences}.
    1. Ejecuta un 'Deep Research' (cadena de pensamiento profundo) obligatorio rastreando en tu memoria las mejores fuentes de Ingeniería Civil para esta dimensión. Escribe tu análisis en la variable JSON.
    2. Delimita únicamente los {max_errores} peores errores basándote ESTRICTAMENTE en la matriz LaTeX provista.
    3. Mantén las citas literales ("...") invariables.
    4. Redacta un sustento teórico APA (4-6 líneas) aplicando rigor epistemológico profundo:
       -> PROHIBIDO citar o respaldarse en "Hernández Sampieri" y cualquier otro teórico de las ciencias sociales, investigación social o psicología.
       -> OBLIGATORIO usar exclusivamente literatura científica anglosajona moderna en INGENIERÍA CIVIL y ciencias aplicadas exactas. Uso de autores en español solo si es absolutamente inevitable.
       -> OBLIGATORIO: Debes insertar explícitamente las citas parentéticas (Autor, Año) dentro de la redacción de este párrafo. Todo autor que pongas en el arreglo "referencias_apa" DEBE estar citado aquí.
    5. Asigna un puntaje (número entero).
    Devuelve EXACTAMENTE UN JSON PURO con las siguientes claves y tipos de dato:
    {{
      "deep_research_analysis": "Ejecuta aquí tu razonamiento interno y Deep Research sobre qué literatura civil usar...",
      "errores_hallados": [{{"error_description": "...", "exact_quote": "..."}}],
      "sustento_teorico": "...",
      "referencias_apa": ["Referencia 1 estilo APA del autor usado en sustento", "Referencia 2..."],
      "puntaje": 0
    }}
    """
    
    modelos_reduce = ['gemini-3-pro-preview', 'gemini-3-pro', 'gemini-2.5-pro', 'gemini-1.5-pro-latest', 'gemini-pro']
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
            
    return {"errores_hallados": [], "referencias_apa": [], "sustento_teorico": f"Error técnico de API crítico (Model Not Found): {error_msg}", "puntaje": 0}

if st.button("Iniciar Evaluación Completa", type="primary"):
    if not uploaded_file or not evaluator_name or not selected_rubrics or not correo_destino:
        st.warning("Completa los datos del evaluador, el correo de destino, selecciona rúbricas y sube el PDF.")
        st.stop()
        
    client = get_gemini_client()
    file_bytes = uploaded_file.read()
    
    st.info("Leyendo y particionando archivo PDF en memoria...")
    chunks = extract_chunks(file_bytes, chunk_size=15)
    
    informe_final = []
    todas_las_referencias = []
    
    # Barra de progreso principal
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(selected_rubrics)
    
    for idx, rubric in enumerate(selected_rubrics):
        rubrica_texto_latex = rubricas_db[rubric]
        status_text.text(f"Fase MAP [Gemini Flash]: Filtrando evidencias para '{rubric}' a lo largo del documento...")
        
        all_candidates = []
        for chunk in chunks:
            candidates = map_phase(client, chunk, rubric, rubrica_texto_latex, rigor_val)
            if candidates:
                all_candidates.extend(candidates)
                
        status_text.text(f"Fase REDUCE [Gemini Pro]: Consolidando evaluación rigurosa para '{rubric}'...")
        rubric_result = reduce_phase(client, rubric, rubrica_texto_latex, all_candidates, rigor_val, max_observaciones)
        
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
        for err in res.get('errores_hallados', []):
            st.error(f"**Observación:** {err.get('error_description')}\n\n**Evidencia Literal:** {err.get('exact_quote')}")
        total_score += int(res.get('puntaje', 0))
        st.divider()
        
    st.markdown(f"### Puntaje Total: **{total_score}**")
    
    # ==========================
    # MOTOR DE COMPILACIÓN LATEX
    # ==========================
    st.info("Compilando reporte formal en LaTeX (pdflatex)...")
    
    def escape_latex(text):
        if not text:
            return ""
        # Especiales de LaTeX
        chars = {
            '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'
        }
        escaped = "".join([chars.get(c, c) for c in text])
        return escaped

    # Generación de la Plantilla LaTeX
    latex_content = r"\documentclass[12pt,a4paper]{article}" + "\n"
    latex_content += r"\usepackage[utf8]{inputenc}" + "\n"
    latex_content += r"\usepackage[T1]{fontenc}" + "\n"
    latex_content += r"\usepackage[spanish]{babel}" + "\n"
    latex_content += r"\usepackage{geometry}" + "\n"
    latex_content += r"\usepackage{xcolor}" + "\n"
    latex_content += r"\geometry{margin=2.5cm}" + "\n"
    latex_content += r"\begin{document}" + "\n\n"
    latex_content += r"\begin{center}" + "\n"
    latex_content += r"{\LARGE \textbf{Dictamen Oficial de Evaluación de Tesis}} \\ [0.5cm]" + "\n"
    latex_content += r"\end{center}" + "\n\n"
    
    latex_content += rf"\textbf{{Evaluador:}} {escape_latex(evaluator_name)}\\" + "\n"
    latex_content += rf"\textbf{{Rol:}} {escape_latex(evaluator_role)}\\" + "\n"
    latex_content += rf"\textbf{{Institución:}} {escape_latex(university)}\\" + "\n\n"
    latex_content += r"\hrule\vspace{0.5cm}" + "\n\n"

    for item in informe_final:
        res = item['resultado']
        rubrica_esc = escape_latex(item['rubrica'])
        sustento_esc = escape_latex(res.get('sustento_teorico', ''))
        puntaje = res.get('puntaje', 0)
        
        latex_content += rf"\subsection*{{{rubrica_esc} (Puntaje Asignado: {puntaje})}}" + "\n"
        latex_content += rf"\textbf{{Sustento Teórico (APA):}} {sustento_esc} \vspace{{0.3cm}} \\" + "\n"
        
        errores = res.get('errores_hallados', [])
        if errores:
            latex_content += r"\begin{itemize}" + "\n"
            for err in errores:
                desc_esc = escape_latex(err.get('error_description', ''))
                quote_esc = escape_latex(err.get('exact_quote', ''))
                latex_content += rf"  \item \textbf{{Observación:}} {desc_esc} \\ \textit{{Evidencia Cruda:}} ``{quote_esc}''" + "\n"
            latex_content += r"\end{itemize}" + "\n"
        
        referencias_bloque = res.get('referencias_apa', [])
        todas_las_referencias.extend(referencias_bloque)
        
        latex_content += r"\vspace{0.5cm}" + "\n\n"

    if todas_las_referencias:
        latex_content += r"\newpage" + "\n"
        latex_content += r"\section*{Referencias Bibliográficas Consolidadas}" + "\n"
        latex_content += r"\begin{itemize}" + "\n"
        # Limpieza de duplicados y orden
        referencias_unicas = sorted(list(set(todas_las_referencias)))
        for r in referencias_unicas:
            latex_content += rf"  \item {escape_latex(r)}" + "\n"
        latex_content += r"\end{itemize}" + "\n\n"

    latex_content += r"\vfill\hrule\vspace{0.2cm}\begin{center}" + "\n"
    latex_content += rf"\textbf{{PUNTAJE GLOBAL OBTENIDO: {total_score}}}" + "\n"
    latex_content += r"\end{center}" + "\n\n"
    latex_content += r"\end{document}"

    os.makedirs("reportes_temp", exist_ok=True)
    tex_path = os.path.join("reportes_temp", "informe_oficial.tex")
    pdf_path = os.path.join("reportes_temp", "informe_oficial.pdf")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    compilado_ok = True
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory=reportes_temp", tex_path], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        compilado_ok = False
        stderr_log = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ""
        stdout_log = e.stdout.decode('utf-8', errors='ignore') if e.stdout else ""
        # Extraer las últimas líneas de error para no saturar la pantalla
        full_log = (stdout_log + "\n" + stderr_log)[-2000:]
        st.error("Error crítico de sintaxis en LaTeX:\n\n```text\n" + full_log + "\n```")

    if compilado_ok and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            
        st.success("¡Documento LaTeX generado exitosamente!")
        st.download_button(
            label="📄 Descargar Informe Final (PDF)",
            data=pdf_bytes,
            file_name="Dictamen_Tesis.pdf",
            mime="application/pdf"
        )
        
        # ==========================
        # MOTOR DE CORREO SMTP
        # ==========================
        try:
            emisor = st.secrets["EMAIL_ADDRESS"]
            clave_app = st.secrets["EMAIL_PASSWORD"]
            
            st.info(f"Enviando reporte de forma automática a {correo_destino} ...")
            msg = EmailMessage()
            msg['Subject'] = 'Resultados de Evaluación de Tesis - Dictamen IA'
            msg['From'] = emisor
            msg['To'] = correo_destino
            msg.set_content(f"Estimado tesista/interesado,\n\nAdjunto sírvase encontrar el dictamen riguroso generado por {evaluator_name}.\n\nPuntaje Global: {total_score}\n\nAtentamente,\nRobot Calificador")
            
            msg.add_attachment(pdf_bytes, maintype='application', subtype='pdf', filename='Dictamen_Tesis.pdf')
            
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(emisor, clave_app)
                server.send_message(msg)
                
            st.success("📧 ¡Reporte enviado por correo exitosamente!")
        except Exception as smtp_err:
            st.warning("No se pudo enviar el correo.")
            st.write("Asegúrate de haber configurado tu `EMAIL_ADDRESS` y `EMAIL_PASSWORD` (App Password de Gmail) en la sección 'Secrets' de la plataforma Streamlit Cloud.")
            st.code(str(smtp_err))

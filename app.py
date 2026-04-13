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
    Eres un Evaluador de Tesis de Maestría de Alto Nivel con perfil puramente de Ingeniería Civil. Rigor: {rigor}.
    Evalúas mediante la física, la mecánica, confiabilidad de datos discretos y juicio ingenieril, y RECHAZAS metodologías genéricas, ciencias sociales o a "Hernández Sampieri".
    Dimensión a evaluar: {rubric_title}
    
    CRITERIOS INALTERABLES (Matriz LaTeX original):
    {rubric_content}
    
    Evidencias extraídas de la tesis: {evidences}.
    
    INSTRUCCIONES ESTRICTAS (BASADAS EN "Thesis_review_writer.md"):
    1. Ejecuta un 'Deep Research' (cadena de pensamiento profundo) obligatorio rastreando fuentes estrictamente de Ingeniería Civil anglosajona/internacional verificable. Escribe tu análisis en la variable respectiva JSON.
    2. Identifica hasta {max_errores} observaciones basándote ESTRICTAMENTE en la matriz LaTeX.
    3. Para CADA observación redacta UN (1) solo párrafo narrativo continuo (de unas 4-8 líneas), integrado de manera fluida y orgánica (Cero viñetas, Cero enumeraciones, Cero etiquetas en negrita como "Observación:" o "Definición:"). Todo debe fluir en prosa en tercera persona académica y de tono impersonal.
    4. PROHIBICIÓN ABSOLUTA DE INVENCIÓN (ACADEMIC FABRICATION): 
       - NO inventes estudios ("Un metaanálisis reciente dice..."), encuestas ni contextos geográficos.
       - NO inventes estadísticas de alta precisión ni números (Ej. "El 67.3%...", "2.3 +- 1.1").
       - Usa fraseo observacional o normativo verídico (Ej: "En la práctica académica se observa...", "Según ASTM D1633...").
    5. PROHIBICIONES LÉXICAS Y ESTILÍSTICAS:
       - PROHIBIDO el uso de las siguientes palabras de relleno: crucial, significativo, ingenieril, disciplinar, esencial, fundamental, notable, relevante, imprescindible, valioso, considerable, trascendental, integral, exhaustivo, óptimo, además, sin embargo, por lo tanto, en conclusión, en consecuencia, por ende, por otro lado, asimismo, no obstante, cabe destacar, es importante señalar que, juega un papel crucial, contribuye significativamente, implementar, optimizar, aprovechar, facilitar, potenciar, maximizar, innovar, transformar, concepto, perspectiva, enfoque, factor, contexto, desafío, oportunidad, metodología, dinámica, por consiguiente, en definitiva, en resumen.
       - Usa lenguaje rígidamente descriptivo y técnico en español latinoamericano.
    6. ESTRUCTURA NARRATIVA OBLIGATORIA DEL PÁRRAFO:
       (a) Breve fundamento teórico/conceptual (1-2 oraciones) y con CITA PARENTÉTICA APA fluida sin detener la lectura.
       (b) Desarrollo incisivo del error conectando el vacío teórico detectado con la práctica.
       (c) Cita literal extraída estrictamente del arreglo de Evidencias, insertada obligatoriamente con el comando LaTeX \\enquote{{...}} indicando de dónde proviene en el texto.
    7. FORMATO MATEMÁTICO LATEX ESTRICTO:
       - Obligatorio usar "," (coma) para decimales (ej. $3{{,}}14$). NUNCA ".".
       - Obligatorio usar "\\," para miles (ej. $1\\,234$).
       - Cero uso del símbolo "$" monetario; usa exclusivamente "USD~$...$".
       - Jamás uses apóstrofos simples (') en mode matemático (emplea ^\\prime).
       - Expresa unidades físicas siempre dentro de modo matemático usando \\text{{}} (Ej. $10\\,\\text{{kg}}$). Sin anidación conflictiva.
    8. TODO AUTOR debe aparecer en "referencias_apa" en FORMATO APA 7ma Edición estricto. Si contiene un enlace URL, enciérralo usando el comando \\url{enlace}. Ejemplo: Autor, A. (2020). Título. \\url{https://...}
    9. Asigna un "puntaje" entero.
    
    Devuelve EXACTAMENTE UN JSON PURO con las siguientes claves:
    {{
      "deep_research_analysis": "Ejecuta aquí tu razonamiento interno objetivo y riguroso...",
      "observaciones_narrativas": [
        "Párrafo narrativo continuo de 4-8 líneas que contenga sustento (Smith, 2020), la observación entrelazada fluidamente y termine con una cita textual obligatoria usando el código \\enquote{{texto literal}} evidenciado previamente.",
        "Párrafo 2..."
      ],
      "referencias_apa": ["Smith, A. (2020). Título...", "Ref 2..."],
      "puntaje": 0
    }}
    """
    
    modelos_reduce = ['gemini-3-pro-preview', 'gemini-3-pro', 'gemini-2.5-pro', 'gemini-1.5-pro-latest']
    error_msg = ""
    for m in modelos_reduce:
        try:
            res = client.models.generate_content(
                model=m, 
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            parsed = json.loads(res.text)
            if not isinstance(parsed, dict):
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    return parsed[0]
                else:
                    raise Exception(f"El JSON devuelto no es un diccionario: {str(parsed)[0:100]}")
            return parsed
        except Exception as e:
            error_msg = str(e)
            continue
            
    return {"observaciones_narrativas": [f"Error técnico de API crítico (Model Not Found o Bad JSON): {error_msg}"], "referencias_apa": [], "puntaje": 0}

def deduplicate_phase(client, informe_final):
    from google.genai import types
    prompt = f"""
    Eres un analista experto en la consolidación de informes técnicos de Ingeniería Civil.
    Tienes el siguiente JSON que contiene la evaluación de distintos criterios (rúbricas) de una tesis:
    {json.dumps(informe_final, ensure_ascii=False)}

    INSTRUCCIONES ESTRICTAS:
    1. Compara de forma secuencial y diferencial el texto de TODAS las observaciones ("observaciones_narrativas") a través de TODAS las rúbricas.
    2. Si encuentras observaciones en distintas rúbricas que critiquen exactamente el mismo defecto metodológico, teórico o matemático (similitud conceptual o semántica >= 80%):
       - Consolídalas en UNA SOLA observación narrativo-continua, integrando los detalles y evidencias de ambas.
       - Mantén estrictamente el rigor académico civil, las comas en decimales, sin palabras prohibidas (crucial, significativo) y el uso de los comandos LaTeX \\enquote{{...}}.
       - Ubica esta única observación consolidada EXCLUSIVAMENTE en la rúbrica donde apareció primero.
       - ELIMINA la observación redundante de las rúbricas posteriores.
    3. Respeta intactas las observaciones singulares (similitud < 80%).
    4. NO alteres bajo ningún motivo los puntajes ("puntaje") de ninguna rúbrica.
    5. Mantén y unifica todas las referencias APA requeridas por cada observación.

    Devuelve EXACTAMENTE el esquema JSON original reconstruido con tus modificaciones (Debe ser la lista de diccionarios, uno por rúbrica):
    [
        {{
            "rubrica": "Nombre 1",
            "resultado": {{
                "deep_research_analysis": "...",
                "observaciones_narrativas": ["Obs consolidada o preservada..."],
                "referencias_apa": ["..."],
                "puntaje": X
            }}
        }}, 
        ...
    ]
    """
    modelos_dedup = ['gemini-2.5-pro', 'gemini-1.5-pro-latest', 'gemini-3-pro-preview']
    for m in modelos_dedup:
        try:
            res = client.models.generate_content(
                model=m, 
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            parsed = json.loads(res.text)
            return parsed if isinstance(parsed, list) else informe_final
        except Exception:
            continue
    return informe_final

if st.button("Iniciar Evaluación Completa", type="primary"):
    if not uploaded_file or not evaluator_name or not selected_rubrics or not correo_destino:
        st.warning("Completa los datos del evaluador, el correo de destino, selecciona rúbricas y sube el PDF.")
        st.stop()
        
    st.warning("⚠️ PROCESO EN MARCHA: Por favor, NO cierres esta pestaña. Puedes minimizarla o cambiar de ventana, pero si la cierras, el servidor web apagará el motor de Inteligencia Artificial de forma irrevocable. El proceso tomará ~2 horas ininterrumpidas.")
    
    try:
        app_secrets = {
            "EMAIL_ADDRESS": st.secrets["EMAIL_ADDRESS"],
            "EMAIL_PASSWORD": st.secrets["EMAIL_PASSWORD"]
        }
    except Exception:
        app_secrets = {}
        st.error("Por favor, configura EMAIL_ADDRESS y EMAIL_PASSWORD en los secrets.")
        st.stop()
        
    client = get_gemini_client()
    file_bytes = uploaded_file.read()
    
    st.info("Leyendo y particionando archivo PDF en memoria...")
    chunks = extract_chunks(file_bytes, chunk_size=15)
    
    informe_final = []
    todas_las_referencias = []
    
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
        
    status_text.text("Fase DEDUPLICATE [Gemini]: Consolidando y eliminando observaciones redundantes...")
    informe_final = deduplicate_phase(client, informe_final)
    
    status_text.success("Evaluación de IA completada. Calculando puntaje.")
    total_score = 0
    for item in informe_final:
        res = item['resultado']
        total_score += int(res.get('puntaje', 0))
        
    st.info("Compilando reporte formal en LaTeX (pdflatex)...")
    
    def escape_user_data(text):
        if not text: return ""
        chars = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'}
        return "".join([chars.get(c, c) for c in text])

    def sanitize_ai_latex(text):
        if not text: return ""
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\u2212', '-')
        text = re.sub(r'[\x00-\x08\x0b-\x1f]', '', text)
        return text

    latex_content = r"\documentclass[12pt,a4paper]{article}" + "\n"
    latex_content += r"\usepackage[utf8]{inputenc}" + "\n"
    latex_content += r"\usepackage[T1]{fontenc}" + "\n"
    latex_content += r"\usepackage[spanish]{babel}" + "\n"
    latex_content += r"\usepackage[spanish]{csquotes}" + "\n"
    latex_content += r"\usepackage{geometry}" + "\n"
    latex_content += r"\usepackage{xcolor}" + "\n"
    latex_content += r"\usepackage[colorlinks=true,urlcolor=blue,linkcolor=black,citecolor=black]{hyperref}" + "\n"
    latex_content += r"\geometry{margin=2.5cm}" + "\n"
    latex_content += r"\begin{document}" + "\n\n"
    latex_content += r"\begin{center}" + "\n"
    latex_content += r"{\LARGE \textbf{Dictamen Oficial de Evaluación de Tesis}} \\ [0.5cm]" + "\n"
    latex_content += r"\end{center}" + "\n\n"
    
    latex_content += rf"\textbf{{Evaluador:}} {escape_user_data(evaluator_name)}\\" + "\n"
    latex_content += rf"\textbf{{Rol:}} {escape_user_data(evaluator_role)}\\" + "\n"
    latex_content += rf"\textbf{{Institución:}} {escape_user_data(university)}\\" + "\n\n"
    latex_content += r"\hrule\vspace{0.5cm}" + "\n\n"

    for item in informe_final:
        res = item['resultado']
        rubrica_esc = escape_user_data(item['rubrica'])
        puntaje = res.get('puntaje', 0)
        
        latex_content += rf"\subsection*{{{rubrica_esc} (Puntaje Asignado: {puntaje})}}" + "\n"
        
        observaciones = res.get('observaciones_narrativas', [])
        if observaciones:
            latex_content += r"\begin{itemize}" + "\n"
            for obs in observaciones:
                latex_content += rf"  \item {sanitize_ai_latex(obs)}" + "\n"
            latex_content += r"\end{itemize}" + "\n"
        
        referencias_bloque = res.get('referencias_apa', [])
        todas_las_referencias.extend(referencias_bloque)
        
        latex_content += r"\vspace{0.5cm}" + "\n\n"

    if todas_las_referencias:
        latex_content += r"\newpage" + "\n"
        latex_content += r"\section*{Referencias Bibliográficas Consolidadas}" + "\n"
        latex_content += r"\begin{list}{}{\setlength{\itemindent}{-1.27cm}\setlength{\leftmargin}{1.27cm}}" + "\n"
        referencias_unicas = sorted(list(set(todas_las_referencias)))
        for r in referencias_unicas:
            latex_content += rf"  \item {sanitize_ai_latex(r)}" + "\n"
        latex_content += r"\end{list}" + "\n\n"

    latex_content += r"\vfill\hrule\vspace{0.2cm}\begin{center}" + "\n"
    latex_content += rf"\textbf{{PUNTAJE GLOBAL OBTENIDO: {total_score}}}" + "\n"
    latex_content += r"\end{center}" + "\n\n"
    latex_content += r"\end{document}"

    os.makedirs("reportes_temp", exist_ok=True)
    tex_path = os.path.join("reportes_temp", "informe_oficial.tex")
    pdf_path = os.path.join("reportes_temp", "informe_oficial.pdf")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    compilado_ok = False
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    try:
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory=reportes_temp", tex_path], check=False, capture_output=True)
        if os.path.exists(pdf_path):
            compilado_ok = True
        else:
            stderr_log = result.stderr.decode('utf-8', errors='ignore') if result.stderr else ""
            stdout_log = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ""
            full_log = (stdout_log + "\n" + stderr_log)[-4000:]
            st.error("Error crítico de sintaxis en LaTeX:\n\n```text\n" + full_log + "\n```")
    except Exception as e:
        st.error(f"Excepción pdflatex: {e}")

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
        
        st.info("Enviando reporte de forma automática por correo...")
        try:
            emisor = app_secrets.get("EMAIL_ADDRESS", "")
            clave_app = app_secrets.get("EMAIL_PASSWORD", "")
            if not emisor or not clave_app:
                 st.error("No se encontró EMAIL_ADDRESS o EMAIL_PASSWORD en los secrets.")
            else:
                msg = EmailMessage()
                msg['Subject'] = 'Resultados de Evaluación de Tesis - Dictamen IA'
                msg['From'] = emisor
                msg['To'] = correo_destino
                msg.set_content(f"Estimado tesista/interesado,\n\nAdjunto sírvase encontrar el dictamen riguroso generado por {evaluator_name}.\n\nPuntaje Global: {total_score}\n\nAtentamente,\nRobot Calificador")
                
                msg.add_attachment(pdf_bytes, maintype='application', subtype='pdf', filename='Dictamen_Tesis.pdf')
                
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(emisor, clave_app)
                    server.send_message(msg)
                    
                st.success("📧 ¡Reporte enviado exitosamente por correo a " + correo_destino + "!")
        except Exception as smtp_err:
            st.error(f"Error SMTP al enviar correo: {smtp_err}")

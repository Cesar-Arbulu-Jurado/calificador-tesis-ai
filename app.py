import streamlit as st
import fitz  # PyMuPDF
import json
import os
import subprocess
from google import genai
import smtplib
from email.message import EmailMessage
import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup

# Configuración de página
st.set_page_config(page_title="Calificador Automático de Tesis", layout="wide")

st.title("Calificador Automático de Tesis (Multi-Agente Async + Inquisidor Bibliográfico)")
st.write("Sube el PDF, selecciona las rúbricas y obtén tu informe riguroso.")

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
    rigor_val = int(rigor_level.split(" ")[0])

# --- MAIN ÁREA: Carga de PDF y Rúbricas ---
st.header("3. Subir Tesis")
uploaded_file = st.file_uploader("Selecciona el PDF de la tesis", type="pdf")

st.header("4. Rúbricas a Evaluar")
db_path = os.path.join(os.path.dirname(__file__), "rubricas_extraidas.json")
if os.path.exists(db_path):
    with open(db_path, "r", encoding="utf-8") as f:
        rubricas_db = json.load(f)
else:
    st.error("⚠️ Error Crítico: No se encontró 'rubricas_extraidas.json'. El parseador no se ejecutó.")
    st.stop()

rubricas_disponibles = list(rubricas_db.keys())
selected_rubrics = st.multiselect("Selecciona las secciones inalterables de LaTeX", rubricas_disponibles, default=rubricas_disponibles[:2])

@st.cache_data
def load_thesis_writer_rules():
    txt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "1. Thesis_review_writer.md")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    txt_path_local = os.path.join(os.path.dirname(__file__), "1. Thesis_review_writer.md")
    if os.path.exists(txt_path_local):
         with open(txt_path_local, "r", encoding="utf-8") as f:
            return f.read()
    return "NORMA: Redactar de forma objetiva, densa y en formato APA 7ma Edición."

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

sema = asyncio.Semaphore(5)

async def route_thesis_sections(client, chunks, rubrics):
    from google.genai import types
    chunks_summary = ""
    for i, chunk in enumerate(chunks):
        chunks_summary += f"--- CHUNK ID {i} ---\n{chunk[:2000]}...\n\n"
        
    prompt = f"""
    Eres el Agente Director (Enrutador Jerárquico) de un Sistema Multi-Agente.
    Tu misión es leer los resúmenes de los bloques (Chunks) de la tesis y decidir qué bloques contienen información útil para evaluar las siguientes rúbricas:
    {json.dumps(rubrics, ensure_ascii=False)}
    
    Resúmenes de los bloques:
    {chunks_summary}
    
    Devuelve EXACTAMENTE un JSON mapeando cada rúbrica con un arreglo de números enteros (Chunk IDs) relevantes. Si una rúbrica en teoría debe aplicarse transversalmente, asígnale todos los IDs.
    Formato estricto:
    {{
       "Nombre de la Rúbrica 1": [0, 1, 3],
       "Nombre de la Rúbrica 2": [4, 5]
    }}
    """
    fallbacks = ['gemini-1.5-flash', 'gemini-2.5-flash']
    for m in fallbacks:
        try:
            res = await client.aio.models.generate_content(
                model=m,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(res.text)
        except Exception:
            continue
    return {r: list(range(len(chunks))) for r in rubrics}

async def map_phase_async(client, chunk_text, rubric_title, rubric_content, rigor):
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
    async with sema:
        for m in modelos_map:
            try:
                res = await client.aio.models.generate_content(
                    model=m, 
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                return json.loads(res.text) if res.text else []
            except Exception:
                continue
    return []

async def reduce_phase_async(client, rubric_title, rubric_content, map_results, rigor, max_errores):
    from google.genai import types
    evidences = json.dumps(map_results)
    thesis_rules = load_thesis_writer_rules()
    
    prompt = f"""
    Eres un Evaluador de Tesis de Maestría de Alto Nivel con perfil puramente de Ingeniería Civil. Rigor: {rigor}.
    Dimensión a evaluar: {rubric_title}
    
    CRITERIOS INALTERABLES (Matriz LaTeX original):
    {rubric_content}
    
    Evidencias extraídas temporalmente de la tesis por tus agentes subordinados: {evidences}
    
    A continuación, tienes las REGLAS ABSOLUTAS del Meta-Prompt que debes obedecer meticulosamente. Presta especial atención a la Pestaña 8, 9, 10 y 11 sobre estructura, prohibiciones léxicas y formato LaTeX estricto de Ingeniería Civil:
    
    ------ METAPROMPT THESIS_REVIEW_WRITER ------
    {thesis_rules}
    ---------------------------------------------
    
    INSTRUCCIONES FINALES DE EMISIÓN DE DICTAMEN:
    1. Ejecuta el dictamen sobre la dimensión solicitada cumpliendo irrestrictamente TODAS las prohibiciones léxicas, estilo narrativo y estructura matemática declarada.
    2. Identifica hasta {max_errores} observaciones basándote estrictamente en las evidencias reales. CERO FABRICACIÓN ACADÉMICA.
    3. TODO AUTOR debe aparecer en "referencias_apa" en FORMATO APA 7ma Edición estricto. Si le inventas o hallas un enlace URL que sea vivo, enciérralo usando el comando LaTeX \\url{{enlace_aqui}}.
    4. Asigna "puntaje" entero.
    
    Devuelve EXACTAMENTE UN JSON PURO con estas claves:
    {{
      "deep_research_analysis": "Ejecuta aquí tu razonamiento interno objetivo y riguroso...",
      "observaciones_narrativas": [
        "Párrafo narrativo impecable continuo sin etiquetas en negritas tal como manda el Prompt...",
        "Párrafo 2..."
      ],
      "referencias_apa": ["Smith, A. (2020). Título...", "Ref 2..."],
      "puntaje": 0
    }}
    """
    modelos_reduce = ['gemini-3-pro-preview', 'gemini-3-pro', 'gemini-2.5-pro', 'gemini-1.5-pro-latest']
    async with sema:
        for m in modelos_reduce:
            try:
                res = await client.aio.models.generate_content(
                    model=m, 
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                parsed = json.loads(res.text)
                if not isinstance(parsed, dict):
                    if isinstance(parsed, list) and len(parsed) > 0: return parsed[0]
                    raise Exception("Not a dict")
                return parsed
            except Exception:
                continue
    return {"observaciones_narrativas": [f"Error técnico de consolidación en modelo Pro."], "referencias_apa": [], "puntaje": 0}

async def deduplicate_phase_async(client, informe_final):
    from google.genai import types
    prompt = f"""
    Eres un analista experto en consolidación semántica de informes técnicos.
    JSON con resultados preliminares y rúbricas: 
    {json.dumps(informe_final, ensure_ascii=False)}
    
    INSTRUCCIONES ESTRICTAS:
    1. Compara diferencialmente el texto de TODAS las observaciones ("observaciones_narrativas") en el reporte.
    2. Si hallas similitud o redundancia >= 80% criticando exactamente el mismo párrafo fundamental: 
       Consolídalas en UNA sola observación narrativo-continua en la rúbrica donde apareció por primera vez, integrando orgánicamente ambos detalles. ELIMINA la redundante.
       (Mantén el uso estricto de LaTeX, de signos y normativas de APA insertados previamente).
    3. Respeta intactas las observaciones singulares (<80%).
    4. MANTÉN PUNTAJES INTACTOS. No modifiques "puntaje".
    5. Unifica referencias APA necesarias.
    
    Devuelve EXACTAMENTE el esquema JSON original reconstruido con tus fusiones (lista de diccionarios por rúbrica):
    [ {{"rubrica": "...", "resultado": {{"deep_research_analysis": "...", "observaciones_narrativas": ["..."], "referencias_apa": ["..."], "puntaje": 0}} }} ]
    """
    modelos_dedup = ['gemini-2.5-pro', 'gemini-1.5-pro-latest', 'gemini-3-pro-preview']
    async with sema:
        for m in modelos_dedup:
            try:
                res = await client.aio.models.generate_content(
                    model=m, 
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                parsed = json.loads(res.text)
                return parsed if isinstance(parsed, list) else informe_final
            except Exception:
                continue
    return informe_final

async def url_is_valid_and_matches(ref, client):
    url_match = re.search(r'\\url\{([^}]+)\}', ref)
    if not url_match:
        url_match = re.search(r'(https?://[^\s]+)', ref)
        
    if not url_match:
        return ref # Sin URL, se aprueba localmente
        
    url = url_match.group(1)
    
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            async with session.get(url, headers=headers, timeout=12) as response:
                if response.status == 404: 
                    return None # Enlace roto o inexistente
                if response.status != 200:
                    return ref # Pasa de largo si es un 403 Forbidden (ej. IEEE/Elsevier anti-bot)

                html = await response.text()
                
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else ""
        text_preview = soup.get_text()[:400].replace('\n', ' ')
        
        from google.genai import types
        prompt = f"""
        El autor citó esta referencia en su tesis: "{ref}"
        Hemos abierto la URL y el título web es: "{title}"
        Vista previa del contenido de la URL: "{text_preview}"
        
        ¿Esta página web corresponde genuinamente a la referencia investigativa?
        (A veces los tesistas inventan enlaces que redirigen a archivos aleatorios).
        Responde "VALIDO" si tiene correlación lógica o está protegido por un muro de pago, o "FALSO" si manda a otra cosa estúpida o disparatada.
        """
        res = await client.aio.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
        )
        if "FALSO" in res.text.upper():
            return None # Eliminado por el Inquisidor
            
        return ref
    except Exception:
        # Fallo de conexión (DNS, Timeout), asumimos que el tesista tiene razón para no fallar injustamente
        return ref

async def verify_bibliography_agent_async(client, informe_final, status_text):
    status_text.info("🌐 Agente Inquisidor: Verificando veracidad web y URLs de las referencias APA...")
    import copy
    informe_verificado = copy.deepcopy(informe_final)
    
    verifications = []
    mapping = []
    
    for i, item in enumerate(informe_verificado):
        res = item.get('resultado', {})
        referencias = res.get('referencias_apa', [])
        for j, ref in enumerate(referencias):
            mapping.append((i, j))
            verifications.append(url_is_valid_and_matches(ref, client))
            
    if verifications:
        resultados = await asyncio.gather(*verifications)
        for (i, j), valid_ref in zip(mapping, resultados):
            if valid_ref is None:
                informe_verificado[i]['resultado']['referencias_apa'][j] = None
                
    for item in informe_verificado:
        res = item.get('resultado', {})
        if 'referencias_apa' in res:
            res['referencias_apa'] = [r for r in res['referencias_apa'] if r is not None]
            
    return informe_verificado

async def procesar_tesis_async(client, chunks, selected_rubrics, rubricas_db, rigor_val, max_observaciones, progress_bar, status_text):
    status_text.info("🧭 Agente Director: Enrutando bloques de la tesis jerárquicamente...")
    router_map = await route_thesis_sections(client, chunks, selected_rubrics)
    progress_bar.progress(0.1)
    
    status_text.info("⚡ Agentes Especialistas: Ejecutando Incursión Asíncrona Masiva (MAP) en paralelo...")
    map_tasks = {}
    for rubric in selected_rubrics:
        rubrica_texto_latex = rubricas_db[rubric]
        chunk_indices = router_map.get(rubric, list(range(len(chunks))))
        
        if not isinstance(chunk_indices, list):
            chunk_indices = list(range(len(chunks)))
            
        for idx in chunk_indices:
            try: 
                idx = int(idx)
                if 0 <= idx < len(chunks):
                    map_tasks[(rubric, idx)] = asyncio.create_task(
                        map_phase_async(client, chunks[idx], rubric, rubrica_texto_latex, rigor_val)
                    )
            except ValueError:
                continue
                
    if map_tasks:
        await asyncio.gather(*map_tasks.values())
    progress_bar.progress(0.5)
    
    status_text.info("🧠 Agentes Consolidadores: Sintetizando descubrimientos bajo normativa MD...")
    reduce_tasks = {}
    for rubric in selected_rubrics:
        chunk_indices = router_map.get(rubric, list(range(len(chunks))))
        if not isinstance(chunk_indices, list):
            chunk_indices = list(range(len(chunks)))
            
        all_candidates = []
        for idx in chunk_indices:
            try:
                idx = int(idx)
                if 0 <= idx < len(chunks) and (rubric, idx) in map_tasks:
                    res = map_tasks[(rubric, idx)].result()
                    if res:
                        all_candidates.extend(res)
            except ValueError:
                continue
                    
        rubrica_texto_latex = rubricas_db[rubric]
        reduce_tasks[rubric] = asyncio.create_task(
            reduce_phase_async(client, rubric, rubrica_texto_latex, all_candidates, rigor_val, max_observaciones)
        )
        
    if reduce_tasks:
        await asyncio.gather(*reduce_tasks.values())
    progress_bar.progress(0.7)
    
    status_text.info("🧬 Agente Árbitro: Deduplicando semántica y uniendo el compendio final...")
    informe_final_raw = []
    for rubric in selected_rubrics:
        informe_final_raw.append({
            "rubrica": rubric,
            "resultado": reduce_tasks[rubric].result()
        })
        
    informe_final = await deduplicate_phase_async(client, informe_final_raw)
    progress_bar.progress(0.9)
    
    informe_verificado = await verify_bibliography_agent_async(client, informe_final, status_text)
    
    progress_bar.progress(1.0)
    status_text.success("¡Operación Multi-Agente coronada con éxito en tiempo ultra-reducido!")
    return informe_verificado

if st.button("Iniciar Evaluación Rápida (Multi-Agente Async)", type="primary"):
    if not uploaded_file or not evaluator_name or not selected_rubrics or not correo_destino:
        st.warning("Completa los datos del evaluador, el correo de destino, selecciona rúbricas y sube el PDF.")
        st.stop()
        
    st.info("⚠️ PROCESO ULTRA-RÁPIDO EN MARCHA: Por favor, espera sin cerrar la ventana. El nuevo Motor Asíncrono completará todo en pocos minutos.")
    
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
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    informe_final = loop.run_until_complete(
        procesar_tesis_async(client, chunks, selected_rubrics, rubricas_db, rigor_val, max_observaciones, progress_bar, status_text)
    )
    
    total_score = 0
    todas_las_referencias = []
    if not hasattr(informe_final, "__iter__"): informe_final = []
    
    for item in informe_final:
        res = item.get('resultado', {})
        total_score += int(res.get('puntaje', 0))
        todas_las_referencias.extend(res.get('referencias_apa', []))
        
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
        
        text = re.sub(r'(?<!\\)&', r'\&', text)
        text = re.sub(r'(?<!\\)%', r'\%', text)
        text = re.sub(r'(?<!\\)#', r'\#', text)
        
        stack = 0
        balanced = []
        for char in text:
            if char == '{':
                stack += 1
                balanced.append(char)
            elif char == '}':
                if stack > 0:
                    stack -= 1
                    balanced.append(char)
            else:
                balanced.append(char)
                
        while stack > 0:
            balanced.append('}')
            stack -= 1
            
        text = "".join(balanced)
        
        if text.count('$') % 2 != 0:
            text += '$'
            
        return text

    latex_content = r"\documentclass[12pt,a4paper]{article}" + "\n"
    latex_content += r"\usepackage[utf8]{inputenc}" + "\n"
    latex_content += r"\usepackage[T1]{fontenc}" + "\n"
    latex_content += r"\usepackage[spanish]{babel}" + "\n"
    latex_content += r"\usepackage[spanish]{csquotes}" + "\n"
    latex_content += r"\usepackage{geometry}" + "\n"
    latex_content += r"\usepackage{xcolor}" + "\n"
    latex_content += r"\usepackage{microtype}" + "\n" # Corrector de colisiones y justificación
    latex_content += r"\usepackage[colorlinks=true,urlcolor=blue,linkcolor=black,citecolor=black]{hyperref}" + "\n"
    latex_content += r"\geometry{margin=2.5cm}" + "\n"
    latex_content += r"\begin{document}" + "\n"
    latex_content += r"\sloppy" + "\n\n" # Relajador del espaciamiento interpalabra
    
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
                latex_content += rf"  \item\relax {sanitize_ai_latex(obs)}" + "\n"
            latex_content += r"\end{itemize}" + "\n"
        
        latex_content += r"\vspace{0.5cm}" + "\n\n"

    if todas_las_referencias:
        latex_content += r"\newpage" + "\n"
        latex_content += r"\section*{Referencias Bibliográficas Consolidadas}" + "\n"
        latex_content += r"\begin{list}{}{\setlength{\itemindent}{-1.27cm}\setlength{\leftmargin}{1.27cm}}" + "\n"
        referencias_unicas = sorted(list(set(todas_las_referencias)))
        for r in referencias_unicas:
            latex_content += rf"  \item\relax {sanitize_ai_latex(r)}" + "\n"
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
            mime="application/pdf",
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
                msg.set_content(f"Estimado tesista/interesado,\n\nAdjunto sírvase encontrar el dictamen riguroso generado con IA por {evaluator_name}.\n\nPuntaje Global: {total_score}\n\nAtentamente,\nRobot Calificador Multi-Agente")
                
                msg.add_attachment(pdf_bytes, maintype='application', subtype='pdf', filename='Dictamen_Tesis.pdf')
                
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(emisor, clave_app)
                    server.send_message(msg)
                    
                st.success(f"📧 ¡Reporte Multi-agente enviado exitosamente por correo a {correo_destino}!")
        except Exception as smtp_err:
            st.error(f"Error SMTP al enviar correo: {smtp_err}")

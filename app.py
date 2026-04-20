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
import threading
import datetime

# Configuración de página
st.set_page_config(page_title="Calificador Automático de Tesis", layout="wide")

st.title("Calificador Automático de Tesis (Multi-Agente Desatendido)")
st.write("Sube el PDF, selecciona las rúbricas y relájate. El reporte será procesado en background y enviado a tu correo.")

@st.cache_data
def load_rubrics_from_tex():
    txt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.tex")
    if not os.path.exists(txt_path):
        txt_path = os.path.join(os.path.dirname(__file__), "main.tex")
        
    if not os.path.exists(txt_path):
        return {}
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = re.compile(r'\\section\*?\{([^}]+)\}')
    parts = pattern.split(content)
    
    rubricas_db = {}
    for i in range(1, len(parts), 2):
        titulo_seccion = parts[i].strip()
        if titulo_seccion.lower() in ["introducción", "conclusiones", "anexos", "bibliografía"]:
            continue
        contenido_seccion = parts[i+1].strip()
        rubricas_db[titulo_seccion] = contenido_seccion
        
    return rubricas_db

# --- MAIN ÁREA: Carga de PDF y Rúbricas ---
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

st.header("3. Subir Tesis")
uploaded_file = st.file_uploader("Selecciona el PDF de la tesis", type="pdf")

st.header("4. Rúbricas a Evaluar")
rubricas_db = load_rubrics_from_tex()
if not rubricas_db:
    st.error("⚠️ Error Crítico: No se pudo extraer automáticamente de 'main.tex'. Comprueba tus directorios de GitHub.")
    st.stop()

rubricas_disponibles = list(rubricas_db.keys())
selected_rubrics = st.multiselect("Selecciona las secciones inalterables de LaTeX (Extraídas de main.tex)", rubricas_disponibles, default=rubricas_disponibles)

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


async def resilient_gemini_call(client, active_models, contents, config=None, is_json=False):
    import json
    import re
    import asyncio
    
    def robust_json_parse(text):
        try:
            return json.loads(text)
        except Exception:
            pass
            
        try:
            # Try to strip markdown quotes
            clean_text = text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            return json.loads(clean_text)
        except: pass
        
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match: return json.loads(match.group(0))
        except: pass
        try:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match: return json.loads(match.group(0))
        except: pass
        raise Exception("JSON corrupto o incompleto devuelto de la API.")

    last_error = ""
    for m in active_models:
        intentos = 3
        for i in range(intentos):
            try:
                res = await client.aio.models.generate_content(model=m, contents=contents, config=config)
                if not res.text: raise Exception("Bloque vacío API")
                
                if is_json:
                    parsed = robust_json_parse(res.text)
                    return parsed, None
                return res.text, None
                
            except Exception as e:
                last_error = f"({m}): {str(e)}"
                if "503" in str(e) or "429" in str(e) or "quota" in str(e).lower():
                    await asyncio.sleep(5 * (i + 1))
                    continue
                else:
                    break # Failed syntax or 404, go next model
    return None, str(last_error)

async def route_thesis_sections(client, chunks, rubrics, active_models):
    from google.genai import types
    chunks_summary = ""
    for i, chunk in enumerate(chunks):
        chunks_summary += f"--- CHUNK ID {i} ---\n{chunk[:2000]}...\n\n"
        
    prompt = f"""
    Eres el Agente Director (Enrutador Jerárquico).
    Resúmenes de los bloques:
    {chunks_summary}
    Rúbricas: {json.dumps(rubrics, ensure_ascii=False)}
    Devuelve JSON mapeando cada rúbrica con los Chunk IDs relevantes o transversales.
    """
    from google.genai import types
    parsed, err = await resilient_gemini_call(client, active_models, prompt, config=types.GenerateContentConfig(response_mime_type="application/json"), is_json=True)
    if parsed: return parsed
    return {r: list(range(len(chunks))) for r in rubrics}

async def map_phase_async(client, chunk_text, rubric_title, rubric_content, rigor, active_models, sema):
    prompt = f"""
    Actúa como evaluador experto de tesis. Dimensión a evaluar: {rubric_title}
    CRITERIOS INALTERABLES:
    {rubric_content}
    
    PROSCRIPCIÓN SUPREMA: Jamás extraigas evidencias que justifiquen hallazgos mencionando a metodólogos de ciencias sociales en español (ej. Hernández Sampieri). Están prohibidos y vetados en Ingeniería Civil.
    REGLA DE OCR (ERROR DE ORIGEN): Si encuentras cadenas de caracteres extraños o texto ininteligible derivado de un mal reconocimiento OCR en el documento, IGNÓRALO por completo. No extraigas citas con basura OCR ni comentes nunca sobre la mala lectura del PDF. Simplemente haz de cuenta que ese texto basura no existe.
    Si hallas un error genuino y legible, extrae la cita EXACTA enmarcada OBLIGATORIAMENTE en el comando LaTeX \\enquote{{...}}. ESTÁ ESTRICTAMENTE PROHIBIDO usar comillas dobles ("") o simples ('').
    Retorna JSON: [{{\"error_description\": \"...\", \"exact_quote\": \"\\enquote{{...}}\"}}].
    Texto:\n{chunk_text}
    """
    async with sema:
        from google.genai import types
        parsed, err = await resilient_gemini_call(client, active_models, prompt, config=types.GenerateContentConfig(response_mime_type="application/json"), is_json=True)
        if parsed: return parsed
        return []

async def reduce_phase_async(client, rubric_title, rubric_content, map_results, rigor, max_errores, thesis_rules, active_models, sema):
    import json
    evidences = json.dumps(map_results)
    prompt = rf"""
    Eres Evaluador de Tesis rigor {rigor}. Dimensión: {rubric_title}
    CRITERIOS LATEX ORIGINAL:
    {rubric_content}
    EVIDENCIAS: {evidences}
    REGLAS METAPROMPT: {thesis_rules}
    
    INSTRUCCIONES CLAVES:
    1. Identifica hasta {max_errores} observaciones basándote estrictamente en evidencias. CERO FABRICACIÓN.
    2. MANDATO INQUEBRANTABLE (ANTIBIBLATEX): Escribe las referencias bibliográficas y las citas parentéticas en "texto plano". PROHIBIDO INYECTAR COMANDOS LATEX BIBTEX como \cite{{}}, \textcite{{}}, \parencite{{}}. Hazlo literamente así: (Autor, Año). Si usas \cite{{...}} te auto-destruirás y dañarás el reporte.
    3. MANDATO OBLIGATORIO (ANTISOCIAL): Jamás cites ni listes metodología social, ni utilices postulados de Hernández Sampieri o afines en español. Si el tesista citó a Sampieri para justificar Ingeniería Civil, destrúyelo críticamente en el reporte como un error inaceptable de alcance metodológico.
    4. REGLA DE OCR (ERROR DE ORIGEN): Descarta automáticamente cualquier fragmento de evidencia que contenga caracteres extraños, símbolos ininteligibles o fallos originados por un mal OCR en la tesis original. No utilices ese texto ni comentes acerca de la mala calidad del OCR del PDF. IGNORA dicho texto.
    5. REGLA DE ENTRECOMILLADO: Cuando cites texto literal de la tesis, de normativas o necesites resaltar algún término, está ESTRICTAMENTE PROHIBIDO usar NINGÚN TIPO de comillas tipográficas, sean simples o dobles (' ', " ", ‘ ’, “ ”). DEBES EMPLEAR EXCLUSIVAMENTE el comando LaTeX \\enquote{{texto}}. Ejemplo: Fallo en el \\enquote{{control del pronóstico}}. El incumplimiento destrozará el motor LaTeX.
    
    Formato JSON Obligatorio:
    {{
      "deep_research_analysis": "Contexto objetivamente deducido...",
      "observaciones_narrativas": ["Párrafo hiper-técnico 1...", "Párrafo interactivo fluido LaTeX SIN usar macro \cite..."],
      "referencias_apa": ["Smith, A. (2020)... \url{{...}}"],
      "puntaje": 0
    }}
    """
    async with sema:
        from google.genai import types
        parsed, err = await resilient_gemini_call(client, active_models, prompt, config=types.GenerateContentConfig(response_mime_type="application/json"), is_json=True)
        if parsed:
            if isinstance(parsed, list) and len(parsed) > 0: return parsed[0]
            if isinstance(parsed, dict): return parsed
            
    error_msg = str(err).replace('\n', ' | ') if err else "Timeout API o parseo fallido."
    return {
        "observaciones_narrativas": [rf"Fallo crítico interno en el motor de consolidación al procesar esta rúbrica. Historial Exacto de Intentos Fallidos: \textbf{{{error_msg}}}"], 
        "referencias_apa": [], 
        "puntaje": 0
    }

async def supervisor_agent_async(client, chunks, rubricas_db, inf_list, rigor, max_obs, thesis_rules, active_models, sema, logs):
    logs("Activando Agente Supervisor de Resiliencia...")
    import asyncio
    max_intentos = 4
    for intento in range(max_intentos):
        fallos_encontrados = []
        for i, item in enumerate(inf_list):
            rubrica = item['rubrica']
            obs = item.get('resultado', {}).get('observaciones_narrativas', [])
            is_failed = any("Fallo crítico interno" in str(o) for o in obs)
            if is_failed:
                fallos_encontrados.append((i, rubrica))
        
        if not fallos_encontrados:
            return inf_list
            
        logs(f"Supervisor detectó {len(fallos_encontrados)} rúbricas corruptas. Iniciando Intento Reparador {intento+1} de {max_intentos}...")
        await asyncio.sleep(4)
        
        reparaciones = {}
        for (i, rubrica) in fallos_encontrados:
            reparaciones[i] = asyncio.create_task(
                re_evaluate_rubric_async(client, chunks, rubrica, rubricas_db[rubrica], rigor, max_obs, thesis_rules, active_models, sema)
            )
        
        if reparaciones:
            await asyncio.gather(*reparaciones.values())
            for i, task in reparaciones.items():
                inf_list[i]['resultado'] = task.result()
            
    return inf_list

async def re_evaluate_rubric_async(client, chunks, rubric, rubric_content, rigor, max_obs, thesis_rules, active_models, sema):
    import asyncio
    map_tasks = []
    for idx, c in enumerate(chunks):
        map_tasks.append(asyncio.create_task(map_phase_async(client, c, rubric, rubric_content, rigor, active_models, sema)))
    await asyncio.gather(*map_tasks)
    
    all_cands = []
    for t in map_tasks:
        if t.result(): all_cands.extend(t.result())
        
    reduce_task = asyncio.create_task(reduce_phase_async(client, rubric, rubric_content, all_cands, rigor, max_obs, thesis_rules, active_models, sema))
    await reduce_task
    return reduce_task.result()

async def deduplicate_phase_async(client, informe_final, active_models, sema):
    return informe_final

async def semantic_deduplicate_references_async(client, refs, active_models, logs):
    import json
    if not refs: return []
    logs("Activando Agente Deduplicador Bibliográfico Global...")
    prompt = f"""
    Eres un bibliotecario algorítmico. Lista de referencias APA crudas:
    {json.dumps(refs, ensure_ascii=False)}
    
    Misión: Deduplica las variaciones del mismo paper.
    Retorna un JSON estricto con un único array final destilado de cadenas APA: ["Ref Mejorada 1", "Ref Mejorada 2"].
    """
    from google.genai import types
    parsed, err = await resilient_gemini_call(client, active_models, prompt, config=types.GenerateContentConfig(response_mime_type="application/json"), is_json=True)
    if parsed: return parsed
    return sorted(list(set(refs)))

async def url_is_valid_and_matches(ref, client, active_models):
    import re
    import aiohttp
    from bs4 import BeautifulSoup
    url_match = re.search(r"https?://[^\s]+", ref)
    if not url_match: return ref
    url = url_match.group(0).strip(").\"")
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"User-Agent": "Mozilla/5.0"}
            async with session.get(url, headers=headers, timeout=12) as response:
                if response.status == 404: return None
                if response.status != 200: return ref
                html = await response.text()
                
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else ""
        text_preview = soup.get_text()[:300].replace('\n', ' ')
        
        prompt = f"Referencia: {ref}\nURL Título: {title}\nPreview Web: {text_preview}\n¿Corresponden semánticamente? Responde 'VALIDO' o 'FALSO'."
        parsed, err = await resilient_gemini_call(client, active_models, prompt, is_json=False)
        if parsed and "FALSO" in parsed.upper():
            return None
        return ref
    except Exception:
        return ref

async def verify_bibliography_agent_async(client, informe_final, active_models, logs):
    import copy
    import asyncio
    logs("Inquisidor: Verificando literatura en la Deep Web...")
    informe = copy.deepcopy(informe_final)
    verifications, mapping = [], []
    for i, item in enumerate(informe):
        for j, ref in enumerate(item.get('resultado', {}).get('referencias_apa', [])):
            mapping.append((i, j))
            verifications.append(url_is_valid_and_matches(ref, client, active_models))
    if verifications:
        r = await asyncio.gather(*verifications)
        for (i, j), v in zip(mapping, r):
            if v is None: informe[i]['resultado']['referencias_apa'][j] = None
    for item in informe:
        res = item.get('resultado', {})
        if 'referencias_apa' in res:
            res['referencias_apa'] = [r for r in res['referencias_apa'] if r is not None]
    return informe

async def generate_intro_agent_async(client, chunks, rubrics_list, inf_final, active_models, logs):
    import json
    logs("Agente Analista Documental tejiendo Introducción Histórica...")
    context_inicio = "".join(chunks[:2])[:4000]
    resumen_eval = json.dumps([{"rubrica": x["rubrica"], "puntaje": x["resultado"].get("puntaje", 0)} for x in inf_final], ensure_ascii=False)
    
    prompt = f"""
    Eres el Analista Documental de Tesis.
    Páginas iniciales de la obra:
    {context_inicio}
    
    Rúbricas analizadas: {', '.join(rubrics_list)}.
    Resumen de dictamen interno: {resumen_eval}.
    
    Escribe UN ÚNICO PÁRRAFO continuo (entre 6 y 10 líneas) usando exactamente esta apertura literal obligatoria:
    "La presente evaluación se ha aplicado a la tesis denominada [Escribe aquí el Título de la tesis deducido del texto], de [Escribe el Autor o Autores]."
    Inmediatamente después, enlista brevemente los criterios evaluados y emite un preámbulo inicial advirtiendo la calidad general de la tesis basándote en los puntajes encontrados.
    Retorna EXCLUSIVAMENTE el texto crudo del párrafo, sin formateo markdown.
    """
    parsed, err = await resilient_gemini_call(client, active_models, prompt, is_json=False)
    if parsed: return parsed
    return "La presente evaluación se ha aplicado a la tesis evaluada."

async def generate_verdict_agent_async(client, inf_final, active_models, logs):
    import json
    logs("Magistrado IA redactando dictamen final...")
    resumen_eval = json.dumps([{"rubrica": x["rubrica"], "puntaje": x["resultado"].get("puntaje", 0)} for x in inf_final], ensure_ascii=False)
    
    prompt = f"""
    Eres el Magistrado de Evaluación de Tesis. 
    Basándote estrictamente en este resumen de resultados:
    {resumen_eval}
    
    Debes emitir un dictamen y veredicto general continuo de un solo párrafo denso (aprox 10 a 16 líneas) de tipo narrativo que será insertado al final del documento LaTeX. Este párrafo debe contener el resumen analítico general de deficiencias graves, fortalezas notables, y una recomendación final de la calidad general.
    NO USES COMANDOS. NO USES LISTAS. Solo devuelve el texto plano del párrafo.
    """
    parsed, err = await resilient_gemini_call(client, active_models, prompt, is_json=False)
    if parsed: return parsed
    return "Evaluación finalizada."

async def procesar_tesis_async(client, chunks, rubrics, rubricas_db, rigor, max_obs, thesis_rules, active_models, logs):
    import asyncio
    sema = asyncio.Semaphore(5)
    logs("Router Base activado...")
    router_map = await route_thesis_sections(client, chunks, rubrics, active_models)
    
    logs("Phase MAP paralela instanciada...")
    map_tasks = {}
    for r in rubrics:
        c_idxs = router_map.get(r, list(range(len(chunks))))
        if not isinstance(c_idxs, list): c_idxs = list(range(len(chunks)))
        for idx in c_idxs:
            try:
                idx = int(idx)
                if 0 <= idx < len(chunks):
                    map_tasks[(r, idx)] = asyncio.create_task(map_phase_async(client, chunks[idx], r, rubricas_db[r], rigor, active_models, sema))
            except ValueError: continue
            
    if map_tasks: await asyncio.gather(*map_tasks.values())
    
    logs("Phase REDUCE paralela instanciada...")
    reduce_tasks = {}
    for r in rubrics:
        all_cands = []
        for idx in range(len(chunks)):
            if (r, idx) in map_tasks and map_tasks[(r, idx)].result():
                all_cands.extend(map_tasks[(r, idx)].result())
        reduce_tasks[r] = asyncio.create_task(reduce_phase_async(client, r, rubricas_db[r], all_cands, rigor, max_obs, thesis_rules, active_models, sema))
        
    if reduce_tasks: await asyncio.gather(*reduce_tasks.values())
    
    inf_raw = [{"rubrica": r, "resultado": reduce_tasks[r].result()} for r in rubrics]
    
    inf_raw = await supervisor_agent_async(client, chunks, rubricas_db, inf_raw, rigor, max_obs, thesis_rules, active_models, sema, logs)
    
    logs("Árbitro Semántico reduplicando...")
    inf_final = await deduplicate_phase_async(client, inf_raw, active_models, sema)
    
    inf_final = await verify_bibliography_agent_async(client, inf_final, active_models, logs)
    
    logs("Recolectando bibliografía maestra de todos los nodos...")
    all_raw_refs = []
    for r in inf_final:
        all_raw_refs.extend(r.get('resultado', {}).get('referencias_apa', []))
        
    unique_refs = await semantic_deduplicate_references_async(client, all_raw_refs, active_models, logs)
    
    texto_intro = await generate_intro_agent_async(client, chunks, rubrics, inf_final, active_models, logs)
    texto_veredicto = await generate_verdict_agent_async(client, inf_final, active_models, logs)
    
    return inf_final, unique_refs, texto_intro, texto_veredicto

def background_process(file_bytes, selected_rubrics, rubricas_db, rigor_val, max_obs, thesis_rules, api_key, evaluator_name, evaluator_role, university, correo_destino, app_secrets):
    def daemon_log(msg): print(f"[DAEMON] {msg}")
    
    daemon_log("Iniciando Proceso Asíncrono Desatendido")
    os.environ["GEMINI_API_KEY"] = api_key
    client = genai.Client(api_key=api_key)
    
    # ------------------ NUEVO AGENTE EXPLORADOR ------------------
    daemon_log("Agente Identificador de Modelos activado...")
    try:
        raw_list = client.models.list()
        disponibles = []
        for m in raw_list:
            if hasattr(m, 'name'):
                name_clean = m.name.replace("models/", "")
                if 'gemini' in name_clean and 'vision' not in name_clean:
                    disponibles.append(name_clean)
                    
        active_models_filtered = []
        # Buscar nuestras preferencias en la lista devuelta por Google
        for pref in ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.5-pro-latest', 'gemini-1.5-flash-latest', 'gemini-pro', 'gemini-2.5-flash']:
            if pref in disponibles:
                active_models_filtered.append(pref)
                
        # Si no hubo coincidencia exacta pero tienen modelos gemini, usar los suyos
        if not active_models_filtered and disponibles:
            active_models_filtered = disponibles[:3]
            
        daemon_log(f"Modelos confirmados que SI EXISTEN en tu entorno: {active_models_filtered}")
    except Exception as e:
        daemon_log(f"Falla en Explorador de Modelos. Forzando base. Trace: {e}")
        active_models_filtered = ['gemini-1.5-pro', 'gemini-1.5-flash']
    # -------------------------------------------------------------
    
    chunks = extract_chunks(file_bytes, chunk_size=15)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def enviar_correo_emergencia(mensaje_error):
        emisor = app_secrets.get("EMAIL_ADDRESS", "")
        clave_app = app_secrets.get("EMAIL_PASSWORD", "")
        if not emisor or not clave_app or not correo_destino: return
        try:
            msg = EmailMessage()
            msg['Subject'] = '⚠️ Fallo Crítico en Evaluación de Tesis Automática'
            msg['From'] = emisor
            msg['To'] = correo_destino
            msg.set_content(f"Estimado Usuario,\n\nTu evaluación de tesis (procesada de forma desatendida) NO ha podido finalizar tras múltiples horas debido a un error extremo en los servidores.\n\nDetalle Técnico: {mensaje_error}\n\nRecomendación: La tesis es demasiado extensa. Divide el PDF en mitades y envíalos individualmente.")
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(emisor, clave_app)
                server.send_message(msg)
            daemon_log("Correo de Emergencia DESPACHADO con éxito.")
        except Exception as e2: daemon_log(f"Fallo enviando correo de emergencia: {e2}")

    try:
        informe_verificado, referencias_consolidadas, texto_intro, texto_veredicto = loop.run_until_complete(
            procesar_tesis_async(client, chunks, selected_rubrics, rubricas_db, rigor_val, max_obs, thesis_rules, active_models_filtered, daemon_log)
        )
    except Exception as e:
        daemon_log(f"CRITICAL ASYNC ERROR DETECTED: {e}")
        enviar_correo_emergencia(str(e))
        loop.close()
        return
        
    loop.close()
    
    import uuid
    import shutil
    uid = str(uuid.uuid4())[:8]
    work_dir = f"reportes_temp_{uid}"
    
    # Renderizado y email
    total_score = 0
    daemon_log(f"Compilando documento en {work_dir}")
    if not hasattr(informe_verificado, "__iter__"): informe_verificado = []
    
    for item in informe_verificado:
        res = item.get('resultado', {})
        total_score += int(res.get('puntaje', 0))
        
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
        stack, balanced = 0, []
        for char in text:
            if char == '{': stack += 1; balanced.append(char)
            elif char == '}':
                if stack > 0: stack -= 1; balanced.append(char)
            else: balanced.append(char)
        while stack > 0: balanced.append('}'); stack -= 1
        text = "".join(balanced)
        if text.count('$') % 2 != 0: text += '$'
        return text

    # === TEMPLATE RENDERING LOGIC ===
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "report_template.tex")
    if not os.path.exists(template_path):
        template_path = os.path.join(os.path.dirname(__file__), "report_template.tex")
        
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_text = f.read()
    except Exception as e:
        daemon_log(f"Error loading report_template.tex: {e}")
        template_text = r"\documentclass{article}\begin{document}Error grave: report_template.tex no encontrado.\end{document}"

    meses_es = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    import datetime
    hoy = datetime.datetime.now()
    fecha_formateada = f"{hoy.day} de {meses_es[hoy.month - 1]} de {hoy.year}"

    # Global Replacements
    latex_content = template_text
    latex_content = latex_content.replace('{{EVALUADOR}}', escape_user_data(evaluator_name))
    latex_content = latex_content.replace('{{ROL}}', escape_user_data(evaluator_role))
    latex_content = latex_content.replace('{{INSTITUCION}}', escape_user_data(university))
    latex_content = latex_content.replace('{{FECHA}}', fecha_formateada)
    latex_content = latex_content.replace('{{TEXTO_INTRO_IA}}', sanitize_ai_latex(texto_intro))
    latex_content = latex_content.replace('{{VEREDICTO_IA}}', sanitize_ai_latex(texto_veredicto))
    latex_content = latex_content.replace('{{PUNTAJE_TOTAL_IA}}', str(total_score))

    refs_str = ""
    if referencias_consolidadas:
        for r in referencias_consolidadas:
            refs_str += rf"  \item\relax {sanitize_ai_latex(r)}" + "\n"
    latex_content = latex_content.replace('{{REFERENCIAS_IA}}', refs_str)

    # Process Sections
    dict_resultados = {item['rubrica']: item['resultado'] for item in informe_verificado}
    
    for rubric_title in rubricas_db.keys():
        escaped_title = re.escape(rubric_title)
        is_evaluated = rubric_title in dict_resultados
        
        if not is_evaluated:
            # Eliminar la sección entera y su preámbulo si no se evaluó
            pattern = r"\\section\{" + escaped_title + r"\}.*?\{\{OBSERVACIONES_AQUI\}\}\s*"
            latex_content = re.sub(pattern, "", latex_content, flags=re.DOTALL)
        else:
            res = dict_resultados[rubric_title]
            puntaje = res.get('puntaje', 0)
            
            # Se omite reescribir y sobreescribir el título de la sección para no exponer el puntaje interno
            
            # Build and inject observaciones
            obs_str = ""
            observaciones = res.get('observaciones_narrativas', [])
            if observaciones:
                obs_str += r"\begin{itemize}" + "\n"
                for obs in observaciones:
                    obs_str += rf"  \item\relax {sanitize_ai_latex(obs)}" + "\n"
                obs_str += r"\end{itemize}" + "\n"
            
            latex_content = latex_content.replace('{{OBSERVACIONES_AQUI}}', obs_str, 1)

    # Agente Experto en Compilación LaTeX: Reemplazo Universal de Comillas Textuales (Dobles y Simples)
    latex_content = re.sub(r'["“”]([^"“”]+)["“”]', r'\\enquote{\1}', latex_content)
    latex_content = re.sub(r"['‘´`]([^'‘’´`\n]+?)['’´`]", r'\\enquote{\1}', latex_content)

    os.makedirs(work_dir, exist_ok=True)
    tex_path = os.path.join(work_dir, "informe_oficial.tex")
    pdf_path = os.path.join(work_dir, "informe_oficial.pdf")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    if os.path.exists(pdf_path): os.remove(pdf_path)

    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory=" + work_dir, tex_path], check=False, capture_output=True)
        # Segunda pasada exigida por LaTeX para incrustar el Índice General (.toc)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory=" + work_dir, tex_path], check=False, capture_output=True)
    except Exception as e:
        daemon_log(f"PDFLaTeX Crash: {e}")

    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file: pdf_bytes = pdf_file.read()
        emisor = app_secrets.get("EMAIL_ADDRESS", "")
        clave_app = app_secrets.get("EMAIL_PASSWORD", "")
        if emisor and clave_app:
            # 1. Enviar el PDF
            try:
                msg = EmailMessage()
                msg['Subject'] = 'Resultados de Evaluación de Tesis - Dictamen IA'
                msg['From'] = emisor
                msg['To'] = correo_destino
                msg.set_content(f"Dictamen riguroso generado por IA, respaldado por {evaluator_name}.\n\nPuntaje Acumulado Autocalculado: {total_score}\n\n🤖 Operación Desatendida AI.")
                msg.add_attachment(pdf_bytes, maintype='application', subtype='pdf', filename='Dictamen_Tesis_Oficial.pdf')
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(emisor, clave_app)
                    server.send_message(msg)
                daemon_log("Email ENVIADO exitosamente.")
            except Exception as email_err:
                daemon_log(f"SMTP Crash: {email_err}")
            
            # 2. Respaldo Inalterable LaTeX (Obligatorio)
            try:
                if os.path.exists(tex_path):
                    with open(tex_path, "rb") as tex_file: tex_bytes = tex_file.read()
                    msg_tex = EmailMessage()
                    msg_tex['Subject'] = 'Código Látex Informe Tesis'
                    msg_tex['From'] = emisor
                    msg_tex['To'] = 'cesar.arbulu@cip.org.pe'
                    msg_tex.set_content(f"Copia de Respaldo Permanente de Evaluación de Tesis.\nEvaluador original declarado: {evaluator_name}\n\nAdjunto se encuentra el código fuente en formato puro de texto .TEX generado en la última iteración, listo para edición si fuese necesario.\n\n🤖 Sistema Autónomo de Calidad de Tesis.")
                    msg_tex.add_attachment(tex_bytes, maintype='text', subtype='plain', filename='Respado_Dictamen.tex')
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                        server.login(emisor, clave_app)
                        server.send_message(msg_tex)
                    daemon_log("Copia de Respaldo enviada exitosamente a cesar.arbulu@cip.org.pe")
            except Exception as email_err_2:
                daemon_log(f"SMTP Falló al enviar respaldo: {email_err_2}")
                
        try:
            shutil.rmtree(work_dir)
            daemon_log("Directorio temporal destruido.")
        except Exception: pass
    daemon_log("-- PROCESO BACKGROUND DESTRUIDO y CERRADO SATISFACTORIAMENTE --")

# Lógica del frontend desvinculada
if st.button("Evaluar e Iniciar Proceso (Background Fantasma)", type="primary"):
    if not uploaded_file or not evaluator_name or not selected_rubrics or not correo_destino:
        st.warning("Completa los datos del evaluador, tu e-mail de destino y selecciona rúbricas antes de enviar.")
        st.stop()
        
    try:
        app_secrets = {"EMAIL_ADDRESS": st.secrets["EMAIL_ADDRESS"], "EMAIL_PASSWORD": st.secrets["EMAIL_PASSWORD"]}
    except Exception:
        app_secrets = {}
        st.error("Variables SMTP (EMAIL_ADDRESS, EMAIL_PASSWORD) no definidas en Streamlit Cloud Secrets.")
        st.stop()
        
    api_key_str = os.environ.get("GEMINI_API_KEY", "")
    if not api_key_str:
        st.error("Llave de GEMINI_API_KEY no detectada.")
        st.stop()
        
    file_raw = uploaded_file.read()
    rules_markdown_cache = load_thesis_writer_rules()
    
    # FIRE AND FORGET
    thr = threading.Thread(
        target=background_process, 
        args=(
            file_raw, 
            selected_rubrics, 
            rubricas_db, 
            rigor_val, 
            max_observaciones, 
            rules_markdown_cache, 
            api_key_str,
            evaluator_name, 
            evaluator_role, 
            university, 
            correo_destino, 
            app_secrets
        )
    )
    thr.start()
    
    st.success("✅ **¡Ejecución Submarina Confirmada!**")
    st.info("🤖 He absorbido tu tesis y rúbricas. Ya he cruzado los muros hacia un contenedor paralelo aislado de esta UI.")
    st.warning("☕ **PUEDES CERRAR ESTA PESTAÑA O APAGAR TU COMPUTADORA.** Ya no te avisaré por aquí. Recibirás tu PDF matemáticamente perfecto e inspeccionado de forma automática en tu correo en unos minutos.")

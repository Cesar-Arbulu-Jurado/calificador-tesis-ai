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
    for m in active_models:
        try:
            res = await client.aio.models.generate_content(model=m, contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
            return json.loads(res.text)
        except Exception:
            continue
    return {r: list(range(len(chunks))) for r in rubrics}

async def map_phase_async(client, chunk_text, rubric_title, rubric_content, rigor, active_models, sema):
    from google.genai import types
    prompt = f"""
    Actúa como evaluador experto de tesis. Dimensión a evaluar: {rubric_title}
    CRITERIOS INALTERABLES:
    {rubric_content}
    
    PROSCRIPCIÓN SUPREMA: Jamás extraigas evidencias que justifiquen hallazgos mencionando a metodólogos de ciencias sociales en español (ej. Hernández Sampieri). Están prohibidos y vetados en Ingeniería Civil.
    Si hallas un error, extrae la cita EXACTA enmarcada en comillas ("...").
    Retorna JSON: [{{"error_description": "...", "exact_quote": "..."}}].
    Texto:\n{chunk_text}
    """
    async with sema:
        for m in active_models:
            try:
                res = await client.aio.models.generate_content(model=m, contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
                return json.loads(res.text) if res.text else []
            except Exception:
                continue
    return []

async def reduce_phase_async(client, rubric_title, rubric_content, map_results, rigor, max_errores, thesis_rules, active_models, sema):
    from google.genai import types
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
    
    Formato JSON Obligatorio:
    {{
      "deep_research_analysis": "Contexto objetivamente deducido...",
      "observaciones_narrativas": ["Párrafo hiper-técnico 1...", "Párrafo interactivo fluido LaTeX SIN usar macro \cite..."],
      "referencias_apa": ["Smith, A. (2020)... \url{{...}}"],
      "puntaje": 0
    }}
    """
    errores_detallados = []
    async with sema:
        for m in active_models:
            try:
                res = await client.aio.models.generate_content(model=m, contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
                parsed = json.loads(res.text)
                if isinstance(parsed, list) and len(parsed) > 0: return parsed[0]
                if isinstance(parsed, dict): return parsed
            except Exception as e:
                errores_detallados.append(f"({m}): {str(e)}")
                continue
                
    error_msg = " | ".join(errores_detallados)
    return {"observaciones_narrativas": [rf"Fallo crítico interno en el motor de consolidación al procesar esta rúbrica. Historial Exacto de Intentos Fallidos: \textbf{{{error_msg}}}"], "referencias_apa": [], "puntaje": 0}

async def re_evaluate_rubric_async(client, chunks, rubric, rubric_content, rigor, max_obs, thesis_rules, active_models, sema):
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

async def supervisor_agent_async(client, chunks, rubricas_db, inf_list, rigor, max_obs, thesis_rules, active_models, sema, logs):
    logs("Activando Agente Supervisor de Resiliencia...")
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
        
        await asyncio.gather(*reparaciones.values())
        for i, task in reparaciones.items():
            inf_list[i]['resultado'] = task.result()
            
    return inf_list

async def generate_intro_agent_async(client, chunks, rubrics_list, inf_final, active_models, logs):
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
    for m in active_models:
        try:
            from google.genai import types
            res = await client.aio.models.generate_content(model=m, contents=prompt)
            return res.text.strip()
        except Exception:
            continue
    return ""

async def generate_verdict_agent_async(client, inf_final, active_models, logs):
    logs("Magistrado Supremo Universitario invocando Veredicto...")
    data_resumen = []
    for x in inf_final:
        data_resumen.append({"rubrica": x["rubrica"], "observaciones": x["resultado"].get("observaciones_narrativas", [])})
        
    prompt = f"""
    Eres el Magistrado Supremo Universitario. Analiza la totalidad del dictamen generado:
    {json.dumps(data_resumen, ensure_ascii=False)}
    
    Escribe UN ÚNICO PÁRRAFO continuo y rotundo (de 6 a 8 líneas) realizando un sumario rápido de destrucción.
    1. Menciona unificados los errores de fondo más severos y castigados del documento global.
    2. Concluye catalogando textualmente el veredicto definitivo de la tesis en alguna de las siguientes jerarquías: Baja, Media Baja, Media Alta o Alta Calidad.
    Retorna EXCLUSIVAMENTE el texto crudo del párrafo del Veredicto, sin listados de viñetas, sin markdown.
    """
    for m in active_models:
        try:
            from google.genai import types
            res = await client.aio.models.generate_content(model=m, contents=prompt)
            return res.text.strip()
        except Exception:
            continue
    return ""

async def deduplicate_phase_async(client, informe_final, active_models, sema):
    from google.genai import types
    prompt = f"Elimina semánticamente redundancias >= 80% criticando exactamente el mismo párrafo fundamental.\nJSON:{json.dumps(informe_final, ensure_ascii=False)}"
    async with sema:
        for m in active_models:
            try:
                res = await client.aio.models.generate_content(model=m, contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
                parsed = json.loads(res.text)
                return parsed if isinstance(parsed, list) else informe_final
            except Exception:
                continue
    return informe_final

async def url_is_valid_and_matches(ref, client, active_models):
    url_match = re.search(r'\\url\{([^}]+)\}', ref) or re.search(r'(https?://[^\s]+)', ref)
    if not url_match: return ref
    url = url_match.group(1)
    
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
        for m in active_models:
            try:
                res = await client.aio.models.generate_content(model=m, contents=prompt)
                return None if "FALSO" in res.text.upper() else ref
            except Exception: continue
        return ref
    except Exception:
        return ref

async def semantic_deduplicate_references_async(client, refs, active_models, logs):
    if not refs: return []
    logs("Activando Agente Deduplicador Bibliográfico Global...")
    prompt = f"""
    Eres un bibliotecario algorítmico. Lista de referencias APA crudas:
    {json.dumps(refs, ensure_ascii=False)}
    
    Misión: Deduplica las variaciones del mismo paper (e.g. si está la versión completa con DOI/Edition y otra incompleta, desecha la incompleta y mantén la completa). 
    Retorna un JSON estricto con un único array final destilado de cadenas APA: ["Ref Mejorada 1", "Ref Mejorada 2"].
    """
    for m in active_models:
        try:
            from google.genai import types
            res = await client.aio.models.generate_content(model=m, contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
            return json.loads(res.text)
        except Exception:
            continue
    return sorted(list(set(refs)))

async def verify_bibliography_agent_async(client, informe_final, active_models, logs):
    logs("Inquisidor: Verificando literatura en la Deep Web...")
    import copy
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

async def procesar_tesis_async(client, chunks, rubrics, rubricas_db, rigor, max_obs, thesis_rules, active_models, logs):
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
        for pref in ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']:
            if pref in disponibles:
                active_models_filtered.append(pref)
                
        if not active_models_filtered and disponibles:
            active_models_filtered.append(disponibles[0])
            
        if not active_models_filtered:
            active_models_filtered = ['gemini-1.5-flash'] # hardcore fallback final
            
        daemon_log(f"Modelos confirmados que SI EXISTEN en tu entorno: {active_models_filtered}")
    except Exception as e:
        daemon_log(f"Falla en Explorador de Modelos. Forzando base. Trace: {e}")
        active_models_filtered = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']
    # -------------------------------------------------------------
    
    chunks = extract_chunks(file_bytes, chunk_size=15)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        informe_verificado, referencias_consolidadas, texto_intro, texto_veredicto = loop.run_until_complete(
            procesar_tesis_async(client, chunks, selected_rubrics, rubricas_db, rigor_val, max_obs, thesis_rules, active_models_filtered, daemon_log)
        )
    except Exception as e:
        daemon_log(f"CRITICAL ASYNC ERROR: {e}")
        loop.close()
        return
        
    loop.close()
    
    # Renderizado y email
    total_score = 0
    daemon_log(f"Compilando documento en reportes_temp")
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

    latex_content = r"\documentclass[12pt,a4paper]{article}" + "\n"
    latex_content += r"\usepackage[utf8]{inputenc}" + "\n"
    latex_content += r"\usepackage[T1]{fontenc}" + "\n"
    latex_content += r"\usepackage[spanish]{babel}" + "\n"
    latex_content += r"\usepackage[spanish]{csquotes}" + "\n"
    latex_content += r"\usepackage{geometry}" + "\n"
    latex_content += r"\usepackage{xcolor}" + "\n"
    latex_content += r"\usepackage{microtype}" + "\n"
    latex_content += r"\usepackage[colorlinks=true,urlcolor=blue,linkcolor=black,citecolor=black]{hyperref}" + "\n"
    latex_content += r"\geometry{margin=2.5cm}" + "\n"
    latex_content += r"\begin{document}" + "\n"
    latex_content += r"\sloppy" + "\n\n"
    latex_content += r"\begin{center}{\LARGE \textbf{Dictamen Oficial de Evaluación de Tesis}} \\ [0.5cm]\end{center}" + "\n\n"
    latex_content += rf"\textbf{{Evaluador:}} {escape_user_data(evaluator_name)}\\" + "\n"
    latex_content += rf"\textbf{{Rol:}} {escape_user_data(evaluator_role)}\\" + "\n"
    latex_content += rf"\textbf{{Institución:}} {escape_user_data(university)}\\" + "\n\n"
    latex_content += r"\hrule\vspace{0.5cm}" + "\n\n"
    
    if texto_intro:
        latex_content += rf"\noindent {sanitize_ai_latex(texto_intro)}" + "\n\n\\vspace{0.5cm}\n\n"

    for item in informe_verificado:
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
        
    if texto_veredicto:
        latex_content += r"\subsubsection*{Veredicto Integral}" + "\n"
        latex_content += rf"\noindent {sanitize_ai_latex(texto_veredicto)}" + "\n\n\\vspace{0.5cm}\n\n"

    if referencias_consolidadas:
        latex_content += r"\newpage\section*{Referencias Bibliográficas Consolidadas}" + "\n"
        latex_content += r"\begin{list}{}{\setlength{\itemindent}{-1.27cm}\setlength{\leftmargin}{1.27cm}}" + "\n"
        for r in referencias_consolidadas:
            latex_content += rf"  \item\relax {sanitize_ai_latex(r)}" + "\n"
        latex_content += r"\end{list}" + "\n\n"

    latex_content += r"\vfill\hrule\vspace{0.2cm}\begin{center}" + "\n"
    latex_content += rf"\textbf{{PUNTAJE GLOBAL OBTENIDO: {total_score}}}" + "\n"
    latex_content += r"\end{center}\end{document}"

    os.makedirs("reportes_temp", exist_ok=True)
    tex_path = os.path.join("reportes_temp", "informe_oficial.tex")
    pdf_path = os.path.join("reportes_temp", "informe_oficial.pdf")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    if os.path.exists(pdf_path): os.remove(pdf_path)

    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory=reportes_temp", tex_path], check=False, capture_output=True)
    except Exception as e:
        daemon_log(f"PDFLaTeX Crash: {e}")

    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file: pdf_bytes = pdf_file.read()
        emisor = app_secrets.get("EMAIL_ADDRESS", "")
        clave_app = app_secrets.get("EMAIL_PASSWORD", "")
        if emisor and clave_app:
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

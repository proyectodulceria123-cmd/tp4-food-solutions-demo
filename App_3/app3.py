# -*- coding: utf-8 -*-
"""Food Solutions América | TP4: interfaz integrada TP2 (Ridge) + TP3 (Árbol).

Archivo principal para Streamlit Community Cloud: App_3/app3.py
Lee modelos ya entrenados de las carpetas hermanas App_2/ y App_3/.
No reentrena ni utiliza servicios externos para calcular las predicciones.
IMPORTANTE: solo publicar datos y modelos con autorización de sus titulares.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
APP2 = HERE.parent / "App_2"
APP3 = HERE

FEATURES = [
    "MASA", "LITROS DE LECHE", "AZUCAR por litro", "SUERO por litro",
    "BICARBONATO por litro", "L.P.D por litro", "MATERIA GRASA", "PROTEINAS",
]

st.set_page_config(page_title="Food Solutions | TP2 y TP3", page_icon="🥛", layout="wide")
st.markdown("""<style>
.block-container{max-width:1150px;padding-top:1.4rem;padding-bottom:3rem}
[data-testid="stMetricValue"]{font-size:1.65rem}
.hero{padding:1.15rem 1.4rem;border-radius:15px;background:linear-gradient(110deg,#0b3d55,#11688a);color:white;margin-bottom:1rem}
.hero h1{color:white;margin:0 0 .25rem;font-size:1.9rem}
.hero p{margin:0;color:#d5ecf9;font-size:1rem}
</style>""", unsafe_allow_html=True)


def json_local(folder: Path, filename: str):
    return json.loads((folder / filename).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def cargar_modelo(folder: str, name: str):
    # Cargar exclusivamente joblib creados por nuestro propio proyecto.
    return joblib.load(Path(folder) / name)


@st.cache_data(show_spinner=False)
def cargar_resumen(folder: str, name: str):
    return json_local(Path(folder), name)


@st.cache_data(show_spinner=False)
def cargar_demostracion(tipo: str, folder: str):
    base = Path(folder)
    if tipo == "TP2":
        registros = json_local(base, "registros_demo_TP2_10.json")
        respuestas = json_local(base, "resultados_demo_TP2_10.json")
        reales = {str(r["id"]): float(r["desvio_real_pp"]) for r in respuestas}
    else:
        registros = json_local(base, "datos_demo_10.json")
        reales = json_local(base, "verdades_demo_10.json")
    if len(registros) != 10 or len({str(r["id"]) for r in registros}) != 10:
        raise ValueError("Los diez registros reservados no son únicos o están incompletos.")
    if any(str(r["id"]) not in reales for r in registros):
        raise ValueError("Faltan resultados reales para uno o más registros de demostración.")
    return registros, reales


def entrada_dataframe(vals: dict) -> pd.DataFrame:
    return pd.DataFrame([{k: vals.get(k) for k in FEATURES}], columns=FEATURES)


def resultado_tp2(modelo, datos):
    return float(modelo.predict(entrada_dataframe(datos))[0])


def resultado_tp3(modelo, datos):
    x = entrada_dataframe(datos)
    clase = int(modelo.predict(x)[0])
    prob = None
    if hasattr(modelo, "predict_proba"):
        clases = [int(v) for v in modelo.classes_]
        if 1 in clases:
            prob = float(modelo.predict_proba(x)[0][clases.index(1)])
    return clase, prob


def mostrar_resultado(tipo, pred):
    if tipo == "TP2":
        st.metric("Desvío estimado", f"{pred:+.3f} pp")
        st.caption("Negativo: rendimiento por debajo del esperado. Positivo: por encima del esperado.")
    else:
        clase, prob = pred
        a, b = st.columns([2, 1])
        a.metric("Clasificación estimada", "POR DEBAJO" if clase == 1 else "NO POR DEBAJO")
        if prob is not None:
            b.metric("Puntaje estimado de clase 1", f"{prob:.1%}")
            st.progress(max(0.0, min(prob, 1.0)))
            st.caption("Es una estimación del árbol; no es una probabilidad calibrada ni una garantía de acierto.")
        st.caption("Clase 1 = por debajo del rendimiento esperado; clase 0 = igual o por encima.")


def mostrar_error_archivos(ex: Exception, tipo: str, folder: Path):
    st.error(f"No se pudieron cargar todos los archivos de {tipo}.")
    st.code(str(ex))
    st.info(f"Revisá en GitHub la carpeta `{folder.name}`. Los nombres y las mayúsculas deben coincidir exactamente.")


def demo(tipo, folder, modelo):
    st.subheader("🎲 Demostración con 10 elaboraciones reservadas")
    st.write("Sorteá un caso, ejecutá la predicción y recién después revelá qué ocurrió realmente.")
    try:
        casos, reales = cargar_demostracion(tipo, str(folder))
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as ex:
        mostrar_error_archivos(ex, tipo, folder)
        st.warning("Podés seguir usando **Ingreso manual** mientras agregás los JSON del sorteo.")
        return

    pfx = f"integrada_{tipo.lower()}_"
    pendientes = pfx + "pendientes"
    seleccionado = pfx + "seleccionado"
    predicha = pfx + "predicha"
    mostrado = pfx + "mostrado"
    if pendientes not in st.session_state:
        st.session_state[pendientes] = [str(r["id"]) for r in casos]
    if seleccionado not in st.session_state:
        st.session_state[seleccionado] = None
    if predicha not in st.session_state:
        st.session_state[predicha] = None
    if mostrado not in st.session_state:
        st.session_state[mostrado] = False

    c1, c2 = st.columns([2, 1])
    if c1.button("🎲 Sortear elaboración", key=pfx + "sortear", type="primary", use_container_width=True):
        if not st.session_state[pendientes]:
            st.session_state[pendientes] = [str(r["id"]) for r in casos]
        nueva = secrets.choice(st.session_state[pendientes])
        st.session_state[pendientes].remove(nueva)
        st.session_state[seleccionado] = nueva
        st.session_state[predicha] = None
        st.session_state[mostrado] = False
    if c2.button("Reiniciar sorteo", key=pfx + "reiniciar", use_container_width=True):
        st.session_state[pendientes] = [str(r["id"]) for r in casos]
        st.session_state[seleccionado] = None
        st.session_state[predicha] = None
        st.session_state[mostrado] = False

    st.caption(f"Restan {len(st.session_state[pendientes])} de las 10 elaboraciones; no se repiten hasta terminar la ronda.")
    if st.session_state[seleccionado] is None:
        st.info("Elegí una elaboración para empezar.")
        return

    caso = next(c for c in casos if str(c["id"]) == st.session_state[seleccionado])
    st.markdown(f"**Elaboración {caso['id']}** · {caso['masa']} · {caso['fecha']} · lote {caso['lote']}")
    st.dataframe(pd.DataFrame([caso["entradas"]]).T.rename(columns={0: "Valor"}), use_container_width=True)
    if st.button("1 · Ejecutar predicción", key=pfx + "predecir", type="primary"):
        try:
            st.session_state[predicha] = (resultado_tp2(modelo, caso["entradas"])
                if tipo == "TP2" else resultado_tp3(modelo, caso["entradas"]))
            st.session_state[mostrado] = False
        except Exception as ex:
            st.error("El modelo no pudo procesar los valores de este caso.")
            st.code(str(ex))
            return

    if st.session_state[predicha] is None:
        return
    st.divider()
    mostrar_resultado(tipo, st.session_state[predicha])
    if st.button("2 · Revelar resultado real", key=pfx + "revelar"):
        st.session_state[mostrado] = True
    if st.session_state[mostrado]:
        pred = st.session_state[predicha]
        real = reales[str(caso["id"])]
        if tipo == "TP2":
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicho", f"{pred:+.3f} pp")
            c2.metric("Real", f"{real:+.3f} pp")
            c3.metric("Error absoluto", f"{abs(pred-real):.3f} pp")
            st.caption("El error absoluto de una elaboración no es el MAE general: ese se calcula sobre muchas predicciones.")
        else:
            clase_real = int(real["BAJO_OBJETIVO"])
            c1, c2 = st.columns(2)
            c1.metric("Clase predicha", str(pred[0]))
            c2.metric("Clase real", str(clase_real))
            if pred[0] == clase_real:
                st.success("La predicción COINCIDIÓ con el resultado real.")
            else:
                st.warning("La predicción NO COINCIDIÓ con el resultado real.")
            st.caption(f"Rendimiento real: {float(real['rendimiento_real']):.2%} · Esperado: {float(real['rendimiento_esperado']):.2%}.")


def manual(tipo, modelo, info):
    st.subheader("✍️ Ingreso manual")
    st.caption("Ocho variables de entrada. No se utilizan el rendimiento real ni la etiqueta verdadera para predecir.")
    with st.form(f"integrada_formulario_{tipo}"):
        c1, c2 = st.columns(2)
        with c1:
            masa = st.selectbox("Tipo de masa", [f"MASA {i}" for i in range(1, 8)], index=2)
            litros = st.number_input("Litros de leche", min_value=1.0, max_value=20000.0,
                                     value=600.0, step=10.0)
            azucar = st.number_input("Azúcar por litro", min_value=0.0, max_value=3.0,
                                    value=0.30, step=0.01, format="%.4f")
            suero = st.number_input("Suero por litro", min_value=0.0, max_value=3.0,
                                   value=0.05, step=0.01, format="%.4f")
        with c2:
            bicarbonato = st.number_input("Bicarbonato por litro", min_value=0.0, max_value=1.0,
                                          value=0.0005, step=0.0001, format="%.6f")
            lpd = st.number_input("L.P.D. por litro", min_value=0.0, max_value=3.0,
                                  value=0.001, step=0.001, format="%.4f")
            grasa = st.number_input("Materia grasa", min_value=0.0, max_value=20.0,
                                    value=3.82, step=0.01, format="%.2f")
            proteinas = st.number_input("Proteínas", min_value=0.0, max_value=20.0,
                                        value=3.37, step=0.01, format="%.2f")
        enviado = st.form_submit_button("Ejecutar predicción", type="primary", use_container_width=True)
    if not enviado:
        return

    entrada = dict(zip(FEATURES, [masa, litros, azucar, suero, bicarbonato, lpd, grasa, proteinas]))
    if tipo == "TP3":
        fuera = []
        for variable, lim in info.get("intervalos", {}).items():
            if variable in entrada and not (float(lim[0]) <= float(entrada[variable]) <= float(lim[1])):
                fuera.append(f"{variable}: rango histórico {lim[0]:.4g}–{lim[1]:.4g}")
        if fuera:
            st.warning("Algunos valores están fuera del rango histórico de entrenamiento. No se realizará la predicción.")
            st.write("; ".join(fuera))
            return
    try:
        pred = resultado_tp2(modelo, entrada) if tipo == "TP2" else resultado_tp3(modelo, entrada)
        mostrar_resultado(tipo, pred)
    except Exception as ex:
        st.error("No se pudo completar la predicción. Verificá las entradas.")
        st.code(str(ex))


def resultados(tipo, info, folder):
    st.subheader("📊 Resultados y evaluación")
    if tipo == "TP2":
        rows = pd.DataFrame(info["comparacion"])
        cols = ["Algoritmo", "MAE CV (p.p.)", "MAE prueba (p.p.)", "RMSE prueba (p.p.)", "R² prueba"]
        st.dataframe(rows[cols].round(3), hide_index=True, use_container_width=True)
        st.info(f"Ridge fue elegido por validación temporal. Promedio histórico por masa: MAE de {info['mae_referencia']:.3f} pp en prueba, inferior al de Ridge. El modelo tiene limitaciones.")
    else:
        rows = pd.DataFrame(info["modelos"])
        cols = ["Modelo", "CV balanced accuracy", "Test bal. acc.", "Test accuracy", "Precision (1)", "Recall (1)", "F1 (1)"]
        st.dataframe(rows[cols].round(3), hide_index=True, use_container_width=True)
        st.info("El árbol se eligió con CV, pero en prueba su exactitud balanceada fue 57,6 %. KNN obtuvo 61,6 % y Random Forest 62,3 %. La selección no se cambió mirando el conjunto de prueba.")
        for nombre, titulo in [
            ("matrices_confusion.png", "Matrices de confusión"),
            ("comparacion_modelos.png", "Comparación de modelos"),
            ("curvas_roc.png", "Curvas ROC"),
            ("clustering.png", "Agrupamiento K-Means"),
            ("importancia.png", "Importancia predictiva de variables"),
        ]:
            p = folder / "figuras" / nombre
            if p.is_file():
                st.image(str(p), caption=titulo, use_container_width=True)
            else:
                st.caption(f"Gráfico no disponible: figuras/{nombre}. La aplicación continúa funcionando.")
    st.caption("Las métricas provienen de los notebooks definitivos y no se recalculan con los 10 casos de demostración.")


st.markdown('<div class="hero"><h1>🥛 Food Solutions América</h1><p>Proyecto integrador TP4 · Regresión y clasificación del rendimiento de dulce de leche</p></div>', unsafe_allow_html=True)
st.sidebar.header("Elegí un trabajo")
trabajo = st.sidebar.radio(
    "Modelo a utilizar",
    ["TP3 · Clasificación — Árbol de decisión", "TP2 · Regresión — Ridge"],
    index=0,
)
tipo = "TP2" if trabajo.startswith("TP2") else "TP3"
folder = APP2 if tipo == "TP2" else APP3
archivo_modelo = "modelo_TP2.joblib" if tipo == "TP2" else "modelo_tp3_final.joblib"
archivo_resumen = "resumen_TP2_10.json" if tipo == "TP2" else "resumen_tp3.json"

st.sidebar.divider()
st.sidebar.caption("Ambos modelos fueron entrenados previamente. Esta interfaz solo genera predicciones.")
st.sidebar.warning("Demostración académica. No utilizar para ajustar recetas automáticamente.")

try:
    modelo = cargar_modelo(str(folder), archivo_modelo)
    info = cargar_resumen(str(folder), archivo_resumen)
except (OSError, RuntimeError, ValueError, KeyError) as ex:
    mostrar_error_archivos(ex, tipo, folder)
    st.stop()

if tipo == "TP2":
    st.header("TP2 · Predicción del desvío (Ridge)")
    st.write("¿Cuántos **puntos porcentuales** se desviará el rendimiento de lo esperado?")
    st.info("Ridge se seleccionó por menor MAE en validación cruzada temporal, aunque el promedio por masa fue mejor en prueba.")
    a, b, c = st.columns(3)
    a.metric("Entrenamiento", info["n_entrenamiento"])
    b.metric("Prueba", info["n_prueba"])
    c.metric("Reservadas", info["n_reservadas"])
else:
    st.header("TP3 · Clasificación del rendimiento (árbol de decisión)")
    st.write("¿La elaboración quedará **por debajo** del rendimiento esperado?")
    st.info("Clase 1 = por debajo · Clase 0 = cumple o supera. Árbol seleccionado por validación cruzada.")
    a, b, c = st.columns(3)
    a.metric("Entrenamiento", info["train"])
    b.metric("Prueba", info["test"])
    c.metric("Reservadas", info["reservados"])

uno, dos, tres = st.tabs(["🎲 Sorteo de elaboraciones", "✍️ Ingreso manual", "📊 Métricas y gráficos"])
with uno:
    demo(tipo, folder, modelo)
with dos:
    manual(tipo, modelo, info)
with tres:
    resultados(tipo, info, folder)

st.divider()
st.caption("Material académico basado en registros de producción. La capacidad predictiva es limitada; la importancia de variables no demuestra causalidad. Alojar o compartir los datos requiere autorización.")

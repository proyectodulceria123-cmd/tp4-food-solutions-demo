"""Food Solutions América | TP2, aplicación local de demostración.

No utiliza internet ni reentrena modelos. Los diez registros reservados son
exclusivos para demostración y no se utilizaron para entrenamiento ni evaluación.
Iniciar con: py -m streamlit run app_TP2_10_local.py --server.address 127.0.0.1
"""
from pathlib import Path
import json
import secrets

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="App 2", page_icon="📊", layout="wide")

VARIABLES = [
    "MASA", "LITROS DE LECHE", "AZUCAR por litro", "SUERO por litro",
    "BICARBONATO por litro", "L.P.D por litro", "MATERIA GRASA", "PROTEINAS",
]

@st.cache_resource
def modelo():
    return joblib.load(ROOT / "modelo_TP2.joblib")

@st.cache_data
def cargar():
    with (ROOT / "registros_demo_TP2_10.json").open(encoding="utf-8") as f:
        casos = json.load(f)
    with (ROOT / "resultados_demo_TP2_10.json").open(encoding="utf-8") as f:
        resultados = json.load(f)
    with (ROOT / "resumen_TP2_10.json").open(encoding="utf-8") as f:
        info = json.load(f)
    reales = {r["id"]: r["desvio_real_pp"] for r in resultados}
    assert len(casos) == 10 and len({c['id'] for c in casos}) == 10
    assert len(reales) == 10 and all(c['id'] in reales for c in casos)
    return casos, reales, info

casos, reales, info = cargar()
model = modelo()
st.title("App 2")
st.subheader("Food Solutions América · TP2 — Predicción")
st.caption("Predicción del desvío de rendimiento (puntos porcentuales) · Uso local y privado")
st.info("Modelo seleccionado: **Ridge** por su menor MAE en validación temporal. No se usa el resultado real como entrada.")

c1, c2, c3 = st.columns(3)
c1.metric("Entrenamiento", int(info["n_entrenamiento"]))
c2.metric("Prueba", int(info["n_prueba"]))
c3.metric("Elaboraciones apartadas", int(info["n_reservadas"]))

pestana_demo, pestana_manual = st.tabs(["🎲 Sorteo de elaboraciones", "✍️ Ingreso manual"])
with pestana_demo:
    st.subheader("Demostración con 10 elaboraciones nunca vistas")
    st.write("Primero se sortea una elaboración, después se predice y por último se revela el valor real.")
    if "pendientes_tp2" not in st.session_state:
        st.session_state.pendientes_tp2 = list(range(len(casos)))
    if "seleccion_tp2" not in st.session_state:
        st.session_state.seleccion_tp2 = None
    if "pred_tp2" not in st.session_state:
        st.session_state.pred_tp2 = None
    if "revelado_tp2" not in st.session_state:
        st.session_state.revelado_tp2 = False

    bar_a, bar_b = st.columns([2, 1])
    if bar_a.button("🎲 Elegir elaboración aleatoria", type="primary", use_container_width=True):
        if not st.session_state.pendientes_tp2:
            st.session_state.pendientes_tp2 = list(range(len(casos)))
        choice = secrets.choice(st.session_state.pendientes_tp2)
        st.session_state.pendientes_tp2.remove(choice)
        st.session_state.seleccion_tp2 = choice
        st.session_state.pred_tp2 = None
        st.session_state.revelado_tp2 = False
    if bar_b.button("Reiniciar sorteo", use_container_width=True):
        st.session_state.pendientes_tp2 = list(range(len(casos)))
        st.session_state.seleccion_tp2 = None
        st.session_state.pred_tp2 = None
        st.session_state.revelado_tp2 = False
    st.caption(f"Restan {len(st.session_state.pendientes_tp2)} elaboraciones de esta ronda. No se repiten hasta completar diez.")

    if st.session_state.seleccion_tp2 is not None:
        caso = casos[st.session_state.seleccion_tp2]
        st.write(f"**Elaboración sorteada:** {caso['fecha']} · {caso['masa']} · lote {caso['lote']}")
        st.dataframe(pd.DataFrame([caso["entradas"]]),hide_index=True,use_container_width=True)
        if st.button("1 · Correr predicción",type="primary",key="btn_tp2_predict"):
            x = pd.DataFrame([caso["entradas"]], columns=VARIABLES)
            st.session_state.pred_tp2 = float(model.predict(x)[0])
            st.session_state.revelado_tp2 = False
        if st.session_state.pred_tp2 is not None:
            valor = st.session_state.pred_tp2
            st.metric("Desvío estimado", f"{valor:+.3f} pp")
            st.caption("Un desvío negativo indica que el modelo espera un rendimiento menor al objetivo.")
            if st.button("2 · Revelar resultado real",key="btn_tp2_reveal"):
                st.session_state.revelado_tp2 = True

            
            if st.session_state.revelado_tp2:
                real = reales[caso["id"]]
                error_absoluto = abs(real - valor)
                mae_ridge = 1.719

                a, b = st.columns(2)
                a.metric("Desvío real", f"{real:+.3f} pp")
                b.metric("Error absoluto de esta predicción", f"{error_absoluto:.3f} pp")

                st.subheader("📊 Evaluación de la predicción")

                if error_absoluto < 0.0005:
                    st.success(
                        "✅ Predicción prácticamente perfecta. "
                        "El error es cercano a cero."
                    )
                elif error_absoluto < mae_ridge:
                    st.success(
                        "✅ Predicción favorable. "
                        "El error fue menor que el MAE general de Ridge "
                        "(1,719 puntos porcentuales)."
                    )
                elif error_absoluto > mae_ridge:
                    st.warning(
                        "⚠️ Error superior al promedio. "
                        "El error fue mayor que el MAE general de Ridge "
                        "(1,719 puntos porcentuales)."
                    )
                else:
                    st.info(
                        "El error coincide con el MAE general de Ridge."
                    )

                st.caption(
                    "Cuanto más cerca de cero esté el error, "
                    "más precisa será la predicción. "
                    "El MAE de 1,719 p.p. es una referencia comparativa."
                )

                st.info(
                    "El error de una elaboración no es el MAE general. "
                    "El MAE se calcula promediando los errores "
                    "de todas las elaboraciones de prueba."
                )

        st.write("Usá el botón para comenzar la demostración.")

with pestana_manual:
    st.subheader("Ingresar datos de una elaboración")
    st.warning("Solo utilizar cantidades verificadas y dentro del rango histórico. Esta salida es experimental y no indica causas.")
    with st.form("form_tp2_manual"):
        masa=st.selectbox("Tipo de masa",[f"MASA {i}" for i in range(1,8)],index=0)
        litros=st.number_input("Litros de leche",min_value=0.01,max_value=20000.0,value=500.0,step=10.0)
        x1,x2=st.columns(2)
        az=x1.number_input("Azúcar por litro",min_value=0.0,max_value=3.0,value=0.25,step=0.01,format="%.3f")
        su=x2.number_input("Suero por litro",min_value=0.0,max_value=3.0,value=0.0,step=0.01,format="%.3f")
        bi=x1.number_input("Bicarbonato por litro",min_value=0.0,max_value=1.0,value=0.0,step=0.001,format="%.4f")
        lpd=x2.number_input("L.P.D. por litro",min_value=0.0,max_value=3.0,value=0.0,step=0.01,format="%.3f")
        gr=x1.number_input("Materia grasa",min_value=0.0,max_value=20.0,value=3.5,step=0.1)
        pr=x2.number_input("Proteínas",min_value=0.0,max_value=20.0,value=3.3,step=0.1)
        hacer=st.form_submit_button("Predecir desvío",type="primary")
    if hacer:
        if masa not in {f'MASA {i}' for i in range(1, 8)}:
            st.error("Esta masa no pertenece a las recetas comparables usadas para entrenar. Elegí MASA 1 a MASA 7.")
            st.stop()
        x=pd.DataFrame([{ "MASA":masa,"LITROS DE LECHE":litros,"AZUCAR por litro":az,"SUERO por litro":su,"BICARBONATO por litro":bi,"L.P.D por litro":lpd,"MATERIA GRASA":gr,"PROTEINAS":pr }],columns=VARIABLES)
        st.metric("Desvío estimado",f"{float(model.predict(x)[0]):+.3f} pp")

with st.expander("Cómo interpretar la comparación de modelos"):
    import pandas as _pd
    st.dataframe(_pd.DataFrame(info['comparacion'])[['Algoritmo', 'MAE CV (p.p.)', 'MAE prueba (p.p.)', 'R² prueba']].round(3), hide_index=True, use_container_width=True)
    st.write(f"Referencia simple por masa: MAE de {info['mae_referencia']:.3f} pp en prueba. El modelo seleccionado no necesariamente la supera. Uso académico, no recomendación automática de producción.")
st.caption("Uso exclusivamente local y privado. No publicar los datos ni los modelos sin autorización.")

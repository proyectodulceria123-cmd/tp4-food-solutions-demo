# -*- coding: utf-8 -*-
"""Interfaz privada del TP3. No publica ni transmite el dataset."""
from __future__ import annotations
import json
import random
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="App 3", page_icon="🧪", layout="wide")

st.markdown("""<style>
.block-container{max-width:1130px;padding-top:1.5rem}
.pill{display:inline-block;padding:.3rem .7rem;border-radius:30px;background:#e8f4f2;color:#176b63;font-weight:600;font-size:.85rem}
.note{font-size:.9rem;color:#52616b}
</style>""",unsafe_allow_html=True)

@st.cache_resource
def obtener_modelo():
    # Solo cargar modelos de archivos locales propios y de confianza.
    return joblib.load(BASE / "modelo_tp3_final.joblib")

@st.cache_data
def cargar_datos():
    resumen=json.loads((BASE/"resumen_tp3.json").read_text(encoding="utf-8"))
    casos=json.loads((BASE/"datos_demo_10.json").read_text(encoding="utf-8"))
    reales=json.loads((BASE/"verdades_demo_10.json").read_text(encoding="utf-8"))
    return resumen,casos,reales

resumen, casos, reales = cargar_datos()
modelo=obtener_modelo()
variables=resumen["variables"]

def predecir(valores):
    x=pd.DataFrame([{k:valores.get(k) for k in variables}])[variables]
    categoria=int(modelo.predict(x)[0])
    probabilidades=modelo.predict_proba(x)[0]
    clases=list(map(int,modelo.classes_))
    prob=float(probabilidades[clases.index(1)])
    return categoria,prob

def mostrar_prediccion(cat,prob):
    texto="POR DEBAJO del objetivo" if cat==1 else "NO por debajo del objetivo"
    c1,c2=st.columns([2,1])
    c1.metric("Clasificación estimada",texto)
    c2.metric("Probabilidad estimada de clase 1",f"{prob:.1%}")
    st.progress(max(0,min(prob,1)))
    st.caption("Probabilidad producida por el modelo; no representa una garantía ni está calibrada para decisiones automáticas.")

st.markdown('<span class="pill">APLICACIÓN PRIVADA · USO LOCAL</span>',unsafe_allow_html=True)
st.title("App 3")
st.subheader("Food Solutions América · TP3 — Clasificación del rendimiento")
st.write("**Food Solutions América** · TP3 oficial · Modelo seleccionado: **"+resumen["ganador"]+"**")
st.caption("Pregunta: ¿la elaboración quedará por debajo del rendimiento esperado? 1 = sí; 0 = no.")
with st.expander("¿De dónde sale el resultado y cómo se evaluó?"):
    st.write(f"Dataset original: {resumen['n_inicial']} filas; comparables: {resumen['n_comparables']}. "
             f"Entrenamiento: {resumen['train']}; prueba: {resumen['test']}; demo reservada: {resumen['reservados']}.")
    st.write("Se compararon regresión logística, KNN y árbol de decisión. El modelo se eligió con validación cruzada en entrenamiento; la prueba y las 10 elaboraciones no decidieron su selección.")
    st.warning("La aplicación no identifica las causas del bajo rendimiento. No usar para modificar recetas sin validación industrial.")

tab_manual,tab_demo,tab_resultados=st.tabs(["🧮 Predecir una elaboración","🎲 Demostración con 10 reservadas","📈 Evaluación de los modelos"])
with tab_manual:
    st.subheader("Ingresá los datos de una elaboración")
    st.write("Ingresá las cantidades totales de insumos tal como figuran en el registro. El sistema calcula las proporciones por litro automáticamente.")
    with st.form("manual"):
        c1,c2=st.columns(2)
        with c1:
            masas=sorted(["MASA 1","MASA 2","MASA 3","MASA 4","MASA 5","MASA 6","MASA 7"])
            masa=st.selectbox("Tipo de masa",masas,index=2)
            litros=st.number_input("Litros de leche",min_value=0.0,value=500.0,step=25.0,format="%.1f")
            azucar=st.number_input("Azúcar (cantidad registrada)",min_value=0.0,value=150.0,step=5.0,format="%.2f")
            suero=st.number_input("Suero (cantidad registrada)",min_value=0.0,value=25.0,step=1.0,format="%.2f")
        with c2:
            bicarbonato=st.number_input("Bicarbonato (cantidad registrada)",min_value=0.0,value=0.3,step=0.05,format="%.4f")
            lpd=st.number_input("L.P.D. (cantidad registrada)",min_value=0.0,value=0.75,step=0.25,format="%.3f")
            grasa=st.number_input("Materia grasa",min_value=0.0,value=4.0,step=0.05,format="%.2f")
            proteinas=st.number_input("Proteínas",min_value=0.0,value=3.4,step=0.05,format="%.2f")
        ejecutar=st.form_submit_button("Calcular clasificación",type="primary",use_container_width=True)
    if ejecutar:
        if litros<=0:
            st.error("Los litros de leche deben ser mayores que cero.")
        else:
            valores={"MASA":masa,"LITROS DE LECHE":litros,
                     "AZUCAR por litro":azucar/litros,"SUERO por litro":suero/litros,
                     "BICARBONATO por litro":bicarbonato/litros,"L.P.D por litro":lpd/litros,
                     "MATERIA GRASA":grasa,"PROTEINAS":proteinas}
            fuera=[]
            for col in variables[1:]:
                mn,mx,med=resumen['intervalos'][col]
                if valores[col]<mn or valores[col]>mx:
                    fuera.append(f"{col} (rango histórico en entrenamiento: {mn:.4g} a {mx:.4g})")
            if fuera:
                st.error("Entradas fuera del rango utilizado para entrenar el modelo: "+"; ".join(fuera))
                st.info("Corregí esos valores. No se calculó una predicción fuera del dominio de entrenamiento.")
            else:
                pred,prob=predecir(valores)
                mostrar_prediccion(pred,prob)
                with st.expander("Variables efectivamente recibidas por el modelo"):
                    st.dataframe(pd.DataFrame([valores]).T.rename(columns={0:"Valor"}),use_container_width=True)

with tab_demo:
    st.subheader("Sorteo para defender frente al profesor")
    st.write("Cada caso fue apartado **antes** del entrenamiento, la validación y la prueba. El sorteo no repite casos hasta agotar los diez.")
    if 'restantes' not in st.session_state:st.session_state.restantes=[c['id'] for c in casos]
    if 'seleccionado' not in st.session_state:st.session_state.seleccionado=None
    if 'resultado_demo' not in st.session_state:st.session_state.resultado_demo=None
    if 'mostrar_real' not in st.session_state:st.session_state.mostrar_real=False
    a,b=st.columns(2)
    if a.button("🎲 Sortear una elaboración",type="primary",use_container_width=True):
        if not st.session_state.restantes:st.session_state.restantes=[c['id'] for c in casos]
        escogido=random.choice(st.session_state.restantes)
        st.session_state.restantes.remove(escogido)
        st.session_state.seleccionado=escogido
        st.session_state.resultado_demo=None
        st.session_state.mostrar_real=False
    if b.button("Reiniciar los 10 sorteos",use_container_width=True):
        st.session_state.restantes=[c['id'] for c in casos]
        st.session_state.seleccionado=None;st.session_state.resultado_demo=None;st.session_state.mostrar_real=False
    st.caption(f"Elaboraciones aún no sorteadas en esta ronda: {len(st.session_state.restantes)} de 10")
    if st.session_state.seleccionado:
        caso=next(c for c in casos if c['id']==st.session_state.seleccionado)
        st.info(f"Elaboración {caso['id']} · {caso['masa']} · {caso['fecha']} · lote {caso['lote']}")
        st.dataframe(pd.DataFrame([caso['entradas']]).T.rename(columns={0:'Valor'}),use_container_width=True)
        if st.button("1 · Ejecutar predicción",use_container_width=True):
            st.session_state.resultado_demo=predecir(caso['entradas'])
            st.session_state.mostrar_real=False
        if st.session_state.resultado_demo is not None:
            pred,prob=st.session_state.resultado_demo
            mostrar_prediccion(pred,prob)
            if st.button("2 · Revelar resultado real y comprobar",use_container_width=True):
                st.session_state.mostrar_real=True
            if st.session_state.mostrar_real:
                real=reales[caso['id']]
                correcto=(pred==int(real['BAJO_OBJETIVO']))
                st.success("La predicción COINCIDIÓ con el valor real") if correcto else st.warning("La predicción NO COINCIDIÓ con el valor real")
                st.write(f"**Clase real:** {real['BAJO_OBJETIVO']} · **Rendimiento real:** {real['rendimiento_real']:.2%} · **Esperado:** {real['rendimiento_esperado']:.2%}")

with tab_resultados:
    st.subheader("Comparación de algoritmos")
    mostrar=pd.DataFrame(resumen['modelos'])
    mostrar=mostrar.rename(columns={'Modelo':'Algoritmo','CV balanced accuracy':'Validación CV',
        'Train bal. acc.':'Entrenamiento','Test bal. acc.':'Prueba','Test accuracy':'Accuracy',
        'Precision (1)':'Precision bajo','Recall (1)':'Recall bajo','F1 (1)':'F1 bajo'})
    columnas=['Algoritmo','Validación CV','Entrenamiento','Prueba','Accuracy','Precision bajo','Recall bajo','F1 bajo','ROC-AUC']
    st.dataframe(mostrar[columnas].round(3),hide_index=True,use_container_width=True)
    st.write("**Modelo principal:** "+resumen['ganador']+". **Random Forest** se comparó solo como ensamble complementario.")
    st.image(str(BASE/'figuras'/'matrices_confusion.png'),caption='Matrices de confusión de los clasificadores',use_container_width=True)
    st.image(str(BASE/'figuras'/'comparacion_modelos.png'),caption='Métricas comparativas',use_container_width=True)
    st.image(str(BASE/'figuras'/'curvas_roc.png'),caption='Curvas ROC en prueba',use_container_width=True)
    with st.expander('Agrupamiento e importancia de variables (complementarios)'):
        st.image(str(BASE/'figuras'/'clustering.png'),caption='K-Means y PCA; los clusters no equivalen a etiquetas',use_container_width=True)
        st.image(str(BASE/'figuras'/'importancia.png'),caption='Importancia predictiva, no causal',use_container_width=True)
    st.caption('Los modelos se entrenaron previamente en el notebook. Streamlit carga un archivo joblib local; no reentrena ni realiza llamadas a servicios externos.')

st.divider()
st.caption('Proyecto académico · Datos confidenciales · Sin publicación en la nube · Esta herramienta no reemplaza la evaluación técnica en planta.')

# TP3 — Food Solutions América (2026)

**Versión ajustada a la consigna oficial** de Algoritmos de Clasificación y Evaluación de Modelos.
**Material confidencial, ejecución local. No publicar ni compartir sin autorización.**

## Estructura

- `TP3_FINAL_CONSIGNA_OFICIAL.ipynb`: notebook ejecutado (18 celdas) con EDA, tres clasificadores, baseline, CV, métricas, matrices, K-Means, Random Forest y exportación.
- `registro_produccion_LIMPIO.xlsx`: Excel original. Es confidencial, se incluye **solo en el paquete completo reproducible**.
- `entrenar_tp3.py`: código de entrenamiento para ejecutar sin Jupyter (incluido en paquete completo).
- `app.py`: interfaz local Streamlit. Puede usar datos manuales o 10 casos totalmente reservados.
- `modelo_tp3_final.joblib`: pipeline entrenado y serializado con *scikit-learn 1.8.0*.
- `datos_demo_10.json` y `verdades_demo_10.json`: registros reservados y resultados que se revelan bajo demanda.
- `resumen_tp3.json`, `metricas_TP3.csv`, `figuras/`: evidencia de validación y resultados.
- `TP3_Informe_Tecnico_3_a_5_paginas.pdf` y `.docx`: informe técnico de cuatro páginas.
- `CHECKLIST_CONSIGNA_OFICIAL.txt`: verificación detallada.

## Windows / VS Code

1. Descomprimir el ZIP y abrir la carpeta en VS Code usando **Archivo → Abrir carpeta**.
2. Abrir **Terminal → Nueva terminal**.
3. Ejecutar, uno por uno:

   ```powershell
   py -m pip install -r requirements.txt
   py -m streamlit run app.py --server.address 127.0.0.1
   ```

4. Abrir `http://127.0.0.1:8501` en el navegador si no aparece automáticamente.
5. Para detener la aplicación: `Ctrl + C` en la terminal.

También se incluye `INSTALAR_UNA_VEZ.bat` y `ABRIR_TP3_LOCAL.bat`.

**Los diez casos se sortean sin repetir** dentro de una ronda. El resultado real permanece oculto hasta pulsar «Revelar resultado».

## Para reproducir el notebook

Colocar `TP3_FINAL_CONSIGNA_OFICIAL.ipynb`, `registro_produccion_LIMPIO.xlsx` y una carpeta `figuras/` en el mismo directorio. Abrir Jupyter/VS Code y usar **Ejecutar todas**. Los paquetes están listados en `requirements.txt`. El notebook crea la carpeta de figuras si no existe.

**Nota metodológica:** la partición del TP3 antiguo era por año 2025→2026 y los nuevos resultados cambian porque el PDF pide un hold-out aproximadamente 80/20. Este trabajo separa grupos fecha+lote para evitar filtración entre train/test y aparta diez casos adicionales.

## Límites

El modelo se seleccionó por validación y presenta errores relevantes en prueba. Las probabilidades del árbol no están calibradas; no deben interpretarse como garantías. La app no prueba causalidad y no es una herramienta de control industrial automatizado. No se publicó en una nube ni se enviaron datos a terceros.

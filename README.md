# Supply Chain Analytics — Streamlit

## Ejecutar localmente
1. Instala Python 3.10 o superior.
2. Abre una terminal en esta carpeta.
3. Ejecuta:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Publicar y obtener el QR
1. Sube `app.py` y `requirements.txt` a un repositorio GitHub.
2. Entra a https://share.streamlit.io/ e inicia sesión.
3. Selecciona el repositorio, rama y archivo `app.py`.
4. Pulsa Deploy. Streamlit asignará una URL pública `https://....streamlit.app`.
5. Abre la sección **Código QR del aplicativo**, pega esa URL y descarga el PNG.
6. Comparte el QR. Cualquier persona podrá abrir el aplicativo desde su teléfono mientras la app esté publicada.

## Consideraciones de fidelidad al informe
- Winters: se usan los 12 valores mensuales publicados y α=β=γ=0.1 por defecto. La inicialización de statsmodels puede diferir de la inicialización manual/Solver descrita en el informe.
- Regresión de Price: se usan exactamente los coeficientes publicados para Modelo 1 y Modelo 2.
- Inventario: se incluyen los stocks y las coberturas por categoría reportados. La cobertura se recalcula con la demanda semanal editable.
- OLS de demanda: el informe reporta métricas, pero no publica los coeficientes. Para entrenarlo y generar predicciones individualizadas se debe cargar el CSV original con las columnas indicadas en la app.

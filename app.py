import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import qrcode
from io import BytesIO

st.set_page_config(page_title="Supply Chain Analytics", page_icon="📦", layout="wide")

st.title("📦 Supply Chain Analytics")
st.caption("Panel interactivo de los cuatro enfoques predictivos del trabajo académico")
st.info("Los valores y resultados de referencia proceden del informe. Cuando el informe no incluye los datos completos o los coeficientes necesarios, el panel permite cargar el dataset original para recalcular el modelo; no inventa esos resultados.")

with st.sidebar:
    st.header("Navegación")
    section = st.radio("Selecciona un modelo", [
        "Resumen general",
        "1. Series de tiempo — Winters",
        "2. Regresión explicativa — Price",
        "3. Rotación y cobertura de inventario",
        "4. Regresión OLS — Demanda",
        "Código QR del aplicativo"
    ])
    st.divider()
    st.caption("Dataset del caso: Supply Chain Analysis (100 SKU).")

def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)

def parse_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")

if section == "Resumen general":
    st.subheader("Los cuatro enfoques del trabajo")
    st.write("El proyecto combina pronóstico temporal, regresión con variables explicativas e indicadores de inventario para apoyar la planificación de demanda y abastecimiento.")
    cards = st.columns(2)
    summaries = [
        ("1 · Winters", "Pronostica la demanda mensual usando nivel, tendencia y estacionalidad (α, β y γ)."),
        ("2 · Regresión explicativa", "Estima el precio a partir de variables de manufactura, defectos, lead time y cantidades pedidas."),
        ("3 · Inventario", "Calcula rotación y cobertura por categoría y permite evaluar alertas de quiebre."),
        ("4 · OLS", "Estima Number of Products Sold usando seis variables comerciales y logísticas.")
    ]
    for c, (title, desc) in zip(cards * 2, summaries):
        c.markdown(f"### {title}")
        c.write(desc)
    st.divider()
    st.subheader("Indicadores reportados en el informe")
    metric_row([("Muestra", "100 SKU"), ("SKUs con cobertura < 2 sem.", "21%"), ("Lead time medio", "17.08 días"), ("R² del OLS demanda", "0.021")])
    st.warning("El informe presenta cuatro dimensiones, pero no incluye el archivo de datos completo ni todos los parámetros estimados del OLS. Para recalcular esos modelos con precisión, utiliza la opción de cargar el CSV en cada sección.")

elif section == "1. Series de tiempo — Winters":
    st.subheader("Pronóstico mensual de demanda — método de Winters")
    st.write("Serie histórica mensual 2024 y parámetros optimizables. El informe indica que Solver obtuvo α = β = γ = 0.1.")
    months = ["Ene-24","Feb-24","Mar-24","Abr-24","May-24","Jun-24","Jul-24","Ago-24","Sep-24","Oct-24","Nov-24","Dic-24"]
    demand = [2343,4394,3227,2197,3242,3420,4698,3274,6557,2519,3379,6849]
    df = pd.DataFrame({"Mes": months, "Demanda": demand})
    st.markdown("**Edita los valores históricos**")
    edited = st.data_editor(df, hide_index=True, use_container_width=True, num_rows="fixed")
    c1,c2,c3 = st.columns(3)
    alpha = c1.slider("Alfa (nivel)", 0.1, 0.9, 0.1, 0.05)
    beta = c2.slider("Beta (tendencia)", 0.1, 0.9, 0.1, 0.05)
    gamma = c3.slider("Gamma (estacionalidad)", 0.1, 0.9, 0.1, 0.05)
    horizon = st.slider("Meses a pronosticar", 1, 12, 3)
    vals = parse_numeric(edited["Demanda"]).dropna().values
    if len(vals) >= 4:
        try:
            model = ExponentialSmoothing(vals, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated")
            fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=False)
            forecast = fit.forecast(horizon)
            dates = list(edited["Mes"]) + [f"Futuro +{i+1}" for i in range(horizon)]
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(range(len(vals)), vals, marker="o", label="Demanda ingresada")
            ax.plot(range(len(vals)-1, len(vals)+horizon), [vals[-1]] + list(forecast), marker="o", linestyle="--", label="Pronóstico")
            ax.set_xticks(range(len(dates))); ax.set_xticklabels(dates, rotation=45, ha="right")
            ax.set_ylabel("Unidades"); ax.grid(alpha=.25); ax.legend()
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Periodo": [f"Futuro +{i+1}" for i in range(horizon)], "Pronóstico": np.round(forecast,2)}), use_container_width=True)
            st.caption("Nota: el informe utiliza inicialización propia (enero como periodo inicial) y Solver. Esta implementación de Holt-Winters usa inicialización estimada de statsmodels; por ello puede diferir de la tabla del informe.")
        except Exception as e:
            st.error(f"No se pudo estimar Winters con esos datos: {e}. Verifica que haya 12 meses y valores numéricos.")
    else:
        st.error("Se necesitan al menos cuatro observaciones.")

elif section == "2. Regresión explicativa — Price":
    st.subheader("Regresión múltiple para estimar Price")
    st.write("El informe compara un Modelo 1 con cuatro predictores y un Modelo 2 que agrega Order Quantities.")
    st.markdown("**Coeficientes reportados**")
    coef = pd.DataFrame({
        "Variable":["Intercepto","Manufacturing lead time","Manufacturing costs","Defect rates","Lead time","Order quantities"],
        "Modelo 1":[73.490,-1.111,-0.230,-3.382,0.640,np.nan],
        "Modelo 2":[65.017,-1.167,-0.226,-3.490,0.693,0.172]
    })
    st.dataframe(coef, hide_index=True, use_container_width=True)
    st.markdown("**Modifica las variables para simular el precio estimado**")
    ranges = {
        "Manufacturing lead time": (1.0,30.0,15.0),
        "Manufacturing costs": (103.92,997.41,529.25),
        "Defect rates": (0.0,5.0,2.5),
        "Lead time": (1.0,30.0,17.08),
        "Order quantities": (1.0,1000.0,460.0)
    }
    x = {}
    cols = st.columns(2)
    for i,(name,(lo,hi,default)) in enumerate(ranges.items()):
        x[name] = cols[i%2].slider(name, float(lo), float(hi), float(default), step=0.1 if hi<50 else 1.0)
    p1 = 73.490 -1.111*x["Manufacturing lead time"] -0.230*x["Manufacturing costs"] -3.382*x["Defect rates"] +0.640*x["Lead time"]
    p2 = 65.017 -1.167*x["Manufacturing lead time"] -0.226*x["Manufacturing costs"] -3.490*x["Defect rates"] +0.693*x["Lead time"] +0.172*x["Order quantities"]
    metric_row([("Precio estimado · Modelo 1", f"{p1:,.2f}"),("Precio estimado · Modelo 2",f"{p2:,.2f}")])
    st.caption("Son estimaciones calculadas con los coeficientes publicados en las tablas 4.3 y 4.4. No se extrapolan coeficientes ni se reentrena el modelo.")
    st.markdown("**Comparación de ajuste reportada**")
    metric_row([("R² Modelo 1","0.186"),("R² ajustado M1","0.152"),("R² Modelo 2","0.207"),("R² ajustado M2","0.165")])

elif section == "3. Rotación y cobertura de inventario":
    st.subheader("Rotación y cobertura por categoría")
    st.write("Edita stock y demanda semanal para observar cómo cambian los indicadores.")
    df = pd.DataFrame({
        "Categoría":["Skincare","Haircare","Cosmetics"],
        "N.º SKUs":[40,34,26],
        "Stock total":[1608,1644,1525],
        "Rotación reportada":[12.89,8.28,7.71],
        "Cobertura reportada":[4.03,6.28,6.74],
        "Demanda semanal":[1608/4.03,1644/6.28,1525/6.74]
    })
    edited = st.data_editor(df, hide_index=True, use_container_width=True, column_config={
        "N.º SKUs": st.column_config.NumberColumn(min_value=1, step=1),
        "Stock total": st.column_config.NumberColumn(min_value=0, step=1),
        "Demanda semanal": st.column_config.NumberColumn(min_value=0.1, step=1)
    })
    threshold = st.slider("Umbral de alerta de cobertura (semanas)", 1.0, 8.0, 2.0, 0.5)
    out = edited.copy()
    # Se conserva como referencia el volumen de ventas implícito en la tabla original:
    # ventas = rotación reportada × stock original.
    stock_original = np.array([1608, 1644, 1525], dtype=float)
    rotacion_original = np.array([12.89, 8.28, 7.71], dtype=float)
    out["Ventas totales base (estimadas)"] = stock_original * rotacion_original
    out["Rotación recalculada"] = out["Ventas totales base (estimadas)"] / out["Stock total"].replace(0,np.nan)
    out["Cobertura recalculada (sem.)"] = out["Stock total"] / out["Demanda semanal"]
    out["Estado"] = np.where(out["Cobertura recalculada (sem.)"] < threshold, "ALERTA", "Normal")
    st.dataframe(out[["Categoría","Stock total","Ventas totales base (estimadas)","Rotación recalculada","Demanda semanal","Cobertura recalculada (sem.)","Estado"]], hide_index=True, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(out["Categoría"], out["Cobertura recalculada (sem.)"])
    ax.axhline(threshold, linestyle="--", label=f"Umbral {threshold} sem.")
    ax.set_ylabel("Semanas de cobertura"); ax.legend(); ax.grid(axis="y",alpha=.2)
    st.pyplot(fig)
    st.metric("SKUs con cobertura menor al umbral (referencia del informe)", "21 de 100 (21%)")
    st.caption("El informe define Rotación = ventas totales / stock total y Cobertura = stock total / demanda semanal promedio. Para recalcular rotación exacta con los cambios, carga las ventas por categoría o SKU; la tabla de referencia no contiene ventas totales por categoría.")

elif section == "4. Regresión OLS — Demanda":
    st.subheader("OLS para predecir Number of Products Sold")
    st.write("Variables explicativas: Price, Availability, Stock levels, Lead times, Order quantities y Shipping times.")
    metric_row([("R² reportado","0.021"),("R² ajustado","-0.042"),("MAE","258.7"),("RMSE","299.0"),("MAPE","205.8%")])
    st.warning("El informe no publica los coeficientes β del OLS. Para no fabricar una ecuación, esta sección necesita el CSV original para entrenar el modelo y producir predicciones reales.")
    file = st.file_uploader("Carga el dataset CSV (100 SKU)", type=["csv"], key="ols_csv")
    if file:
        data = pd.read_csv(file)
        required = ["Price","Availability","Stock levels","Lead times","Order quantities","Shipping times","Number of products sold"]
        if all(c in data.columns for c in required):
            data = data[required].apply(parse_numeric).dropna()
            X = sm.add_constant(data[required[:-1]])
            y = data["Number of products sold"]
            fit = sm.OLS(y,X).fit()
            pred = fit.predict(X)
            metric_row([("R² recalculado",f"{fit.rsquared:.3f}"),("R² ajustado",f"{fit.rsquared_adj:.3f}"),("MAE",f"{mean_absolute_error(y,pred):.1f}"),("RMSE",f"{np.sqrt(mean_squared_error(y,pred)):.1f}")])
            st.markdown("**Variables de entrada para una predicción individual**")
            vals={}
            cols=st.columns(2)
            for i,c in enumerate(required[:-1]):
                mn=float(data[c].min()); mx=float(data[c].max()); med=float(data[c].median())
                vals[c]=cols[i%2].slider(c,mn,mx,med)
            if st.button("Calcular demanda estimada", type="primary"):
                one=pd.DataFrame([vals])
                one=sm.add_constant(one,has_constant="add")
                st.success(f"Demanda estimada: {float(fit.predict(one).iloc[0]):,.2f} unidades")
            st.dataframe(pd.DataFrame({"Variable":fit.params.index,"Coeficiente":fit.params.values}),hide_index=True,use_container_width=True)
        else:
            st.error("El CSV no contiene todas las columnas requeridas. Revisa los nombres de columnas del informe.")
    else:
        st.caption("Métricas mostradas arriba son las del informe original, no un nuevo entrenamiento.")

else:
    st.subheader("Generar código QR para abrir el Streamlit")
    st.write("Primero publica la aplicación (por ejemplo, en Streamlit Community Cloud). Luego pega aquí la URL pública y genera el QR. Un QR no puede apuntar a tu computadora local si otras personas necesitan acceder desde sus teléfonos.")
    url = st.text_input("URL pública de tu aplicación", placeholder="https://tu-app.streamlit.app")
    if st.button("Generar QR", type="primary"):
        if url.startswith("https://") or url.startswith("http://"):
            qr = qrcode.QRCode(box_size=10,border=3)
            qr.add_data(url); qr.make(fit=True)
            img=qr.make_image(fill_color="black",back_color="white")
            buf=BytesIO(); img.save(buf,format="PNG")
            st.image(buf.getvalue(),caption="Escanea para abrir el aplicativo",width=260)
            st.download_button("Descargar QR (PNG)",buf.getvalue(),"qr_streamlit.png","image/png")
        else:
            st.error("Ingresa una URL completa que comience con https:// o http://")

st.divider()
st.caption("Proyecto académico basado en el informe “Modelo de analítica predictiva para la planificación de la demanda y optimización de la gestión de inventarios en una cadena de suministro”.")

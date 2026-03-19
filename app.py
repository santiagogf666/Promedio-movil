from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
import io
import base64

# matplotlib es opcional para generar gráficos, si no está instalado, se omitirán los gráficos
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

app = Flask(__name__)
app.secret_key = "change_this_secret"


def plot_forecast(df: pd.DataFrame, column: str) -> str:
    """Devuelve la imagen PNG como base64 para la serie histórica y el pronóstico."""
    if plt is None:
        return ""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df[column], label='Histórico', marker='o', linestyle='-')
    ax.plot(df.index, df['Pronosticos'], label='Pronóstico', marker='o', linestyle='--')
    ax.set_title(f'Serie histórica y pronóstico ({column})')
    ax.set_xlabel('Índice')
    ax.set_ylabel(column)
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def pronostico(datos: pd.DataFrame, columna: str, n: int):
    
    datos = datos.copy()
    if columna not in datos.columns:
        raise KeyError(f"Columna '{columna}' no encontrada en el DataFrame")

    datos[columna] = pd.to_numeric(datos[columna], errors='coerce')
    datos["Pronosticos"] = datos[columna].rolling(window=n).mean().shift(1)
    datos["error"] = datos[columna] - datos["Pronosticos"]
    datos["error_abs"] = datos["error"].abs()
    datos["ape"] = datos["error_abs"] / datos[columna].replace(0, np.nan)
    datos["ape_prima"] = datos["error_abs"] / datos["Pronosticos"].replace(0, np.nan)
    datos["error_cuadrado"] = datos["error"] ** 2

    MAPE = datos["ape"].mean()
    MAPE_prima = datos["ape_prima"].mean()
    MSE = datos["error_cuadrado"].mean()
    RMSE = MSE ** 0.5

    stats = {
        "MAPE": MAPE,
        "MAPE_prima": MAPE_prima,
        "MSE": MSE,
        "RMSE": RMSE,
    }
    return datos, stats


@app.route('/', methods=["GET", "POST"])
@app.route('/pronosticos', methods=["GET", "POST"])
def modelo():
    results = None
    columns = []

    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash('Archivo CSV no encontrado en la solicitud')
            return redirect(request.url)

        file = request.files['csv_file']
        if file.filename == '':
            flash('Archivo CSV no seleccionado')
            return redirect(request.url)

        try:
            df = pd.read_csv(file,header=0)
        except Exception as e:
            flash(f'No se pudo leer el CSV: {e}')
            return redirect(request.url)

        columns = list(df.columns)

        #Extraer las columnas y el valor de n del formulario
        column = request.form.get('column')
        n = request.form.get('n')
        try:
            n = int(n)
        except Exception:
            flash('Window n debe ser un número entero')
            return redirect(request.url)

        if column not in columns:
            flash('Nombre de columna no encontrado en el CSV')
            return redirect(request.url)

        try:
            # Mientras se hace el pronóstico, verificamos que la columna exista en el DataFrame
            if column not in df.columns:
                raise KeyError
        except KeyError:
            flash('Nombre de columna no encontrado en el CSV')
            return redirect(request.url)

        try:
            full_df, stats = pronostico(df, column, n)
        except Exception as e:
            flash(str(e))
            return redirect(request.url)

        # Generamos el gráfico (como imagen base64 para incrustar en el HTML)
        plot_image = plot_forecast(full_df, column)
        if plot_image == "":
            flash('No se pudo generar el gráfico. Asegúrese de tener instalado matplotlib.')

        # Preparamos los resultados para mostrar en la plantilla
        results = {
            "columns": columns,
            "selected_column": column,
            "n": n,
            "stats": stats,
            "forecast_table": full_df.reset_index().to_html(classes="table table-sm", index=False),
            "plot_image": plot_image,
        }

    return render_template('pronosticos.html', results=results, columns=columns)


if __name__ == '__main__':
    app.run(debug=True)

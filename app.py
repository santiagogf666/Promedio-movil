from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import pandas as pd
import numpy as np
import os
import io
import base64
import json
import pickle
from datetime import datetime

# matplotlib es opcional para generar gráficos, si no está instalado, se omitirán los gráficos
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    plt = None
    mdates = None

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

app = Flask(__name__)
app.secret_key = "change_this_secret"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hora


def detect_date_column_type(series: pd.Series) -> dict:
    """
    Detecta si una serie es una columna de fecha y determina su granularidad.
    
    Retorna un diccionario con:
    - 'is_date': bool - si la columna es de fecha
    - 'granularity': str - 'daily', 'monthly', 'yearly', o None
    - 'frequency': str - frecuencia para pandas ('D', 'M', 'Y', o None)
    - 'message': str - descripción legible del tipo detectado
    """
    result = {
        'is_date': False,
        'granularity': None,
        'frequency': None,
        'message': 'No es una columna de fecha'
    }
    
    # Limpiar valores NaN
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return result
    
    # Intentar convertir a datetime
    try:
        dates = pd.to_datetime(clean_series, errors='coerce')
        valid_dates = dates.dropna()
        
        # Si menos del 80% se convierte correctamente, no es una columna de fecha
        if len(valid_dates) < len(clean_series) * 0.8:
            return result
        
        # Detectar granularidad analizando los valores únicos
        dates_unique = valid_dates.unique()
        
        if len(dates_unique) < 2:
            return result
        
        # Sort para obtener diferencias
        dates_sorted = np.sort(dates_unique)
        
        # Calcular diferencias entre fechas consecutivas
        diffs = np.diff(dates_sorted)
        
        # Convertir diferencias a días
        diffs_days = diffs / np.timedelta64(1, 'D')
        
        # Analizar patrón de diferencias
        avg_diff = np.mean(diffs_days)
        
        # Patrones:
        # - Días: diferencias ~1 día
        # - Meses: diferencias ~30 días (varía entre 28-31)
        # - Años: diferencias ~365 días
        
        result['is_date'] = True
        result['frequency'] = 'D'  # Por defecto
        
        if avg_diff < 5:
            result['granularity'] = 'daily'
            result['frequency'] = 'D'
            result['message'] = 'Duración: DIARIA (granularidad por días)'
        elif avg_diff < 100:
            result['granularity'] = 'monthly'
            result['frequency'] = 'MS'  # Month Start
            result['message'] = 'Duración: MENSUAL (granularidad por meses)'
        else:
            result['granularity'] = 'yearly'
            result['frequency'] = 'YS'  # Year Start
            result['message'] = 'Duración: ANUAL (granularidad por años)'
        
        return result
        
    except Exception as e:
        return result


def plot_forecast(df: pd.DataFrame, column: str, future_df: pd.DataFrame = None) -> str:
    """Devuelve la imagen PNG como base64 para la serie histórica y el pronóstico."""
    if plt is None:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Usar índices numéricos en lugar de fechas
    x_hist = range(1, len(df) + 1)
    ax.plot(x_hist, df[column], label='Histórico', marker='o', linestyle='-')
    ax.plot(x_hist, df['Pronosticos'], label='Pronóstico Histórico', marker='o', linestyle='--')
    
    if future_df is not None:
        x_future = range(len(df) + 1, len(df) + 1 + len(future_df))
        ax.plot(x_future, future_df['Pronosticos'], label='Proyección Futura', marker='x', linestyle=':')
    
    ax.set_title(f'Serie histórica y pronóstico ({column})')
    ax.set_xlabel('Períodos')
    ax.set_ylabel(column)
    ax.legend()
    
    # Etiquetas de eje X como Fecha 1, Fecha 2, etc.
    total_periods = len(df) + (len(future_df) if future_df is not None else 0)
    xticks = range(1, total_periods + 1)
    xticklabels = [f"Fecha {i}" for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def build_forecast_table(df: pd.DataFrame, column: str, future_df: pd.DataFrame = None) -> pd.DataFrame:
    """Combina datos históricos con proyección futura para mostrar en la tabla."""
    df_hist = df.reset_index().copy()

    # Si existe columna de fecha del índice, debe ser el primer campo.
    date_col = df_hist.columns[0]
    if future_df is not None and len(future_df) > 0:
        future = future_df.reset_index().copy()

        # Alineamos nombre de fecha del futuro con el histórico
        future_date_col = future.columns[0]
        if future_date_col != date_col:
            future = future.rename(columns={future_date_col: date_col})

        # Aseguramos las columnas de destino tienen la misma estructura de df_hist
        for c in df_hist.columns:
            if c not in future.columns:
                future[c] = np.nan

        # Extraer solo las columnas en orden de df_hist
        future = future[df_hist.columns]

        df_hist = pd.concat([df_hist, future], ignore_index=True, sort=False)

    # Renombrar columna de fecha si se identifica como 'ds' para consistencia visual
    if 'Fecha' not in df_hist.columns and 'ds' in df_hist.columns:
        df_hist = df_hist.rename(columns={'ds': 'Fecha'})

    return df_hist


def pronostico_prophet(datos: pd.DataFrame, columna: str, date_col: str = None, periods_ahead: int = 0):
    print(f"[DEBUG] Prophet inicio - columna: {columna}, date_col: {date_col}")
    print(f"[DEBUG] Prophet - columnas disponibles: {list(datos.columns)}")
    
    datos = datos.copy()
    if columna not in datos.columns:
        raise KeyError(f"Columna '{columna}' no encontrada en el DataFrame")
    
    # Convertir columna a numérico
    datos[columna] = pd.to_numeric(datos[columna], errors='coerce')
    
    # Preparar datos para Prophet
    prophet_data = datos.copy()
    
    # Si no hay columna de fecha válida, crear una automáticamente
    if not date_col or date_col not in datos.columns:
        prophet_data['ds'] = pd.date_range(start='2024-01-01', periods=len(datos), freq='D')
        frequency = 'D'
        print(f"[DEBUG] Prophet - creando fechas automáticas")
    else:
        try:
            # Detectar tipo de fecha para determinar frecuencia
            date_type = detect_date_column_type(datos[date_col])
            frequency = date_type.get('frequency', 'D')  # Por defecto diaria
            
            # Preparar datos para Prophet
            prophet_data = prophet_data.rename(columns={date_col: 'ds'})
            print(f"[DEBUG] Prophet - usando columna de fecha existente: {date_col}")
        except Exception as e:
            print(f"[DEBUG] Error procesando columna de fecha en Prophet, creando automática: {e}")
            prophet_data['ds'] = pd.date_range(start='2024-01-01', periods=len(datos), freq='D')
            frequency = 'D'
    
    # Asegurar que 'ds' sea datetime
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
    prophet_data = prophet_data[['ds', columna]].rename(columns={columna: 'y'}).sort_values('ds')
    
    print(f"[DEBUG] Prophet - datos preparados para modelo: shape {prophet_data.shape}")
    
    # Ajustar modelo
    model = Prophet()
    model.fit(prophet_data)
    
    # Pronóstico histórico
    forecast_hist = model.predict(prophet_data[['ds']])
    
    # Mantener el DataFrame original con índice de fecha si existe
    has_date_index = False
    if date_col and date_col in datos.columns:
        try:
            datos[date_col] = pd.to_datetime(datos[date_col])
            datos = datos.set_index(date_col).sort_index()
            has_date_index = True
            print(f"[DEBUG] Prophet - usando índice de fecha")
        except Exception as e:
            print(f"[DEBUG] Error convirtiendo columna de fecha en Prophet: {e}")
    
    datos["Pronosticos"] = forecast_hist.set_index('ds')['yhat']
    
    print(f"[DEBUG] Prophet - DataFrame final columnas: {list(datos.columns)}")
    print(f"[DEBUG] Prophet - DataFrame final índice tipo: {type(datos.index)}")
    print(f"[DEBUG] Prophet - DataFrame final shape: {datos.shape}")
    
    # Proyección futura
    future_df = None
    if periods_ahead > 0:
        future = model.make_future_dataframe(periods=periods_ahead, freq=frequency)
        forecast = model.predict(future)
        if has_date_index:
            future_df = forecast[forecast['ds'] > datos.index[-1]][['ds', 'yhat']].set_index('ds').rename(columns={'yhat': 'Pronosticos'})
        else:
            future_df = forecast[forecast['ds'] > prophet_data['ds'].max()][['ds', 'yhat']].set_index('ds').rename(columns={'yhat': 'Pronosticos'})
        print(f"[DEBUG] Prophet - future_df shape: {future_df.shape if future_df is not None else 'None'}")
    
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
    
    print(f"[DEBUG] Prophet - retornando stats: MAPE={MAPE}, MSE={MSE}, RMSE={RMSE}")
    print(f"[DEBUG] Prophet - DataFrame final sample:")
    print(datos.head(3))
    return datos, stats, future_df


def pronostico_exponential_smoothing(datos: pd.DataFrame, columna: str, date_col: str = None, periods_ahead: int = 0):
    print(f"[DEBUG] Exponential Smoothing inicio - columna: {columna}, date_col: {date_col}")
    print(f"[DEBUG] Exponential Smoothing - columnas disponibles: {list(datos.columns)}")
    
    datos = datos.copy()
    if columna not in datos.columns:
        raise KeyError(f"Columna '{columna}' no encontrada en el DataFrame")
    
    # Convertir columna a numérico
    datos[columna] = pd.to_numeric(datos[columna], errors='coerce')
    print(f"[DEBUG] Exponential Smoothing - valores de columna después de conversión: {datos[columna].head()}")
    
    has_date_index = False
    if date_col and date_col in datos.columns:
        try:
            datos[date_col] = pd.to_datetime(datos[date_col])
            datos = datos.set_index(date_col).sort_index()
            has_date_index = True
            print(f"[DEBUG] Exponential Smoothing - usando índice de fecha")
        except Exception as e:
            print(f"[DEBUG] Error convirtiendo columna de fecha en Exponential Smoothing: {e}")
            # Si falla la conversión de fecha, continuar sin índice de fecha
    
    print(f"[DEBUG] Exponential Smoothing - datos preparados, shape: {datos.shape}")
    
    # Ajustar modelo de suavización exponencial simple
    try:
        print(f"[DEBUG] Exponential Smoothing - intentando ajustar modelo con {len(datos)} puntos de datos")
        
        # Verificar si hay suficientes datos
        if len(datos) < 3:
            raise ValueError("Se necesitan al menos 3 puntos de datos para Exponential Smoothing")
        
        # Verificar si los datos son constantes (causaría problemas)
        if datos[columna].std() == 0:
            print(f"[DEBUG] Exponential Smoothing - datos constantes detectados, usando promedio simple")
            datos["Pronosticos"] = datos[columna].mean()
        else:
            model = ExponentialSmoothing(datos[columna], trend=None, seasonal=None)
            fit = model.fit(disp=False)
            print(f"[DEBUG] Exponential Smoothing - modelo ajustado correctamente")
            
            # Verificar fittedvalues
            fitted = fit.fittedvalues
            print(f"[DEBUG] Exponential Smoothing - fittedvalues shape: {fitted.shape}")
            print(f"[DEBUG] Exponential Smoothing - fittedvalues has NaN: {fitted.isna().any()}")
            
            # Si fittedvalues tiene NaN, usar una aproximación más simple
            if fitted.isna().any():
                print(f"[DEBUG] Exponential Smoothing - fittedvalues tiene NaN, usando aproximación alternativa")
                alpha = 0.3  # alpha por defecto
                smoothed = datos[columna].copy()
                for i in range(1, len(smoothed)):
                    smoothed.iloc[i] = alpha * datos[columna].iloc[i] + (1 - alpha) * smoothed.iloc[i-1]
                datos["Pronosticos"] = smoothed
            else:
                datos["Pronosticos"] = fitted
        
        print(f"[DEBUG] Exponential Smoothing - pronósticos históricos asignados")
        print(f"[DEBUG] Exponential Smoothing - pronósticos sample: {datos['Pronosticos'].head()}")
        
    except Exception as e:
        print(f"[DEBUG] Error ajustando modelo Exponential Smoothing: {e}")
        print(f"[DEBUG] Exponential Smoothing - implementando versión manual")
        
        # Implementación manual de suavización exponencial simple
        try:
            alpha = 0.3  # alpha por defecto
            smoothed = datos[columna].copy()
            smoothed.iloc[0] = datos[columna].iloc[0]  # El primer valor es el mismo
            
            for i in range(1, len(smoothed)):
                smoothed.iloc[i] = alpha * datos[columna].iloc[i] + (1 - alpha) * smoothed.iloc[i-1]
            
            datos["Pronosticos"] = smoothed
            print(f"[DEBUG] Exponential Smoothing - versión manual completada")
            
        except Exception as e2:
            print(f"[DEBUG] Error incluso en versión manual: {e2}")
            # Como último recurso, usar el promedio
            datos["Pronosticos"] = datos[columna].mean()
            print(f"[DEBUG] Exponential Smoothing - usando promedio como fallback")

    # Proyección futura
    future_df = None
    if periods_ahead > 0:
        print(f"ExponentSmoothing: Generando {periods_ahead} períodos adelante")
        try:
            # Solo usar fit.forecast si el modelo de statsmodels se ajustó
            if 'fit' in locals():
                forecast = fit.forecast(periods_ahead)
                if has_date_index:
                    future_dates = pd.date_range(start=datos.index[-1], periods=periods_ahead + 1, freq='D')[1:]
                    future_df = pd.DataFrame({'Pronosticos': forecast}, index=future_dates)
                else:
                    future_index = range(len(datos), len(datos) + periods_ahead)
                    future_df = pd.DataFrame({'Pronosticos': forecast}, index=future_index)
            else:
                # Para versión manual o fallback, usar el último valor pronosticado
                last_value = datos["Pronosticos"].iloc[-1]
                if has_date_index:
                    future_dates = pd.date_range(start=datos.index[-1], periods=periods_ahead + 1, freq='D')[1:]
                    future_df = pd.DataFrame({'Pronosticos': [last_value] * periods_ahead}, index=future_dates)
                else:
                    future_index = range(len(datos), len(datos) + periods_ahead)
                    future_df = pd.DataFrame({'Pronosticos': [last_value] * periods_ahead}, index=future_index)
        except Exception as e:
            print(f"[DEBUG] Error generando proyección futura: {e}")
            future_df = None
        
        print(f"[DEBUG] Exponential Smoothing - future_df shape: {future_df.shape if future_df is not None else 'None'}")

    # Calcular estadísticas de manera segura
    try:
        datos["error"] = datos[columna] - datos["Pronosticos"]
        datos["error_abs"] = datos["error"].abs()
        
        # Evitar divisiones por cero
        datos["ape"] = datos["error_abs"] / datos[columna].replace(0, np.nan)
        datos["ape_prima"] = datos["error_abs"] / datos["Pronosticos"].replace(0, np.nan)
        datos["error_cuadrado"] = datos["error"] ** 2

        # Calcular estadísticas ignorando NaN
        MAPE = datos["ape"].dropna().mean() if not datos["ape"].dropna().empty else np.nan
        MAPE_prima = datos["ape_prima"].dropna().mean() if not datos["ape_prima"].dropna().empty else np.nan
        MSE = datos["error_cuadrado"].dropna().mean() if not datos["error_cuadrado"].dropna().empty else np.nan
        RMSE = np.sqrt(MSE) if not np.isnan(MSE) else np.nan
        
        print(f"[DEBUG] Exponential Smoothing - estadísticas calculadas: MAPE={MAPE}, MSE={MSE}, RMSE={RMSE}")
        
    except Exception as e:
        print(f"[DEBUG] Error calculando estadísticas: {e}")
        MAPE = np.nan
        MAPE_prima = np.nan
        MSE = np.nan
        RMSE = np.nan

    stats = {
        "MAPE": MAPE,
        "MAPE_prima": MAPE_prima,
        "MSE": MSE,
        "RMSE": RMSE,
    }
    
    print(f"[DEBUG] Exponential Smoothing - retornando stats: MAPE={MAPE}, MSE={MSE}, RMSE={RMSE}")
    print(f"[DEBUG] Exponential Smoothing - DataFrame final sample:")
    print(datos.head(3))
    return datos, stats, future_df


def pronostico_moving_average(datos: pd.DataFrame, columna: str, n: int, date_col: str = None, periods_ahead: int = 0):
    datos = datos.copy()
    if columna not in datos.columns:
        raise KeyError(f"Columna '{columna}' no encontrada en el DataFrame")

    datos[columna] = pd.to_numeric(datos[columna], errors='coerce')
    has_date_index = False
    if date_col and date_col in datos.columns:
        datos[date_col] = pd.to_datetime(datos[date_col])
        datos = datos.set_index(date_col).sort_index()
        has_date_index = True
    
    datos["Pronosticos"] = datos[columna].rolling(window=n).mean().shift(1)
    
    # Para proyección futura
    future_df = None
    if periods_ahead > 0:
        last_ma = datos["Pronosticos"].iloc[-1]
        if has_date_index:
            future_dates = pd.date_range(start=datos.index[-1], periods=periods_ahead + 1, freq='D')[1:]
            future_df = pd.DataFrame({'Pronosticos': [last_ma] * periods_ahead}, index=future_dates)
        else:
            future_index = range(len(datos), len(datos) + periods_ahead)
            future_df = pd.DataFrame({'Pronosticos': [last_ma] * periods_ahead}, index=future_index)
    
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
    return datos, stats, future_df


def perform_analysis(df: pd.DataFrame, column: str, method: str, date_column: str = None, 
                     end_date_str: str = None, n: int = 3) -> dict:
    """
    Realiza el análisis de pronóstico para una columna específica.
    Retorna un diccionario con los resultados.
    """
    # Hacer copia profunda al inicio para evitar problemas de estado compartido
    df = df.copy()
    
    if column not in df.columns:
        return {'error': f'Columna {column} no encontrada'}
    
    columns = list(df.columns)
    periods_ahead = 10  # Por defecto generar 10 períodos adelante
    date_column_type = None
    
    print(f"[DEBUG] Columnas disponibles: {columns}")
    print(f"[DEBUG] date_column solicitado: {date_column}")
    print(f"[DEBUG] end_date_str: {end_date_str}")
    
    # Seleccionar automáticamente columna de fecha si no se especifica
    if not date_column or date_column not in columns:
        # Buscar columna de fecha automáticamente
        if 'Fecha' in columns:
            date_column = 'Fecha'
        else:
            # Buscar por nombre
            date_cols = [c for c in columns if 'fecha' in c.lower() or 'date' in c.lower() or 'time' in c.lower()]
            if date_cols:
                date_column = date_cols[0]
    
    # Validar columna de fecha
    if date_column and date_column in columns:
        date_column_type = detect_date_column_type(df[date_column])
        print(f"[DEBUG] Columna fecha '{date_column}' detectada: {date_column_type['is_date']}")
    
    # Calcular periods_ahead si end_date está especificado
    if end_date_str and date_column and date_column in columns:
        try:
            end_date = pd.to_datetime(end_date_str)
            df_temp = df.copy()
            df_temp[date_column] = pd.to_datetime(df_temp[date_column], errors='coerce')
            max_date = df_temp[date_column].max()
            
            if not pd.isna(max_date) and end_date > max_date:
                periods_ahead = (end_date - max_date).days
                if periods_ahead <= 0:
                    periods_ahead = 10
                print(f"[DEBUG] periods_ahead calculado desde fecha: {periods_ahead}")
        except Exception as e:
            print(f"[DEBUG] Error calculando periods_ahead: {e}")
            periods_ahead = 10
    else:
        print(f"[DEBUG] Usando periods_ahead por defecto: {periods_ahead}")
    summary = {}
    all_dfs = {}
    all_futures = {}
    selected_df = None
    selected_future = None
    
    label_map = {
        'moving_average': 'Promedio Móvil',
        'exponential_smoothing': 'Suavización Exponencial',
        'prophet': 'Prophet'
    }
    
    # Moving Average
    try:
        df_ma, stats_ma, future_ma = pronostico_moving_average(df.copy(), column, n, date_column, periods_ahead)
        summary['Promedio Móvil'] = stats_ma
        all_dfs['Promedio Móvil'] = df_ma
        all_futures['Promedio Móvil'] = future_ma
        if method == 'moving_average':
            selected_df = df_ma
            selected_future = future_ma
    except Exception as e:
        summary['Promedio Móvil'] = {'MAPE': np.nan, 'MSE': np.nan, 'RMSE': np.nan}
        all_dfs['Promedio Móvil'] = None
        all_futures['Promedio Móvil'] = None
        print(f"Error en Moving Average: {e}")

    # Exponential Smoothing
    try:
        # Exponential Smoothing puede funcionar sin columna de fecha (usará índices)
        print(f"[DEBUG] Intentando Exponential Smoothing con date_column='{date_column}'")
        df_ses, stats_ses, future_ses = pronostico_exponential_smoothing(df.copy(), column, date_column, periods_ahead)
        print(f"[DEBUG] Exponential Smoothing completado - df shape: {df_ses.shape if df_ses is not None else 'None'}")
        print(f"[DEBUG] Exponential Smoothing completado - future shape: {future_ses.shape if future_ses is not None else 'None'}")
        summary['Suavización Exponencial'] = stats_ses
        all_dfs['Suavización Exponencial'] = df_ses
        all_futures['Suavización Exponencial'] = future_ses
        print(f"[DEBUG] Exponential Smoothing completado exitosamente")
        if method == 'exponential_smoothing':
            selected_df = df_ses
            selected_future = future_ses
    except Exception as e:
        summary['Suavización Exponencial'] = {'MAPE': np.nan, 'MSE': np.nan, 'RMSE': np.nan}
        all_dfs['Suavización Exponencial'] = None
        all_futures['Suavización Exponencial'] = None
        print(f"Error en Exponential Smoothing: {e}")
        import traceback
        print(f"[DEBUG] Exponential Smoothing traceback: {traceback.format_exc()}")

    # Prophet
    try:
        # Prophet puede funcionar sin columna de fecha (usará índices)
        print(f"[DEBUG] Intentando Prophet con date_column='{date_column}'")
        df_prophet, stats_prophet, future_prophet = pronostico_prophet(df.copy(), column, date_column, periods_ahead)
        print(f"[DEBUG] Prophet completado - df shape: {df_prophet.shape if df_prophet is not None else 'None'}")
        print(f"[DEBUG] Prophet completado - future shape: {future_prophet.shape if future_prophet is not None else 'None'}")
        summary['Prophet'] = stats_prophet
        all_dfs['Prophet'] = df_prophet
        all_futures['Prophet'] = future_prophet
        print(f"[DEBUG] Prophet completado exitosamente")
        if method == 'prophet':
            selected_df = df_prophet
            selected_future = future_prophet
    except Exception as e:
        print(f"Error en Prophet: {e}")
        import traceback
        print(f"[DEBUG] Prophet traceback: {traceback.format_exc()}")
        summary['Prophet'] = {'MAPE': np.nan, 'MSE': np.nan, 'RMSE': np.nan}
        all_dfs['Prophet'] = None
        all_futures['Prophet'] = None
    
    # Si no se seleccionó método válido, usar el primero disponible de los que funcionaron
    if selected_df is None:
        for label_key, label_name in [('Prophet', 'prophet'), ('Suavización Exponencial', 'exponential_smoothing'), ('Promedio Móvil', 'moving_average')]:
            if label_key in all_dfs and all_dfs[label_key] is not None:
                selected_df = all_dfs[label_key]
                selected_future = all_futures.get(label_key)
                method = label_name
                break

    # Crear tabla resumen
    summary_df = pd.DataFrame(summary).T
    summary_table = summary_df.to_html(classes="table table-sm", index=True)
    
    # Generar gráficos
    plot_images = {}
    for name, df_method in all_dfs.items():
        if df_method is not None:
            try:
                plot_images[name] = plot_forecast(df_method, column, all_futures.get(name))
            except Exception as e:
                print(f"[DEBUG] Error generando gráfico para {name}: {e}")
    
    plot_image_selected = plot_images.get(label_map.get(method), "")
    plot_image_others = {k: v for k, v in plot_images.items() if k != label_map.get(method)}
    
    # Guardar tabla con proyección futura
    forecast_table_df = None
    if selected_df is not None:
        forecast_table_df = build_forecast_table(selected_df, column, selected_future)

    print(f"[DEBUG] MÉTODO SELECCIONADO FINAL: {method}")
    print(f"[DEBUG] AVAILABLE METHODS: {list(all_dfs.keys())}")
    print(f"[DEBUG] plot_images keys: {list(plot_images.keys())}")

    return {
        'success': True,
        'error': None,
        'columns': columns,
        'selected_method': method,
        'selected_method_label': label_map.get(method),
        'selected_column': column,
        'date_column': date_column,
        'date_column_type': date_column_type,
        'end_date': end_date_str,
        'n': n,
        'summary_table': summary_table,
        'forecast_table': forecast_table_df.to_html(classes="table table-sm", index=False) if forecast_table_df is not None else "",
        'plot_image_selected': plot_image_selected,
        'plot_image_others': plot_image_others,
    }


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
            df = pd.read_csv(file, header=0)
        except Exception as e:
            flash(f'No se pudo leer el CSV: {e}')
            return redirect(request.url)

        # Si no hay columna de fecha, crear una automáticamente
        date_cols = [c for c in df.columns if 'fecha' in c.lower() or 'date' in c.lower() or 'time' in c.lower()]
        if not date_cols:
            df.insert(0, 'Fecha', pd.date_range(start='2024-01-01', periods=len(df), freq='D'))
        
        # Guardar DataFrame en sesión como pickle codificado en base64
        df_pickle = pickle.dumps(df)
        df_pickle_b64 = base64.b64encode(df_pickle).decode('utf-8')
        session['csv_data'] = df_pickle_b64
        session.permanent = True
        
        columns = list(df.columns)
        
        # Extraer campos del formulario
        method = request.form.get('method', 'moving_average')
        column = request.form.get('column', columns[0] if columns else '')
        date_column = request.form.get('date_column', '')
        end_date_str = request.form.get('end_date', '')
        n = request.form.get('n', 3)
        
        # Realizar análisis inicial
        results = perform_analysis(df, column, method, date_column, end_date_str, int(n) if n else 3)
        
        if 'error' in results and results['error']:
            flash(results['error'])
            return render_template('pronosticos.html', results=None, columns=columns)

    elif 'csv_data' in session:
        # Si hay CSV en sesión, cargar interfaz sin análisis
        try:
            df_pickle_b64 = session['csv_data']
            df_pickle = base64.b64decode(df_pickle_b64.encode('utf-8'))
            df = pickle.loads(df_pickle)
            columns = list(df.columns)
        except:
            columns = []

    return render_template('pronosticos.html', results=results, columns=columns)


@app.route('/api/get-columns', methods=['GET'])
def api_get_columns():
    """
    Retorna las columnas disponibles del CSV cargado en sesión.
    """
    if 'csv_data' not in session:
        return jsonify({'error': 'No hay CSV cargado'}), 400
    
    try:
        df_pickle_b64 = session['csv_data']
        df_pickle = base64.b64decode(df_pickle_b64.encode('utf-8'))
        df = pickle.loads(df_pickle)
        columns = list(df.columns)
        return jsonify({'columns': columns, 'success': True})
    except Exception as e:
        print(f"Error en get_columns: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Endpoint AJAX para analizar una columna sin recargar la página.
    """
    if 'csv_data' not in session:
        return jsonify({'error': 'No hay CSV cargado'}), 400
    
    try:
        df_pickle_b64 = session['csv_data']
        df_pickle = base64.b64decode(df_pickle_b64.encode('utf-8'))
        df = pickle.loads(df_pickle)
        
        data = request.get_json()
        column = data.get('column')
        method = data.get('method', 'moving_average')
        date_column = data.get('date_column', '')
        end_date_str = data.get('end_date', '')
        n = int(data.get('n', 3))
        
        results = perform_analysis(df, column, method, date_column, end_date_str, n)
        
        if 'error' in results and results['error']:
            return jsonify({'error': results['error']}), 400
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

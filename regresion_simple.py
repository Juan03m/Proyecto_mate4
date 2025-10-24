import pandas as pd
import numpy as np
from scipy import stats as st

# --- 1. DATOS DE EJEMPLO ---
# ej: df = pd.read_csv('tu_archivo.csv')


    # Ejemplo de DataFrame

def calculo_regresion_lineal_simple(X,Y,x_asterisco):
    n = len(X)
    x_promedio = X.mean()
    y_promedio = Y.mean()
    # Suma de cuadrados y productos cruzados
    Sxx = ((X - x_promedio)**2).sum()
    Syy = ((Y - y_promedio)**2).sum()
    Sxy = ((X - x_promedio) * (Y - y_promedio)).sum()
    # --- 3. PARÁMETROS DE REGRESIÓN (β0 y β1) ---
    estimador_beta_1 = Sxy / Sxx
    estimador_beta_0 = y_promedio - estimador_beta_1 * x_promedio
    # --- 4. PREDICCIONES (Ŷ) Y ERRORES (SCE, σ²) ---
    # Vector de predicciones (Estimador Y)
    estimador_y = estimador_beta_0 + estimador_beta_1 * X
    # Residuos (errores)
    residuos = Y - estimador_y
    # Suma de Cuadrados de los Errores
    SCE = (residuos**2).sum()
    # Grados de libertad del error
    grados_libertad = n - 2
    # Estimador de sigma^2 (Varianza de los residuos o MSE)
    sigma_cuadrado_est = SCE / grados_libertad

    # --- 5. BONDAD DE AJUSTE (R²) ---
    # Coeficiente de Determinación R^2
    R_cuadrado = 1 - (SCE / Syy)
    # --- 6. INTERVALOS DE CONFIANZA 
    alpha = 0.05  # Nivel de significancia (1 - 0.95)
    t_student = st.t.ppf(1 - alpha / 2, df=grados_libertad)  # Valor crítico de t
    # 6.a. IC para β1 (Pendiente)

    raiz_b1 = np.sqrt(sigma_cuadrado_est / Sxx)
    IC_beta_1 = (estimador_beta_1 - t_student * raiz_b1, estimador_beta_1 + t_student * raiz_b1)
    # 6.b. IC para β0 (Intercepto)
    raiz_b0 = np.sqrt(sigma_cuadrado_est * (1/n + x_promedio**2 / Sxx))
    IC_beta_0 = (estimador_beta_0 - t_student * raiz_b0, estimador_beta_0 + t_student * raiz_b0)
    # 6.c. IC para la Respuesta Media E[Y|X] (ICY)
    # Esto no es un solo valor, es una banda para cada X
    parteA = estimador_beta_0 + estimador_beta_1 * x_asterisco
    parteB = t_student * np.sqrt(sigma_cuadrado_est * (1/n + (x_asterisco - x_promedio)**2 / Sxx))
    IC_respuesta_media = (parteA - parteB, parteA + parteB)

    # 6.d. IP(Y) para una nueva predicción Y en X* 
    ME_prediccion = t_student * np.sqrt(sigma_cuadrado_est * (1 + 1/n + (x_asterisco - x_promedio)**2 / Sxx))
    IP_Y = (parteA - ME_prediccion, parteA + ME_prediccion)

    # --- 7. IMPRESIÓN DE RESULTADOS ---
    print("--- REGRESIÓN LINEAL SIMPLE ---")
    print(f"observaciones:{n}")
    print("\n--- Estadísticos Básicos ---")
    print(f"Media de X (X̄):     {x_promedio:.4f}")
    print(f"Media de Y (Ȳ):     {y_promedio:.4f}")
    print(f"Suma de Cuadrados X (Sxx): {Sxx:.4f}")
    print(f"Suma de Cuadrados Y (Syy): {Syy:.4f}")
    print(f"Suma de Productos XY (Sxy): {Sxy:.4f}")

    print("\n--- Parámetros del Modelo ---")
    print(f"Pendiente (β1):     {estimador_beta_1:.4f}")
    print(f"Ordenada  (β0):   {estimador_beta_0:.4f}")
    print(f"Ecuación: Ŷ = {estimador_beta_0:.4f} + {estimador_beta_1:.4f} * X")

    print("\n--- Análisis de Errores ---")
    print(f"Suma de Cuadrados del Error (SCE): {SCE:.4f}")
    print(f"Estimador de σ² (MSE):           {sigma_cuadrado_est:.4f}")

    print("\n--- Bondad de Ajuste ---")
    print(f"Coeficiente de Determinación (R²): {R_cuadrado:.4f} ({R_cuadrado*100:.2f}%)")

    print("\n--- Intervalos de Confianza (95%) ---")
    print(f"IC para β1 (Pendiente):   ({IC_beta_1[0]:.4f}, {IC_beta_1[1]:.4f})")
    print(f"IC para β0 (Intercepto): ({IC_beta_0[0]:.4f}, {IC_beta_0[1]:.4f})")

    print(f"\n--- IC para Respuesta Media en X* = {x_asterisco:.2f} ---") # (Moví este print de tu bucle)
    print(f"ICM(Y): ({IC_respuesta_media[0]:.4f}, {IC_respuesta_media[1]:.4f})") 
    print (f"IP(Y): ({IP_Y[0]:.4f}, {IP_Y[1]:.4f})")


## aclarar por que usamos x asterisco asi 
def calcularX_asterisco(X):
    x_media = X.mean()
    
    diferencias_abs = (X - x_media).abs()
    
    indice_mas_cercano = diferencias_abs.idxmin()
    
    x_asterisco = X.loc[indice_mas_cercano]
    
    return x_asterisco




df = pd.read_csv("./StudentPerformanceFactors.csv")


# Define tus columnas X e Y
variables_predictoras = ['Previous_Scores', 'Attendance', 'Hours_Studied',]
variable_respuesta= 'Exam_Score'

# --- 2. ESTADÍSTICOS BÁSICOS ---
X = df[variables_predictoras]
Y = df[variable_respuesta]



for col in variables_predictoras:
    print(f"\n\n--- Análisis para la variable predictora: {col} ---")
    X_columna = df[col]
    x_asterisco = calcularX_asterisco(X_columna) 
    Y = df[variable_respuesta]

    calculo_regresion_lineal_simple(df[col], Y, x_asterisco=x_asterisco)

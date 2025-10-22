import pandas as pd
import numpy as np
from scipy import stats as st

# --- 1. DATOS DE EJEMPLO ---
# ej: df = pd.read_csv('tu_archivo.csv')

data = {
    'Publicidad_X': [10, 12, 15, 18, 22, 25, 28, 30, 35, 40],
    'Ventas_Y': [88, 90, 95, 100, 105, 110, 115, 120, 130, 140]
}
df = pd.DataFrame(data)

# Define tus columnas X e Y
x_col = 'Publicidad_X'
y_col = 'Ventas_Y'

# --- 2. ESTADÍSTICOS BÁSICOS ---
X = df[x_col]
Y = df[y_col]

n = len(df)
x_promedio = X.mean()
y_promedio = Y.mean()

# Suma de cuadrados y productos cruzados
Sxx = ((X - x_promedio)**2).sum()
Syy = ((Y - y_promedio)**2).sum()
Sxy = ((X - x_promedio) * (Y - y_promedio)).sum()

# --- 3. PARÁMETROS DE REGRESIÓN (β0 y β1) ---
beta_1 = Sxy / Sxx
beta_0 = y_promedio - beta_1 * x_promedio

# --- 4. PREDICCIONES (Ŷ) Y ERRORES (SCE, σ²) ---
# Vector de predicciones (Estimador Y)
estimador_y = beta_0 + beta_1 * X

# Residuos (errores)
residuos = Y - estimador_y

# Suma de Cuadrados de los Errores
SCE = (residuos**2).sum()

# Grados de libertad del error
grados_libertad = n - 2

# Estimador de sigma^2 (Varianza de los residuos o MSE)
sigma_cuadrado_est = SCE / grados_libertad

# Error Estándar Residual (RSE) - la raíz de sigma^2
RSE = np.sqrt(sigma_cuadrado_est)

# --- 5. BONDAD DE AJUSTE (R²) ---
# Coeficiente de Determinación R^2
R_cuadrado = 1 - (SCE / Syy)
# Alternativamente: R_cuadrado = (Sxy**2) / (Sxx * Syy)

# --- 6. INTERVALOS DE CONFIANZA (Asumiendo 95%) ---
alpha = 0.05  # Nivel de significancia (1 - 0.95)
t_critico = st.t.ppf(1 - alpha / 2, df=grados_libertad)

# 6.a. IC para β1 (Pendiente)
SE_beta_1 = np.sqrt(sigma_cuadrado_est / Sxx)
ME_beta_1 = t_critico * SE_beta_1
IC_beta_1 = (beta_1 - ME_beta_1, beta_1 + ME_beta_1)

# 6.b. IC para β0 (Intercepto)
SE_beta_0 = np.sqrt(sigma_cuadrado_est * (1/n + x_promedio**2 / Sxx))
ME_beta_0 = t_critico * SE_beta_0
IC_beta_0 = (beta_0 - ME_beta_0, beta_0 + ME_beta_0)

# 6.c. IC para la Respuesta Media E[Y|X] (ICY)
# Esto no es un solo valor, es una banda para cada X
SE_fit = np.sqrt(sigma_cuadrado_est * (1/n + (X - x_promedio)**2 / Sxx))
ME_fit = t_critico * SE_fit
df['ICY_lim_inf'] = estimador_y - ME_fit
df['ICY_lim_sup'] = estimador_y + ME_fit


# --- 7. IMPRESIÓN DE RESULTADOS ---

print("--- REGRESIÓN LINEAL SIMPLE ---")
print(f"Dataset: {n} observaciones (n)")
print("\n--- Estadísticos Básicos ---")
print(f"Media de X (X̄):     {x_promedio:.4f}")
print(f"Media de Y (Ȳ):     {y_promedio:.4f}")
print(f"Suma de Cuadrados X (Sxx): {Sxx:.4f}")
print(f"Suma de Cuadrados Y (Syy): {Syy:.4f}")
print(f"Suma de Productos XY (Sxy): {Sxy:.4f}")

print("\n--- Parámetros del Modelo ---")
print(f"Pendiente (β1):     {beta_1:.4f}")
print(f"Intercepto (β0):   {beta_0:.4f}")
print(f"Ecuación: Ŷ = {beta_0:.4f} + {beta_1:.4f} * X")

print("\n--- Análisis de Errores ---")
print(f"Suma de Cuadrados del Error (SCE): {SCE:.4f}")
print(f"Estimador de σ² (MSE):           {sigma_cuadrado_est:.4f}")
print(f"Error Estándar Residual (RSE):   {RSE:.4f}")

print("\n--- Bondad de Ajuste ---")
print(f"Coeficiente de Determinación (R²): {R_cuadrado:.4f} ({R_cuadrado*100:.2f}%)")

print("\n--- Intervalos de Confianza (95%) ---")
print(f"IC para β1 (Pendiente):   {IC_beta_1}")
print(f"IC para β0 (Intercepto): {IC_beta_0}")

print("\n--- Predicciones (Ŷ) y Bandas de Confianza (ICY) ---")
df['Y barra (Ŷ)'] = estimador_y
print(df[[x_col, y_col, 'Y barra (Ŷ)', 'ICY_lim_inf', 'ICY_lim_sup']].head())
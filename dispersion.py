import pandas as pd
import matplotlib.pyplot as plt

# 1. Especifica el nombre de tu archivo y las columnas a graficar
archivo_csv = 'StudentPerformanceFactors.csv'  
columna_x = 'Hours_Studied'
columna_y = 'Exam_Score'
nombre_imagen = 'grafico_dispersión.png'

# 2. Carga los datos del CSV
df = pd.read_csv(archivo_csv)

# 3. Crea el gráfico de dispersión
plt.figure(figsize=(8, 6))
plt.scatter(df[columna_x], df[columna_y])

# 4. Añade etiquetas y título (esenciales para claridad)
plt.title(f'Gráfico de Dispersión: {columna_y} vs {columna_x}')
plt.xlabel(columna_x)
plt.ylabel(columna_y)
plt.grid(True)

# 5. Muestra el gráfico
plt.savefig(nombre_imagen)
plt.close()
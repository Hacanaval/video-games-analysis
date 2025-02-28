# Cargar todas las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
from scipy.stats import norm
from scipy.stats import ttest_ind
import datetime
from math import factorial
import math as mt
import scipy.stats as stats
from scipy.stats import levene, ttest_ind

games = pd.read_csv('data/games.csv')
games

#Comprobamos nombres de las columnas, que no tengas espacios
print(games.columns)
#Reemplazamos los nombres de las columnas a minusculas
games.columns = games.columns.str.lower()
#Vericiamos información del df
print(games.info())
#Pasamos year_of_release a entero
games['year_of_release'] = games['year_of_release'].astype('Int64')
#Verificamos la varialbe critic_score para evidenciar si es entero
games_critic_notna = games[games['critic_score'].notna()]
print(games_critic_notna['critic_score'].head(40))
#Pasamos critic_score a entero
games['critic_score'] = games['critic_score'].astype('Int64')


#Veriricamos valores ausentes
print(games.isna().sum())
#Verificamos valores ausentes de name. Eliminamos
games_1 = games[games['name'].isna()]
print(games_1)
games = games.dropna(subset=['name'])
#Verificamos valores ausentes de year_of_release. Eliminamos
games_2 = games[games['year_of_release'].isna()]
print(games_2)
games = games.dropna(subset=['year_of_release'])

#Verificamos duplicados
print(games.duplicated().sum())

games['total_sales'] = games['na_sales'] + games['eu_sales'] + games['jp_sales'] + games['other_sales']
games

#Miramos juegos lanzados por año
games_per_year = games.groupby('year_of_release').size()
print(games_per_year)
# Graficamos en un diagrama de barras:
plt.figure(figsize=(10,5))
games_per_year.plot(kind='bar')
plt.title("Juegos lanzados por año")
plt.xlabel("Año")
plt.ylabel("Cantidad de juegos")
plt.show()

# Observamos el primer y último año de ventas para cada plataforma, a fin de detectar cuándo nacen y cuándo dejan de vender las consolas.
platform_years = games.groupby('platform')['year_of_release'].agg(['min','max'])
print("Años de primera y última venta por plataforma:\n")
print(platform_years.sort_values('min'))

# Filtramos los juegos de la plataforma "DS" cuyo año de lanzamiento sea menor a 2000
ds_before_2000 = games[(games['platform'] == 'DS') & (games['year_of_release'] < 2000)]
ds_before_2000
# Actualizamos el valor de 'year_of_release' en la fila 15957 a 2007
games.at[15957, 'year_of_release'] = 2007

# Verificamos que el cambio se haya realizado correctamente
games.loc[15957]


# Primero, agrupamos las ventas por plataforma para identificar cuáles tienen las mayores ventas totales.
platform_sales = games.groupby('platform')['total_sales'].sum().sort_values(ascending=False)
print('platform_sales\n',platform_sales,'\n')

# Escogemos las cinco plataformas que más vendieron.
top_platforms = platform_sales.head(5).index

# Filtramos el dataset para quedarnos solo con esas plataformas principales.
games_top = games[games['platform'].isin(top_platforms)]

# Usamos groupby y unstack en lugar de pivot para ver cómo evolucionan las ventas por año en cada plataforma.
sales_by_year_platform = (
    games_top
    .groupby(['year_of_release', 'platform'])['total_sales']
    .sum()
    .unstack('platform')  # Convertimos la plataforma en columnas
)
print('sales_by_year_platform',sales_by_year_platform,'\n')
# Graficamos la distribución de ventas por año de estas plataformas.
plt.figure(figsize=(12,6))
sales_by_year_platform.plot()
plt.title("Distribución de Ventas por Año (Plataformas Principales)")
plt.xlabel("Año de Lanzamiento")
plt.ylabel("Ventas Totales (millones)")
plt.show()

# Observamos el primer y último año de ventas para cada plataforma, a fin de detectar cuándo nacen y cuándo dejan de vender las consolas.
platform_years = games.groupby('platform')['year_of_release'].agg(['min','max'])
print("Años de primera y última venta por plataforma:\n")
print(platform_years.sort_values('min'))

# Filtramos solo los datos donde year_of_release sea 1995 o menor
games_old = games[games['year_of_release'] <= 1995]

# Agrupamos las ventas por plataforma para identificar cuáles tienen las mayores ventas totales en este período.
platform_sales_old = games_old.groupby('platform')['total_sales'].sum().sort_values(ascending=False)
print('platform_sales (hasta el año 1995)\n', platform_sales_old, '\n')


# Creamos un nuevo DataFrame con los datos desde 2012 en adelante
games_recent = games[games['year_of_release'] >= 2012]

# Verificamos que solo tengamos datos del período seleccionado
print(games_recent['year_of_release'].unique())

# Mostramos la cantidad de registros en el nuevo DataFrame
print(f"Cantidad de filas en el nuevo DataFrame: {games_recent.shape[0]}")


#Agrupamos las ventas por plataforma usando el nuevo DataFrame con datos desde 2012.
platform_sales_recent = games_recent.groupby('platform')['total_sales'].sum().sort_values(ascending=False)
print('platform_sales (desde 2012)\n', platform_sales_recent, '\n')

# Escogemos las cinco plataformas con mayores ventas en este período.
top_platforms_recent = platform_sales_recent.head(5).index

# Filtramos el dataset para quedarnos solo con esas plataformas principales.
games_top_recent = games_recent[games_recent['platform'].isin(top_platforms_recent)]

# Usamos groupby y unstack en lugar de pivot para ver cómo evolucionan las ventas por año en cada plataforma.
sales_by_year_platform_recent = (
    games_top_recent
    .groupby(['year_of_release', 'platform'])['total_sales']
    .sum()
    .unstack('platform')  # Convertimos la plataforma en columnas
)

print('sales_by_year_platform (desde 2012)\n', sales_by_year_platform_recent, '\n')

# Graficamos la distribución de ventas por año de estas plataformas.
plt.figure(figsize=(12,6))
sales_by_year_platform_recent.plot()
plt.title("Distribución de Ventas por Año (Plataformas Principales desde 2012)")
plt.xlabel("Año de Lanzamiento")
plt.ylabel("Ventas Totales (millones)")
plt.show()

# Observamos el primer y último año de ventas para cada plataforma en este período.
platform_years_recent = games_recent.groupby('platform')['year_of_release'].agg(['min','max'])
print("Años de primera y última venta por plataforma (desde 2012):\n")
print(platform_years_recent.sort_values('min'))

# Creamos un diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma.
plt.figure(figsize=(12,6))
sns.boxplot(data=games_recent, x='platform', y='total_sales')

# Ajustamos el rango del eje Y para mejorar la visualización (si hay valores extremos que distorsionen el gráfico).
plt.ylim(0, 2)  # Ajustable según los datos para visualizar mejor

# Configuramos los títulos y etiquetas
plt.title("Distribución de Ventas Globales por Plataforma (2012-2016)")
plt.xlabel("Plataforma")
plt.ylabel("Ventas Totales (millones)")

plt.xticks(rotation=45)  # Rotamos los nombres de las plataformas si es necesario

plt.show()

# Elegimos la plataforma mas popular, PS4
ps4_data = games_recent[games_recent['platform'] == 'PS4']

# Eliminamos valores no numericos en la variable user_score como tbd
ps4_data['user_score'] = pd.to_numeric(ps4_data['user_score'], errors='coerce')

# Eliminamos filas con valores NaN en 'critic_score' y 'user_score' para hacer el análisis de correlación
ps4_data = ps4_data.dropna(subset=['critic_score', 'user_score', 'total_sales'])


# Creamos gráficos de dispersión para ver la relación entre las reseñas y las ventas
plt.figure(figsize=(12,5))


# Gráfico de dispersión: Critic Score vs. Total Sales
plt.subplot(1,2,1)
plt.scatter(ps4_data['critic_score'], ps4_data['total_sales'])
plt.title('PS4: Ventas vs Puntuación de Críticos')
plt.xlabel('Puntuación de Críticos')
plt.ylabel('Ventas Totales (millones)')

# Gráfico de dispersión: User Score vs. Total Sales
plt.subplot(1,2,2)
plt.scatter(ps4_data['user_score'], ps4_data['total_sales'])
plt.title('PS4: Ventas vs Puntuación de Usuarios')
plt.xlabel('Puntuación de Usuarios')
plt.ylabel('Ventas Totales (millones)')

plt.tight_layout()
plt.show()


# Calculamos la correlación entre la puntuación de críticos y las ventas
critic_corr = ps4_data['critic_score'].corr(ps4_data['total_sales'])

# Calculamos la correlación entre la puntuación de usuarios y las ventas
user_corr = ps4_data['user_score'].corr(ps4_data['total_sales'])

# Imprimimos los resultados
print(f"Correlación entre puntuación de críticos y ventas en PS4: {critic_corr:.2f}")
print(f"Correlación entre puntuación de usuarios y ventas en PS4: {user_corr:.2f}")

# Filtramos los juegos de PS4 con datos completos de ventas y puntuaciones
ps4_games = ps4_data[['name', 'platform', 'total_sales', 'critic_score', 'user_score']]

# Buscamos los mismos juegos en otras plataformas
same_games_other_platforms = games_recent[
    (games_recent['name'].isin(ps4_games['name'])) &
    (games_recent['platform'] != 'PS4')
]

# Unimos la información de ventas de PS4 con las de otras plataformas
comparison_df = pd.concat([ps4_games, same_games_other_platforms], ignore_index=True)

# Ordenamos primero por total_sales (de mayor a menor), luego por nombre y luego por plataforma
comparison_df = comparison_df.sort_values(by=['total_sales', 'name', 'platform'], ascending=[False, True, True])


# Mostramos los datos comparativos
comparison_df.head(20)

# Definimos las 5 plataformas principales
top_5_platforms = ['PS3', 'PS4', 'X360', 'XOne', '3DS']

# Seleccionamos el top 10 de juegos más vendidos en general (sin importar la plataforma)
top_10_games = comparison_df.groupby('name')['total_sales'].sum().nlargest(10).index

# Filtramos el dataset solo con esos juegos y las plataformas seleccionadas
top_10_df = comparison_df[(comparison_df['name'].isin(top_10_games)) & (comparison_df['platform'].isin(top_5_platforms))]

# Pivotamos para tener plataformas como columnas y los juegos en las filas
pivot_df = top_10_df.pivot_table(index='name', columns='platform', values='total_sales', aggfunc='sum', fill_value=0)

# Configuración del gráfico
plt.figure(figsize=(12,6))
bar_width = 0.15  # Reduce el ancho de las barras para mayor separación
x = np.arange(len(pivot_df))  # Posiciones de las barras en el eje X

# Iteramos sobre cada plataforma y graficamos sus barras en paralelo
for i, platform in enumerate(pivot_df.columns):
    plt.bar(x + i * bar_width, pivot_df[platform], width=bar_width, label=platform)

# Ajustamos etiquetas del eje X para mejorar la visualización
plt.xticks(x + bar_width * (len(pivot_df.columns) / 2), pivot_df.index, rotation=30, ha='right')

# Ajustamos los márgenes para que se vean bien las etiquetas
plt.margins(x=0.1)

# Títulos y leyenda
plt.title("Top 10 Juegos Más Vendidos y sus Ventas en las 5 Consolas Principales")
plt.xlabel("Juego")
plt.ylabel("Ventas Totales (millones)")
plt.legend(title="Plataforma", bbox_to_anchor=(1.05, 1), loc='upper left')  # Mueve la leyenda fuera del gráfico

plt.show()


# Agrupamos las ventas por género
genre_sales = games.groupby('genre')['total_sales'].sum().sort_values(ascending=False)

# Mostramos la distribución en un gráfico de barras
plt.figure(figsize=(12,6))
genre_sales.plot(kind='bar', color='c')
plt.title("Ventas Totales por Género de Videojuegos")
plt.xlabel("Género")
plt.ylabel("Ventas Totales (millones)")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Contamos la cantidad de juegos por género
genre_counts = games['genre'].value_counts()

# Creamos el gráfico de barras
plt.figure(figsize=(12,6))
genre_counts.plot(kind='bar', color='b')
plt.title("Frecuencia de Juegos por Género")
plt.xlabel("Género")
plt.ylabel("Cantidad de Juegos")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Calculamos la cantidad de juegos por género
genre_counts = games['genre'].value_counts()

# Calculamos las ventas totales por género
genre_sales = games.groupby('genre')['total_sales'].sum()

# Ordenamos ambos valores según el mismo orden de géneros
genres = genre_counts.index  # Lista de géneros ordenados por cantidad de juegos
genre_sales = genre_sales.reindex(genres)  # Aseguramos el mismo orden

# Configuración del gráfico
fig, ax1 = plt.subplots(figsize=(14,6))

bar_width = 0.4
x = np.arange(len(genres))  # Posiciones en el eje X

# Graficamos la cantidad de juegos
ax1.bar(x - bar_width/2, genre_counts, width=bar_width, label="Cantidad de Juegos", color='b', alpha=0.7)

# Graficamos las ventas totales
ax2 = ax1.twinx()  # Creamos un segundo eje Y
ax2.bar(x + bar_width/2, genre_sales, width=bar_width, label="Ventas Totales (millones)", color='r', alpha=0.7)

# Configuración de ejes y etiquetas
ax1.set_xlabel("Género")
ax1.set_ylabel("Cantidad de Juegos", color='b')
ax2.set_ylabel("Ventas Totales (millones)", color='r')
ax1.set_xticks(x)
ax1.set_xticklabels(genres, rotation=45, ha='right')

# Leyenda combinada
fig.legend(loc='upper right', bbox_to_anchor=(1,1))

plt.title("Cantidad de Juegos vs. Ventas Totales por Género")
plt.show()


# Configuramos el tamaño del gráfico
plt.figure(figsize=(14,6))

# Creamos el boxplot de ventas por género
sns.boxplot(x='genre', y='total_sales', data=games, showfliers=False)  # Ocultamos los outliers para mejor visualización

# Ajustamos el formato del gráfico
plt.xticks(rotation=45, ha='right')
plt.title("Distribución de Ventas por Género")
plt.xlabel("Género")
plt.ylabel("Ventas Totales (millones)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostramos el gráfico
plt.show()


# Calculamos las ventas por plataforma en cada región
platform_sales_na = games.groupby('platform')['na_sales'].sum().sort_values(ascending=False).head(5)
platform_sales_eu = games.groupby('platform')['eu_sales'].sum().sort_values(ascending=False).head(5)
platform_sales_jp = games.groupby('platform')['jp_sales'].sum().sort_values(ascending=False).head(5)

# Graficamos las plataformas más populares en cada región
fig, axes = plt.subplots(1, 3, figsize=(18,6), sharey=True)

platform_sales_na.plot(kind='bar', ax=axes[0], color='b', alpha=0.7)
axes[0].set_title("Top 5 Plataformas en Norteamérica")
axes[0].set_ylabel("Ventas Totales (millones)")

platform_sales_eu.plot(kind='bar', ax=axes[1], color='g', alpha=0.7)
axes[1].set_title("Top 5 Plataformas en Europa")

platform_sales_jp.plot(kind='bar', ax=axes[2], color='r', alpha=0.7)
axes[2].set_title("Top 5 Plataformas en Japón")

plt.show()


# Calculamos las ventas por género en cada región
genre_sales_na = games.groupby('genre')['na_sales'].sum().sort_values(ascending=False).head(5)
genre_sales_eu = games.groupby('genre')['eu_sales'].sum().sort_values(ascending=False).head(5)
genre_sales_jp = games.groupby('genre')['jp_sales'].sum().sort_values(ascending=False).head(5)

# Graficamos los géneros más populares en cada región
fig, axes = plt.subplots(1, 3, figsize=(18,6), sharey=True)

genre_sales_na.plot(kind='bar', ax=axes[0], color='b', alpha=0.7)
axes[0].set_title("Top 5 Géneros en Norteamérica")
axes[0].set_ylabel("Ventas Totales (millones)")

genre_sales_eu.plot(kind='bar', ax=axes[1], color='g', alpha=0.7)
axes[1].set_title("Top 5 Géneros en Europa")

genre_sales_jp.plot(kind='bar', ax=axes[2], color='r', alpha=0.7)
axes[2].set_title("Top 5 Géneros en Japón")

plt.show()



# Agrupamos las ventas por clasificación ESRB en cada región
esrb_sales = games.groupby('rating')[['na_sales', 'eu_sales', 'jp_sales']].sum()

# Graficamos la distribución de ventas según ESRB en cada región
plt.figure(figsize=(12,6))
esrb_sales.plot(kind='bar', colormap='coolwarm', alpha=0.8)
plt.title("Ventas Totales por Clasificación ESRB en Cada Región")
plt.xlabel("Clasificación ESRB")
plt.ylabel("Ventas Totales (millones)")
plt.xticks(rotation=45, ha='right')
plt.legend(["Norteamérica", "Europa", "Japón"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Convertimos 'user_score' a numérico, reemplazando errores como 'tbd' con NaN
games['user_score'] = pd.to_numeric(games['user_score'], errors='coerce')

# Filtramos los datos de Xbox One y PC, eliminando valores NaN
xone_scores = games[(games['platform'] == 'XOne') & (games['user_score'].notna())]['user_score']
pc_scores = games[(games['platform'] == 'PC') & (games['user_score'].notna())]['user_score']

# Prueba de igualdad de varianzas con Levene
stat_levene, p_levene = levene(xone_scores, pc_scores)

print(f"Estadístico de Levene: {stat_levene:.4f}")
print(f"Valor p de Levene: {p_levene:.4f}")

# Decisión basada en la prueba de Levene
if p_levene > 0.05:
    print("No podemos rechazar la hipótesis nula de Levene: las varianzas son iguales.")
    equal_var_assumption = True  # Si p > 0.05, asumimos varianzas iguales
else:
    print("Rechazamos la hipótesis nula de Levene: las varianzas son diferentes.")
    equal_var_assumption = False  # Si p <= 0.05, NO asumimos varianzas iguales

# Prueba t de Student con la decisión de Levene. Como las varianzas son iguales, la prueba t se ejecuta con equal_var=True, lo que hace que la distribución de la prueba sea más precisa
t_stat, p_value = ttest_ind(xone_scores, pc_scores, equal_var=equal_var_assumption)

print(f"\nEstadístico de prueba (t): {t_stat:.4f}")
print(f"Valor p: {p_value:.4f}")

# Evaluamos la hipótesis
alpha = 0.05
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Las calificaciones promedio de los usuarios para Xbox One y PC son diferentes.")
else:
    print("No podemos rechazar la hipótesis nula: No hay evidencia suficiente para afirmar que las calificaciones promedio sean diferentes.")


# Filtramos los datos de los géneros Acción y Deportes, eliminando valores NaN
action_scores = games[(games['genre'] == 'Action') & (games['user_score'].notna())]['user_score']
sports_scores = games[(games['genre'] == 'Sports') & (games['user_score'].notna())]['user_score']

# Prueba de igualdad de varianzas con Levene
stat_levene, p_levene = levene(action_scores, sports_scores)

print(f"Estadístico de Levene: {stat_levene:.4f}")
print(f"Valor p de Levene: {p_levene:.4f}")

# Decisión basada en la prueba de Levene
if p_levene > 0.05:
    print("No podemos rechazar la hipótesis nula de Levene: las varianzas son iguales.")
    equal_var_assumption = True  # Si p > 0.05, asumimos varianzas iguales
else:
    print("Rechazamos la hipótesis nula de Levene: las varianzas son diferentes.")
    equal_var_assumption = False  # Si p <= 0.05, NO asumimos varianzas iguales

# Prueba t de Student con la decisión de Levene
t_stat, p_value = ttest_ind(action_scores, sports_scores, equal_var=equal_var_assumption)

print(f"\nEstadístico de prueba (t): {t_stat:.4f}")
print(f"Valor p: {p_value:.4f}")

# Evaluamos la hipótesis
alpha = 0.05
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Las calificaciones promedio de los usuarios para Acción y Deportes son diferentes.")
else:
    print("No podemos rechazar la hipótesis nula: No hay evidencia suficiente para afirmar que las calificaciones promedio sean diferentes.")


# Creamos una copia del DataFrame para trabajar sin afectar el original
games_no_outliers = games.copy()

# Convertimos 'user_score' a numérico
games_no_outliers['user_score'] = pd.to_numeric(games_no_outliers['user_score'], errors='coerce')

# Eliminamos valores NaN para trabajar solo con datos existentes
games_no_outliers = games_no_outliers.dropna(subset=['user_score'])

# Calculamos los cuartiles y el rango intercuartil (IQR) en 'user_score'
Q1 = games_no_outliers['user_score'].quantile(0.25)
Q3 = games_no_outliers['user_score'].quantile(0.75)
IQR = Q3 - Q1

# Definimos los límites para detectar outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtramos los valores dentro del rango permitido
games_no_outliers = games_no_outliers[
    (games_no_outliers['user_score'] >= lower_bound) &
    (games_no_outliers['user_score'] <= upper_bound)
]

# Mostramos el número de registros eliminados
print(f"Registros antes de filtrar: {len(games)}")
print(f"Registros después de filtrar: {len(games_no_outliers)}")
print(f"Valores atípicos eliminados: {len(games) - len(games_no_outliers)}")


from scipy.stats import levene, ttest_ind

# Filtramos los datos de Xbox One y PC desde el dataset sin valores atípicos
xone_scores_filtered = games_no_outliers[(games_no_outliers['platform'] == 'XOne') & (games_no_outliers['user_score'].notna())]['user_score']
pc_scores_filtered = games_no_outliers[(games_no_outliers['platform'] == 'PC') & (games_no_outliers['user_score'].notna())]['user_score']

# Prueba de igualdad de varianzas con Levene
stat_levene, p_levene = levene(xone_scores_filtered, pc_scores_filtered)

print(f"Estadístico de Levene: {stat_levene:.4f}")
print(f"Valor p de Levene: {p_levene:.4f}")

# Decisión basada en la prueba de Levene
if p_levene > 0.05:
    print("No podemos rechazar la hipótesis nula de Levene: las varianzas son iguales.")
    equal_var_assumption = True  # Si p > 0.05, asumimos varianzas iguales
else:
    print("Rechazamos la hipótesis nula de Levene: las varianzas son diferentes.")
    equal_var_assumption = False  # Si p <= 0.05, NO asumimos varianzas iguales

# Prueba t de Student con la decisión de Levene
t_stat, p_value = ttest_ind(xone_scores_filtered, pc_scores_filtered, equal_var=equal_var_assumption)

print(f"\nEstadístico de prueba (t): {t_stat:.4f}")
print(f"Valor p: {p_value:.4f}")

# Evaluamos la hipótesis
alpha = 0.05
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Las calificaciones promedio de los usuarios para Xbox One y PC son diferentes.")
else:
    print("No podemos rechazar la hipótesis nula: No hay evidencia suficiente para afirmar que las calificaciones promedio sean diferentes.")


# Filtramos los datos de los géneros Acción y Deportes desde el dataset sin valores atípicos
action_scores_filtered = games_no_outliers[(games_no_outliers['genre'] == 'Action') & (games_no_outliers['user_score'].notna())]['user_score']
sports_scores_filtered = games_no_outliers[(games_no_outliers['genre'] == 'Sports') & (games_no_outliers['user_score'].notna())]['user_score']

# Prueba de igualdad de varianzas con Levene
stat_levene, p_levene = levene(action_scores_filtered, sports_scores_filtered)

print(f"Estadístico de Levene: {stat_levene:.4f}")
print(f"Valor p de Levene: {p_levene:.4f}")

# Decisión basada en la prueba de Levene
if p_levene > 0.05:
    print("No podemos rechazar la hipótesis nula de Levene: las varianzas son iguales.")
    equal_var_assumption = True  # Si p > 0.05, asumimos varianzas iguales
else:
    print("Rechazamos la hipótesis nula de Levene: las varianzas son diferentes.")
    equal_var_assumption = False  # Si p <= 0.05, NO asumimos varianzas iguales

# Prueba t de Student con la decisión de Levene
t_stat, p_value = ttest_ind(action_scores_filtered, sports_scores_filtered, equal_var=equal_var_assumption)

print(f"\nEstadístico de prueba (t): {t_stat:.4f}")
print(f"Valor p: {p_value:.4f}")

# Evaluamos la hipótesis
alpha = 0.05
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Las calificaciones promedio de los usuarios para Acción y Deportes son diferentes.")
else:
    print("No podemos rechazar la hipótesis nula: No hay evidencia suficiente para afirmar que las calificaciones promedio sean diferentes.")

import csv
import random

def generar_csv_clusters(
    max_valor,
    dimensiones,
    num_puntos,
    num_clusters,
    nombre_archivo="clusters.csv",
    dispersion=0.05
):
    """
    Genera un CSV con estructura: label, x1, x2, ..., xn
    Cada label representa la clase del cluster (cluster_1, cluster_2, ...)

    :param max_valor: valor máximo de las coordenadas
    :param dimensiones: número de dimensiones
    :param num_puntos: número total de puntos
    :param num_clusters: número de clusters
    :param nombre_archivo: nombre del archivo CSV
    :param dispersion: controla qué tan dispersos están los puntos respecto al centro
    """

    if num_clusters > num_puntos:
        raise ValueError("No puede haber más clusters que puntos")

    # Crear centros aleatorios
    centros = [
        [random.uniform(0, max_valor) for _ in range(dimensiones)]
        for _ in range(num_clusters)
    ]

    # Distribuir puntos equitativamente
    puntos_por_cluster = num_puntos // num_clusters
    resto = num_puntos % num_clusters

    with open(nombre_archivo, mode="w", newline="") as archivo:
        writer = csv.writer(archivo)

        header = ["label"] + [f"x{i+1}" for i in range(dimensiones)]
        writer.writerow(header)

        for i in range(num_clusters):
            cantidad = puntos_por_cluster + (1 if i < resto else 0)

            for j in range(1, cantidad + 1):
                label = f"cluster_{i+1}"

                coordenadas = [
                    random.gauss(centros[i][d], max_valor * dispersion)
                    for d in range(dimensiones)
                ]

                writer.writerow([label] + coordenadas)

    print(f"Archivo '{nombre_archivo}' generado correctamente.")

generar_csv_clusters(100, 2, 100, 6, "reps.csv")
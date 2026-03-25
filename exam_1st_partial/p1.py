import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cargar_tabla(nombre_archivo="data.csv"):
    df = pd.read_csv(nombre_archivo, encoding="utf-8-sig")
    return df

def leer_vector_usuario():
    while True:
        try:
            print("Ingrese el vector de entrada:")
            r = float(input("r = "))
            g = float(input("g = "))
            b = float(input("b = "))
            return (r, g, b)
        except ValueError:
            print("Error: debe ingresar valores numéricos.")

def clasificar_vector(vector):
    r, g, b = vector

    # Clase 3: escala de grises continua
    if 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1 and r == g == b:
        return "clase3 (gris)"

    # Clase 1: RGB principales binarios
    clase1 = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    if (r, g, b) in clase1:
        return "clase1"

    # Clase 2: CMY binarios
    clase2 = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
    if (r, g, b) in clase2:
        return "clase2"

    return "ninguna clase"

def graficar_tabla(df, vector_usuario=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    clase1 = df[df["clase"] == "clase1"]
    clase2 = df[df["clase"] == "clase2"]

    ax.scatter(clase1["r"], clase1["g"], clase1["b"],
               marker='o', s=120, label="Clase 1")
    ax.scatter(clase2["r"], clase2["g"], clase2["b"],
               marker='^', s=120, label="Clase 2")
    
    t = np.linspace(0, 1, 100)
    ax.plot3D(t, t, t, linewidth=2, label="Clase 3")

    for _, fila in df.iterrows():
        if fila["clase"] != "clase3":
            ax.text(fila["r"] + 0.03,
                    fila["g"] + 0.03,
                    fila["b"] + 0.03,
                    f'{fila["color"]}\n({fila["clase"]})',
                    fontsize=9)
            
    if vector_usuario is not None:
        x, y, z = vector_usuario
        ax.scatter(x, y, z, marker='x', s=180, label="Vector ingresado")
        ax.text(x + 0.03, y + 0.03, z + 0.03, f"{vector_usuario}", fontsize=10)

    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_title("Clasificación en el espacio RGB")
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.2, 1.5)
    ax.set_zlim(-0.2, 1.5)
    ax.legend()
    plt.show()

def main():
    df = cargar_tabla("data.csv")

    print("Tabla cargada:")
    print(df)
    print()

    while True:
        vector_usuario = leer_vector_usuario()
        resultado = clasificar_vector(vector_usuario)

        print(f"\nEl vector {vector_usuario} pertenece a: {resultado}")

        graficar_tabla(df, vector_usuario)
        print()

if __name__ == "__main__":
    main()
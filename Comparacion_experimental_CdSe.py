from pathlib import Path
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c

from Puntos_Cuanticos_CdSe import (
    MASA_ELECTRON_KG,
    MASA_HUECO_KG,
    energia_transicion_infinita_ev,
)

from Pozo_finito_CdSe import (
    EG_EV,
    BARRERA_ELECTRON_EV,
    BARRERA_HUECO_EV,
    energia_finita_ev,
)

from Correccion_coulombiana_CdSe import (
    calcular_coeficiente,
    correccion_coulombiana_ev,
)


# ============================================================
# COMPARACION CON DATOS EXPERIMENTALES DE ABSORCION
# ============================================================

# Conversion entre energia del foton y longitud de onda.
HC_EV_NM = h * c / 1.602176634e-19 * 1e9

# El pozo finito usa barreras basadas en CdSe/ZnS.
# Su comparacion con muestras de otro entorno es exploratoria.


def leer_datos(ruta):
    """
    Lee un CSV con estas columnas:
        D_nm,lambda_abs_nm,fuente

    D_nm: diametro experimental, no radio.
    lambda_abs_nm: primer pico de absorcion.
    fuente: identificacion del articulo y figura o tabla.
    """
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontro el archivo:\n{ruta}\n"
            "Debes preparar el CSV con datos experimentales reales."
        )

    diametros = []
    longitudes = []
    fuentes = []

    with ruta.open(
        "r", newline="", encoding="utf-8-sig"
    ) as archivo:
        lector = csv.DictReader(archivo)

        requeridas = {"D_nm", "lambda_abs_nm", "fuente"}

        if not requeridas.issubset(lector.fieldnames or []):
            raise ValueError(
                "El CSV debe tener las columnas: "
                "D_nm,lambda_abs_nm,fuente"
            )

        for numero, fila in enumerate(lector, start=2):
            try:
                diametro = float(fila["D_nm"])
                longitud = float(fila["lambda_abs_nm"])
                fuente = (fila["fuente"] or "").strip()
            except (ValueError, TypeError):
                raise ValueError(
                    f"Revisa los valores numericos de la fila {numero}."
                ) from None

            if (
                not np.isfinite(diametro)
                or not np.isfinite(longitud)
                or diametro <= 0
                or longitud <= 0
            ):
                raise ValueError(
                    f"La fila {numero} contiene valores no validos."
                )

            if not fuente:
                raise ValueError(
                    f"Falta la fuente en la fila {numero}."
                )

            diametros.append(diametro)
            longitudes.append(longitud)
            fuentes.append(fuente)

    if not diametros:
        raise ValueError("El CSV no contiene puntos experimentales.")

    diametros = np.asarray(diametros)
    longitudes = np.asarray(longitudes)
    fuentes = np.asarray(fuentes)

    orden = np.argsort(diametros)

    return (
        diametros[orden],
        longitudes[orden],
        fuentes[orden],
    )


def evaluar_modelos(radios_nm, coeficiente):
    """Evalua los tres modelos en los mismos radios."""
    radios_nm = np.asarray(radios_nm, dtype=float)

    infinito = energia_transicion_infinita_ev(radios_nm)

    brus = infinito + correccion_coulombiana_ev(
        radios_nm, coeficiente
    )

    electron_finito = np.array([
        energia_finita_ev(
            radio,
            MASA_ELECTRON_KG,
            BARRERA_ELECTRON_EV,
        )
        for radio in radios_nm
    ])

    hueco_finito = np.array([
        energia_finita_ev(
            radio,
            MASA_HUECO_KG,
            BARRERA_HUECO_EV,
        )
        for radio in radios_nm
    ])

    # Este modelo finito todavia no incluye atraccion Coulomb.
    finito = EG_EV + electron_finito + hueco_finito

    return {
        "Infinito": infinito,
        "Brus": brus,
        "Finito_escenario": finito,
    }


def main():
    base = Path(__file__).resolve().parent

    argumentos = argparse.ArgumentParser()
    argumentos.add_argument(
        "--datos",
        type=Path,
        default=base / "datos_CdSe.csv",
        help="Ruta al CSV de datos experimentales.",
    )
    args = argumentos.parse_args()

    diametros, longitudes_exp, fuentes = leer_datos(args.datos)

    # Convertir diametro experimental a radio.
    radios_exp = diametros / 2

    # Energia del foton correspondiente al pico de absorcion.
    energias_exp = HC_EV_NM / longitudes_exp

    coeficiente, _ = calcular_coeficiente()

    # Las diferencias se calculan en los radios experimentales.
    predicciones = evaluar_modelos(radios_exp, coeficiente)

    carpeta = base / "resultados_comparacion"
    carpeta.mkdir(exist_ok=True)

    # Conservar datos utilizados y su procedencia.
    with (carpeta / "datos_utilizados.csv").open(
        "w", newline="", encoding="utf-8"
    ) as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([
            "D_nm",
            "R_nm",
            "lambda_abs_nm",
            "E_abs_eV",
            "fuente",
        ])

        escritor.writerows(zip(
            diametros,
            radios_exp,
            longitudes_exp,
            energias_exp,
            fuentes,
        ))

    # Guardar diferencias punto por punto.
    filas_comparacion = []
    resumen = []

    for nombre, energias_modelo in predicciones.items():
        validos = np.isfinite(energias_modelo)
        cantidad = np.count_nonzero(validos)

        diferencia = energias_modelo - energias_exp
        diferencia_pct = 100 * diferencia / energias_exp

        for i in range(len(radios_exp)):
            filas_comparacion.append([
                nombre,
                radios_exp[i],
                energias_exp[i],
                energias_modelo[i],
                diferencia[i],
                diferencia_pct[i],
                fuentes[i],
            ])

        if cantidad:
            rmse = np.sqrt(
                np.mean(diferencia[validos]**2)
            )
            media_absoluta_pct = np.mean(
                np.abs(diferencia_pct[validos])
            )
        else:
            rmse = np.nan
            media_absoluta_pct = np.nan

        resumen.append([
            nombre,
            cantidad,
            rmse,
            media_absoluta_pct,
        ])

        print(
            f"{nombre}: {cantidad}/{len(radios_exp)} puntos; "
            f"RMSE = {rmse:.6f} eV; "
            f"diferencia absoluta media = {media_absoluta_pct:.2f} %"
        )

    with (carpeta / "comparacion_por_punto.csv").open(
        "w", newline="", encoding="utf-8"
    ) as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([
            "modelo",
            "R_nm",
            "E_exp_eV",
            "E_modelo_eV",
            "diferencia_eV",
            "diferencia_porcentaje",
            "fuente",
        ])
        escritor.writerows(filas_comparacion)

    with (carpeta / "resumen_diferencias.csv").open(
        "w", newline="", encoding="utf-8"
    ) as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([
            "modelo",
            "puntos_validos",
            "RMSE_eV",
            "diferencia_absoluta_media_porcentaje",
        ])
        escritor.writerows(resumen)

    # Curvas continuas dentro del intervalo de los datos.
    minimo = radios_exp.min()
    maximo = radios_exp.max()

    # Permitir tambien un CSV con un solo radio.
    if np.isclose(minimo, maximo):
        minimo *= 0.9
        maximo *= 1.1

    radios_curva = np.linspace(minimo, maximo, 150)
    curvas = evaluar_modelos(radios_curva, coeficiente)

    etiquetas = {
        "Infinito": "Infinito sin Coulomb",
        "Brus": "Brus",
        "Finito_escenario": "Finito: escenario CdSe/ZnS sin Coulomb",
    }

    fig, ejes = plt.subplots(1, 2, figsize=(12, 5))

    for nombre, energias in curvas.items():
        estilo = ":" if nombre == "Finito_escenario" else "-"

        ejes[0].plot(
            radios_curva,
            energias,
            estilo,
            label=etiquetas[nombre],
        )

        ejes[1].plot(
            radios_curva,
            HC_EV_NM / energias,
            estilo,
            label=etiquetas[nombre],
        )

    ejes[0].scatter(
        radios_exp,
        energias_exp,
        color="black",
        marker="x",
        zorder=5,
        label="Datos experimentales de absorcion",
    )

    ejes[1].scatter(
        radios_exp,
        longitudes_exp,
        color="black",
        marker="x",
        zorder=5,
        label="Datos experimentales de absorcion",
    )

    ejes[0].set_ylabel("Energia de absorcion / transicion (eV)")
    ejes[1].set_ylabel("Longitud de onda (nm)")

    for eje in ejes:
        eje.set_xlabel("Radio R (nm)")
        eje.grid(alpha=0.3)
        eje.legend(fontsize=8)

    fig.suptitle(
        "CdSe: modelos y datos experimentales\n"
        "El escenario finito requiere interpretar el entorno de la muestra"
    )

    fig.tight_layout()
    fig.savefig(
        carpeta / "comparacion_experimental.png",
        dpi=200,
    )

    print(
        "\nAdvertencia de interpretacion: las diferencias del escenario "
        "CdSe/ZnS no validan sus barreras si los datos son de otro entorno."
    )
    print(f"Resultados guardados en: {carpeta}")

    plt.show()


if __name__ == "__main__":
    main()
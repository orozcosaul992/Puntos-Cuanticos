from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, m_e, elementary_charge, h, c


# ============================================================
# BLOQUE 1: CdSe Y POZO ESFERICO INFINITO
# ============================================================

# Material elegido por el equipo: CdSe.
# Modelo esferico con masas efectivas escalares.
# La energia de banda prohibida corresponde aproximadamente
# a temperatura ambiente.

EG_EV = 1.74
MASA_ELECTRON_KG = 0.13 * m_e
MASA_HUECO_KG = 0.45 * m_e

# Se utilizara cuando agreguemos la interaccion electrica.
EPSILON_R = 9.5

# Intervalo de exploracion elegido para el modelo.
# Estos valores NO son datos experimentales.
RADIOS_NM = np.linspace(1.0, 4.0, 61)

# Producto h*c expresado en eV*nm.
HC_EV_NM = h * c / elementary_charge * 1e9


def energia_confinamiento_ev(radio_nm, masa_kg):
    """
    Energia fundamental de una particula en un pozo
    esferico infinito.

    Entradas:
        radio_nm: radio en nanometros.
        masa_kg: masa efectiva en kilogramos.

    Salida:
        Energia de confinamiento en electronvoltios.
    """
    radio_nm = np.asarray(radio_nm, dtype=float)

    if np.any(~np.isfinite(radio_nm)) or np.any(radio_nm <= 0):
        raise ValueError("El radio debe ser positivo y finito.")

    if not np.isfinite(masa_kg) or masa_kg <= 0:
        raise ValueError("La masa debe ser positiva y finita.")

    # Convertir nanometros a metros.
    radio_m = radio_nm * 1e-9

    # Energia del estado fundamental, calculada en joules.
    energia_j = (
        hbar**2 * np.pi**2
        / (2 * masa_kg * radio_m**2)
    )

    # Convertir joules a electronvoltios.
    return energia_j / elementary_charge


def energia_transicion_infinita_ev(radio_nm):
    """
    Energia de transicion sin atraccion electron-hueco:
    Eg + confinamiento del electron + confinamiento del hueco.
    """
    energia_e = energia_confinamiento_ev(
        radio_nm, MASA_ELECTRON_KG
    )

    energia_h = energia_confinamiento_ev(
        radio_nm, MASA_HUECO_KG
    )

    return EG_EV + energia_e + energia_h


def verificar_bloque_1():
    """Comprobar propiedades conocidas del modelo infinito."""

    # Duplicar el radio debe reducir el confinamiento a 1/4.
    energia_r = energia_confinamiento_ev(
        2.0, MASA_ELECTRON_KG
    )

    energia_2r = energia_confinamiento_ev(
        4.0, MASA_ELECTRON_KG
    )

    assert np.isclose(energia_2r, energia_r / 4)

    # El confinamiento aumenta la transicion respecto a Eg.
    energias = energia_transicion_infinita_ev(RADIOS_NM)

    assert np.all(energias > EG_EV)

    # La energia debe disminuir al aumentar el radio.
    assert np.all(np.diff(energias) < 0)

    # Para un radio muy grande, el confinamiento desaparece.
    assert np.isclose(
        energia_transicion_infinita_ev(1e6),
        EG_EV,
        rtol=1e-8,
    )

    print("Verificaciones del bloque 1: CORRECTAS")


def main():
    verificar_bloque_1()

    energia_e = energia_confinamiento_ev(
        RADIOS_NM, MASA_ELECTRON_KG
    )

    energia_h = energia_confinamiento_ev(
        RADIOS_NM, MASA_HUECO_KG
    )

    energia_total = energia_transicion_infinita_ev(RADIOS_NM)

    # Longitud de onda equivalente a la energia de transicion.
    longitud_nm = HC_EV_NM / energia_total

    # Guardar resultados en una carpeta junto al programa.
    carpeta = Path(__file__).resolve().parent / "resultados_CdSe"
    carpeta.mkdir(exist_ok=True)

    tabla = np.column_stack([
        RADIOS_NM,
        energia_e,
        energia_h,
        energia_total,
        longitud_nm,
    ])

    np.savetxt(
        carpeta / "bloque_1_infinito.csv",
        tabla,
        delimiter=",",
        header=(
            "R_nm,Ee_eV,Eh_eV,"
            "E_transicion_eV,lambda_equivalente_nm"
        ),
        comments="",
    )

    # Mostrar un caso concreto para revisar los resultados.
    radio_prueba = 2.0

    ee_prueba = energia_confinamiento_ev(
        radio_prueba, MASA_ELECTRON_KG
    )

    eh_prueba = energia_confinamiento_ev(
        radio_prueba, MASA_HUECO_KG
    )

    et_prueba = energia_transicion_infinita_ev(radio_prueba)

    print(f"\nRadio: {radio_prueba:.1f} nm")
    print(f"Confinamiento del electron: {ee_prueba:.6f} eV")
    print(f"Confinamiento del hueco:    {eh_prueba:.6f} eV")
    print(f"Energia de transicion:     {et_prueba:.6f} eV")

    # Graficar energia y longitud de onda contra radio.
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))

    ejes[0].plot(
        RADIOS_NM, energia_total, label="Pozo infinito"
    )
    ejes[0].axhline(
        EG_EV,
        color="gray",
        linestyle="--",
        label="Eg del CdSe masivo",
    )

    ejes[0].set_xlabel("Radio R (nm)")
    ejes[0].set_ylabel("Energia de transicion (eV)")
    ejes[0].legend()

    ejes[1].plot(RADIOS_NM, longitud_nm)
    ejes[1].set_xlabel("Radio R (nm)")
    ejes[1].set_ylabel("Longitud de onda equivalente (nm)")

    for eje in ejes:
        eje.grid(alpha=0.3)

    fig.suptitle(
        "CdSe: pozo infinito sin interaccion electron-hueco\n"
        "Eg=1.74 eV; me*=0.13 m0; mh*=0.45 m0"
    )

    fig.tight_layout()
    fig.savefig(
        carpeta / "bloque_1_infinito.png",
        dpi=200,
    )

    print(f"\nResultados guardados en: {carpeta}")

    plt.show()


if __name__ == "__main__":
    main()
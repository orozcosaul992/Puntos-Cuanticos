from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, m_e, elementary_charge, h, c
from scipy.optimize import brentq


# ============================================================
# POZO ESFERICO FINITO APLICADO AL CdSe
# ============================================================

# Parametros del material.
EG_EV = 1.74
MASA_ELECTRON_KG = 0.13 * m_e
MASA_HUECO_KG = 0.45 * m_e

# Barreras basadas en el escenario CdSe/ZnS.
BARRERA_ELECTRON_EV = 1.27
BARRERA_HUECO_EV = 0.60

# Aproximaciones:
# - Estado fundamental con l = 0.
# - Masa constante dentro y fuera para cada particula.
# - Barrera exterior de extension infinita.
# - Sin atraccion electron-hueco en este bloque.

# Radios de exploracion elegidos por el equipo.
RADIOS_NM = np.linspace(1.0, 4.0, 61)

HC_EV_NM = h * c / elementary_charge * 1e9


def energia_infinita_ev(radio_nm, masa_kg):
    """Energia fundamental del pozo esferico infinito."""
    radio_m = np.asarray(radio_nm, dtype=float) * 1e-9

    energia_j = (
        hbar**2 * np.pi**2
        / (2 * masa_kg * radio_m**2)
    )

    return energia_j / elementary_charge


def energia_finita_ev(radio_nm, masa_kg, barrera_ev):
    """
    Energia fundamental del pozo esferico finito.

    Entradas: radio en nm, masa en kg y barrera en eV.
    Salida: energia en eV, medida desde el fondo del pozo.

    Devuelve NaN si no existe un estado ligado.
    """
    parametros = np.array(
        [radio_nm, masa_kg, barrera_ev],
        dtype=float,
    )

    if np.any(~np.isfinite(parametros)) or np.any(parametros <= 0):
        raise ValueError("Los parametros deben ser positivos y finitos.")

    # Convertir a unidades del Sistema Internacional.
    radio_m = radio_nm * 1e-9
    barrera_j = barrera_ev * elementary_charge

    rho = radio_m * np.sqrt(
        2 * masa_kg * barrera_j
    ) / hbar

    # Condicion de existencia del estado ligado fundamental.
    if rho <= np.pi / 2:
        return np.nan

    def ecuacion_frontera(z):
        # z = k*R.
        # Forma equivalente a z*cot(z) = -sqrt(rho^2-z^2).
        exterior = np.sqrt(max(0.0, rho**2 - z**2))

        return (
            z * np.cos(z)
            + exterior * np.sin(z)
        )

    z = brentq(
        ecuacion_frontera,
        np.pi / 2,
        min(np.pi, rho),
        xtol=1e-12,
    )

    energia_j = (
        hbar**2 * z**2
        / (2 * masa_kg * radio_m**2)
    )

    return energia_j / elementary_charge


def main():
    # Confinamiento en el pozo infinito.
    ee_infinito = energia_infinita_ev(
        RADIOS_NM, MASA_ELECTRON_KG
    )

    eh_infinito = energia_infinita_ev(
        RADIOS_NM, MASA_HUECO_KG
    )

    # Confinamiento en el pozo finito.
    ee_finito = np.array([
        energia_finita_ev(
            radio,
            MASA_ELECTRON_KG,
            BARRERA_ELECTRON_EV,
        )
        for radio in RADIOS_NM
    ])

    eh_finito = np.array([
        energia_finita_ev(
            radio,
            MASA_HUECO_KG,
            BARRERA_HUECO_EV,
        )
        for radio in RADIOS_NM
    ])

    # Para comparar la transicion necesitamos ambos estados ligados.
    ligados = np.isfinite(ee_finito) & np.isfinite(eh_finito)

    if not np.any(ligados):
        raise RuntimeError(
            "No hay radios con ambos estados ligados."
        )

    # Comprobaciones fisicas basicas.
    assert np.all(
        (ee_finito[ligados] > 0)
        & (ee_finito[ligados] < BARRERA_ELECTRON_EV)
    )

    assert np.all(
        (eh_finito[ligados] > 0)
        & (eh_finito[ligados] < BARRERA_HUECO_EV)
    )

    assert np.all(ee_finito[ligados] < ee_infinito[ligados])
    assert np.all(eh_finito[ligados] < eh_infinito[ligados])

    print("Comprobaciones de los estados ligados: CORRECTAS")
    print(
        "Radios con ambos estados ligados:",
        f"{np.count_nonzero(ligados)} de {len(RADIOS_NM)}",
    )

    # Energia de transicion sin interaccion coulombiana.
    transicion_infinita = EG_EV + ee_infinito + eh_infinito
    transicion_finita = EG_EV + ee_finito + eh_finito

    longitud_infinita = HC_EV_NM / transicion_infinita
    longitud_finita = HC_EV_NM / transicion_finita

    # Mostrar un caso de referencia.
    radio_prueba = 2.0

    ee = energia_finita_ev(
        radio_prueba,
        MASA_ELECTRON_KG,
        BARRERA_ELECTRON_EV,
    )

    eh = energia_finita_ev(
        radio_prueba,
        MASA_HUECO_KG,
        BARRERA_HUECO_EV,
    )

    print("\nCaso de referencia: R = 2 nm")
    print(f"Confinamiento del electron: {ee:.6f} eV")
    print(f"Confinamiento del hueco:    {eh:.6f} eV")

    if np.isfinite(ee) and np.isfinite(eh):
        transicion = EG_EV + ee + eh

        print(f"Energia de transicion:     {transicion:.6f} eV")
        print(
            "Longitud de onda equivalente:",
            f"{HC_EV_NM / transicion:.2f} nm",
        )
    else:
        print("No existen ambos estados ligados en este radio.")

    # Guardar una tabla para los siguientes bloques.
    carpeta = (
        Path(__file__).resolve().parent
        / "resultados_finito_CdSe"
    )
    carpeta.mkdir(exist_ok=True)

    tabla = np.column_stack([
        RADIOS_NM,
        ee_finito,
        eh_finito,
        transicion_finita,
        transicion_infinita,
        longitud_finita,
        longitud_infinita,
    ])

    np.savetxt(
        carpeta / "comparacion_finito_infinito.csv",
        tabla,
        delimiter=",",
        header=(
            "R_nm,Ee_finito_eV,Eh_finito_eV,"
            "E_transicion_finita_eV,E_transicion_infinita_eV,"
            "lambda_finita_nm,lambda_infinita_nm"
        ),
        comments="",
    )

    # Comparar ambos modelos.
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))

    ejes[0].plot(
        RADIOS_NM,
        transicion_finita,
        label="Pozo finito",
    )

    ejes[0].plot(
        RADIOS_NM,
        transicion_infinita,
        "--",
        label="Pozo infinito",
    )

    ejes[0].axhline(
        EG_EV,
        color="gray",
        linestyle=":",
        label="Eg del CdSe masivo",
    )

    ejes[0].set_xlabel("Radio R (nm)")
    ejes[0].set_ylabel("Energia de transicion (eV)")

    ejes[1].plot(
        RADIOS_NM,
        longitud_finita,
        label="Pozo finito",
    )

    ejes[1].plot(
        RADIOS_NM,
        longitud_infinita,
        "--",
        label="Pozo infinito",
    )

    ejes[1].set_xlabel("Radio R (nm)")
    ejes[1].set_ylabel("Longitud de onda equivalente (nm)")

    for eje in ejes:
        eje.legend()
        eje.grid(alpha=0.3)

    fig.suptitle(
        "CdSe: confinamiento sin atraccion electron-hueco\n"
        "Eg=1.74 eV; me*=0.13 m0; mh*=0.45 m0\n"
        "Ve=1.27 eV; Vh=0.60 eV; masa exterior=interior"
    )

    fig.tight_layout()

    fig.savefig(
        carpeta / "comparacion_finito_infinito.png",
        dpi=200,
    )

    print(f"\nResultados guardados en: {carpeta}")

    plt.show()


if __name__ == "__main__":
    main()
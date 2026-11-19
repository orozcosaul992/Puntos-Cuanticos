from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import elementary_charge, epsilon_0
from scipy.integrate import quad
from scipy.special import sici

from Puntos_Cuanticos_CdSe import (
    EG_EV,
    EPSILON_R,
    RADIOS_NM,
    energia_transicion_infinita_ev,
)


# ============================================================
# CORRECCION COULOMBIANA DE PRIMER ORDEN Y FORMULA DE BRUS
# ============================================================

# Este bloque utiliza los estados fundamentales del pozo infinito.
# La permitividad se considera uniforme.
# No se incluyen efectos de polarizacion en la superficie.


def calcular_coeficiente():
    """
    Calcula el coeficiente de la atraccion electron-hueco
    mediante la integral radial reducida.
    """

    def integrando(s):
        # El limite del integrando en s=0 es cero.
        if s == 0:
            return 0.0

        return (
            8 * np.sin(np.pi * s)**2
            * (
                0.5
                - np.sin(2 * np.pi * s) / (4 * np.pi * s)
            )
        )

    coeficiente, error_estimado = quad(
        integrando,
        0.0,
        1.0,
        epsabs=1e-11,
        epsrel=1e-11,
    )

    return coeficiente, error_estimado


def coeficiente_analitico():
    """Misma integral expresada mediante la funcion integral seno."""
    si_2pi = sici(2 * np.pi)[0]
    si_4pi = sici(4 * np.pi)[0]

    return 2 - si_2pi / np.pi + si_4pi / (2 * np.pi)


def correccion_coulombiana_ev(radio_nm, coeficiente):
    """
    Correccion atractiva, negativa, expresada en eV.
    """
    radio_nm = np.asarray(radio_nm, dtype=float)

    if np.any(~np.isfinite(radio_nm)) or np.any(radio_nm <= 0):
        raise ValueError("Los radios deben ser positivos y finitos.")

    radio_m = radio_nm * 1e-9

    correccion_j = (
        -coeficiente * elementary_charge**2
        / (4 * np.pi * epsilon_0 * EPSILON_R * radio_m)
    )

    return correccion_j / elementary_charge


def main():
    coeficiente, error_integral = calcular_coeficiente()
    referencia = coeficiente_analitico()

    # Comprobar la integral con una expresion independiente.
    assert np.isclose(
        coeficiente,
        referencia,
        rtol=0,
        atol=1e-9,
    ), "La integral no coincide con la expresion analitica."

    energia_infinita = energia_transicion_infinita_ev(RADIOS_NM)

    correccion = correccion_coulombiana_ev(
        RADIOS_NM,
        coeficiente,
    )

    energia_brus = energia_infinita + correccion

    # La atraccion debe reducir la energia.
    assert np.all(correccion < 0)
    assert np.all(energia_brus < energia_infinita)

    print("Comprobaciones del bloque: CORRECTAS")
    print(f"\nCoeficiente integrado: {coeficiente:.9f}")
    print(f"Coeficiente analitico: {referencia:.9f}")
    print(f"Error estimado de la integral: {error_integral:.3e}")
    print(f"Permitividad relativa utilizada: {EPSILON_R}")

    # Caso de referencia.
    radio_prueba = 2.0

    energia_prueba = energia_transicion_infinita_ev(radio_prueba)
    correccion_prueba = correccion_coulombiana_ev(
        radio_prueba,
        coeficiente,
    )

    print("\nCaso de referencia: R = 2 nm")
    print(f"Sin atraccion:        {energia_prueba:.6f} eV")
    print(f"Correccion Coulomb:   {correccion_prueba:.6f} eV")
    print(
        "Energia de Brus:     ",
        f"{energia_prueba + correccion_prueba:.6f} eV",
    )

    # Guardar los resultados.
    carpeta = (
        Path(__file__).resolve().parent
        / "resultados_coulomb_CdSe"
    )
    carpeta.mkdir(exist_ok=True)

    tabla = np.column_stack([
        RADIOS_NM,
        energia_infinita,
        correccion,
        energia_brus,
    ])

    np.savetxt(
        carpeta / "correccion_coulombiana.csv",
        tabla,
        delimiter=",",
        header="R_nm,E_infinito_eV,DeltaE_C_eV,E_Brus_eV",
        comments="",
    )

    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))

    ejes[0].plot(
        RADIOS_NM,
        energia_infinita,
        "--",
        label="Infinito sin atraccion",
    )

    ejes[0].plot(
        RADIOS_NM,
        energia_brus,
        label="Brus: con atraccion",
    )

    ejes[0].axhline(
        EG_EV,
        color="gray",
        linestyle=":",
        label="Eg del CdSe masivo",
    )

    ejes[0].set_xlabel("Radio R (nm)")
    ejes[0].set_ylabel("Energia de transicion (eV)")
    ejes[0].set_title("Efecto de la atraccion electron-hueco")
    ejes[0].legend()

    ejes[1].plot(RADIOS_NM, correccion)
    ejes[1].axhline(0, color="gray", linestyle=":")
    ejes[1].set_xlabel("Radio R (nm)")
    ejes[1].set_ylabel("Correccion coulombiana (eV)")
    ejes[1].set_title("Correccion de primer orden")

    for eje in ejes:
        eje.grid(alpha=0.3)

    fig.suptitle(
        "CdSe: orbitales del pozo infinito\n"
        f"Eg={EG_EV} eV; epsilon_r={EPSILON_R}; "
        f"C={coeficiente:.6f}"
    )

    fig.tight_layout()
    fig.savefig(
        carpeta / "correccion_coulombiana.png",
        dpi=200,
    )

    print(f"\nResultados guardados en: {carpeta}")

    plt.show()


if __name__ == "__main__":
    main()
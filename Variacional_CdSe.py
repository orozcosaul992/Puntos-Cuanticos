from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from numpy.polynomial.legendre import leggauss
from scipy.constants import hbar, elementary_charge, epsilon_0
from scipy.optimize import minimize_scalar
from scipy.special import sici

from Puntos_Cuanticos_CdSe import (
    EG_EV,
    EPSILON_R,
    MASA_ELECTRON_KG,
    MASA_HUECO_KG,
    RADIOS_NM,
    energia_transicion_infinita_ev,
)


# ============================================================
# METODO VARIACIONAL: ELECTRON Y HUECO EN POZO INFINITO
# ============================================================

# Parametros de integracion elegidos para el calculo numerico.
N_INTEGRACION = 40
N_COMPROBACION = 60
TOLERANCIA_ENERGIA_EV = 1e-4

# Intervalo de busqueda del parametro variacional.
BETA_MAX = 4.0

# Coeficientes cineticos expresados en eV*nm^2.
A_E = hbar**2 / (2 * MASA_ELECTRON_KG * elementary_charge) * 1e18
A_H = hbar**2 / (2 * MASA_HUECO_KG * elementary_charge) * 1e18

# Coeficiente Coulomb expresado en eV*nm.
COULOMB = elementary_charge / (4 * np.pi * epsilon_0 * EPSILON_R) * 1e9


def preparar_integracion(n):
    """
    Prepara la integracion sobre:
        s = radio del electron / R
        t = radio del hueco / R
        mu = coseno del angulo entre sus posiciones

    La distancia electron-hueco dividida entre R es d.
    """
    x, w = leggauss(n)

    # Nodos y pesos para s y t entre 0 y 1.
    nodos_radiales = (x + 1) / 2
    pesos_radiales = w / 2

    s = nodos_radiales[:, None, None]
    t = nodos_radiales[None, :, None]
    mu = x[None, None, :]

    # Probabilidades radiales del estado producto sin correlacion.
    pesos = (
        pesos_radiales[:, None, None]
        * pesos_radiales[None, :, None]
        * w[None, None, :] / 2
        * 4
        * np.sin(np.pi * s)**2
        * np.sin(np.pi * t)**2
    )

    d = np.sqrt(s**2 + t**2 - 2 * s * t * mu)

    # Derivadas logaritmicas de las funciones espaciales
    # sin(pi*s)/s y sin(pi*t)/t.
    ls = np.pi / np.tan(np.pi * s) - 1 / s
    lt = np.pi / np.tan(np.pi * t) - 1 / t

    # Terminos cineticos obtenidos de los gradientes
    # de la funcion de prueba correlacionada.
    termino_0 = A_E * ls**2 + A_H * lt**2

    termino_1 = -2 * (
        A_E * ls * (s - t * mu) / d
        + A_H * lt * (t - s * mu) / d
    )

    return pesos, d, termino_0, termino_1


def energia_variacional_ev(beta, radio_nm, integracion):
    """Cociente de Rayleigh de la funcion de prueba."""
    pesos, d, termino_0, termino_1 = integracion

    # El cuadrado del factor de correlacion es exp(-2*beta*d).
    probabilidad = pesos * np.exp(-2 * beta * d)

    # Dividir entre esta integral normaliza el valor esperado.
    norma = np.sum(probabilidad)

    cinetica = (
        termino_0
        + beta * termino_1
        + beta**2 * (A_E + A_H)
    ) / radio_nm**2

    potencial = -COULOMB / (radio_nm * d)

    return EG_EV + np.sum(
        probabilidad * (cinetica + potencial)
    ) / norma


def minimizar_energia(radio_nm, integracion):
    """Busca beta y compara con el extremo beta=0."""
    resultado = minimize_scalar(
        lambda beta: energia_variacional_ev(
            beta, radio_nm, integracion
        ),
        bounds=(0.0, BETA_MAX),
        method="bounded",
        options={"xatol": 1e-6},
    )

    if not resultado.success:
        raise RuntimeError("La minimizacion no convergio.")

    if resultado.x > 0.99 * BETA_MAX:
        raise RuntimeError(
            "El minimo esta cerca del limite. Revisar BETA_MAX."
        )

    energia_sin_correlacion = energia_variacional_ev(
        0.0, radio_nm, integracion
    )

    # Incluir beta=0 entre los candidatos.
    if energia_sin_correlacion < resultado.fun:
        return 0.0, energia_sin_correlacion

    return resultado.x, resultado.fun


def main():
    integracion = preparar_integracion(N_INTEGRACION)

    # Coeficiente de la correccion de primer orden.
    coeficiente = (
        2
        - sici(2 * np.pi)[0] / np.pi
        + sici(4 * np.pi)[0] / (2 * np.pi)
    )

    energia_brus = (
        energia_transicion_infinita_ev(RADIOS_NM)
        - coeficiente * COULOMB / RADIOS_NM
    )

    resultados = np.array([
        minimizar_energia(radio, integracion)
        for radio in RADIOS_NM
    ])

    betas = resultados[:, 0]
    energias = resultados[:, 1]

    # Comparar con beta=0 usando la misma integracion.
    energia_beta_cero = np.array([
        energia_variacional_ev(0.0, radio, integracion)
        for radio in RADIOS_NM
    ])

    assert np.all(energias <= energia_beta_cero + 1e-8)

    # Comprobar que beta=0 reproduce Brus aproximadamente.
    diferencia_brus = np.max(
        np.abs(energia_beta_cero - energia_brus)
    )

    assert diferencia_brus < TOLERANCIA_ENERGIA_EV

    # Comprobar convergencia de la energia minimizada en R=2 nm.
    radio_prueba = 2.0

    beta_prueba, energia_prueba = minimizar_energia(
        radio_prueba, integracion
    )

    integracion_fina = preparar_integracion(N_COMPROBACION)

    _, energia_fina = minimizar_energia(
        radio_prueba, integracion_fina
    )

    cambio = abs(energia_fina - energia_prueba)

    assert cambio < TOLERANCIA_ENERGIA_EV, (
        "La integracion necesita mayor precision."
    )

    brus_prueba = (
        energia_transicion_infinita_ev(radio_prueba)
        - coeficiente * COULOMB / radio_prueba
    )

    print("Comprobaciones del bloque: CORRECTAS")
    print("\nCaso de referencia: R = 2 nm")
    print(f"Beta optimo:          {beta_prueba:.6f}")
    print(f"Energia de Brus:      {brus_prueba:.6f} eV")
    print(f"Energia variacional:  {energia_prueba:.6f} eV")
    print(f"Cambio al refinar:    {cambio:.3e} eV")

    # Guardar los resultados.
    carpeta = (
        Path(__file__).resolve().parent
        / "resultados_variacional_CdSe"
    )
    carpeta.mkdir(exist_ok=True)

    tabla = np.column_stack([
        RADIOS_NM,
        betas,
        energia_brus,
        energias,
        energia_brus - energias,
    ])

    np.savetxt(
        carpeta / "variacional.csv",
        tabla,
        delimiter=",",
        header=(
            "R_nm,beta_optimo,E_Brus_eV,"
            "E_variacional_eV,disminucion_respecto_Brus_eV"
        ),
        comments="",
    )

    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))

    ejes[0].plot(
        RADIOS_NM, energia_brus, "--", label="Brus"
    )
    ejes[0].plot(
        RADIOS_NM, energias, label="Variacional"
    )

    ejes[0].set_xlabel("Radio R (nm)")
    ejes[0].set_ylabel("Energia de transicion (eV)")
    ejes[0].set_title("Comparacion de aproximaciones")
    ejes[0].legend()

    # Mostrar la diferencia porque las curvas pueden superponerse.
    ejes[1].plot(
        RADIOS_NM,
        1000 * (energia_brus - energias),
    )

    ejes[1].set_xlabel("Radio R (nm)")
    ejes[1].set_ylabel("E_Brus - E_variacional (meV)")
    ejes[1].set_title("Reduccion de energia al incluir correlacion")

    for eje in ejes:
        eje.grid(alpha=0.3)

    fig.suptitle(
        "CdSe: metodo variacional en pozo infinito\n"
        f"Eg={EG_EV} eV; epsilon_r={EPSILON_R}"
    )

    fig.tight_layout()

    fig.savefig(
        carpeta / "variacional.png",
        dpi=200,
    )

    print(f"\nResultados guardados en: {carpeta}")
    plt.show()


if __name__ == "__main__":
    main()
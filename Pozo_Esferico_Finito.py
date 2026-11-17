import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# ============================================================
# POZO ESFERICO FINITO: ESTADO FUNDAMENTAL CON l = 0
# ============================================================

# Unidades adimensionales.
HBAR = 1.0
MASA = 1.0

# Parametros elegidos para explorar el modelo.
RADIO = 1.0
BARRERA = 20.0


def energia_infinita(radio):
    """Energia fundamental del pozo esferico infinito."""
    return HBAR**2 * np.pi**2 / (2 * MASA * radio**2)


def energia_finita(radio, barrera):
    """
    Calcula la energia fundamental del pozo esferico finito.

    La energia se mide desde el fondo del pozo.
    Devuelve NaN si no existe un estado ligado.
    """
    if not np.isfinite(radio) or radio <= 0:
        raise ValueError("El radio debe ser positivo y finito.")

    if not np.isfinite(barrera) or barrera <= 0:
        raise ValueError("La barrera debe ser positiva y finita.")

    # Parametro adimensional que combina radio, masa y barrera.
    rho = radio * np.sqrt(2 * MASA * barrera) / HBAR

    # Una esfera demasiado pequena o una barrera demasiado baja
    # puede no sostener un estado ligado.
    if rho <= np.pi / 2:
        return np.nan

    def ecuacion_frontera(z):
        """
        Condicion de continuidad para z = k*R.

        Se usa una forma sin la division entre sin(z).
        """
        exterior = np.sqrt(max(0.0, rho**2 - z**2))

        return (
            z * np.cos(z)
            + exterior * np.sin(z)
        )

    # La raiz del estado fundamental esta en este intervalo.
    inferior = np.pi / 2
    superior = min(np.pi, rho)

    z = brentq(
        ecuacion_frontera,
        inferior,
        superior,
        xtol=1e-12,
    )

    return HBAR**2 * z**2 / (2 * MASA * radio**2)


def funcion_radial(r, radio, barrera, energia):
    """
    Funcion radial reducida u(r), normalizada.

    Para l = 0:
        psi(r, theta, phi) = u(r) / r * Y00(theta, phi).

    La probabilidad radial es |u(r)|^2.
    """
    k = np.sqrt(2 * MASA * energia) / HBAR
    kappa = np.sqrt(2 * MASA * (barrera - energia)) / HBAR

    # Integral de |u|^2 dentro del pozo, antes de normalizar.
    norma_interior = (
        radio / 2
        - np.sin(2 * k * radio) / (4 * k)
    )

    # Integral de la cola exponencial desde R hasta infinito.
    norma_exterior = np.sin(k * radio)**2 / (2 * kappa)

    amplitud = 1 / np.sqrt(norma_interior + norma_exterior)

    u = np.zeros_like(r, dtype=float)
    dentro = r <= radio

    u[dentro] = amplitud * np.sin(k * r[dentro])

    u[~dentro] = (
        amplitud
        * np.sin(k * radio)
        * np.exp(-kappa * (r[~dentro] - radio))
    )

    return u


def main():
    energia = energia_finita(RADIO, BARRERA)

    if np.isnan(energia):
        print("No existe un estado ligado para estos parametros.")
        return

    referencia = energia_infinita(RADIO)

    print("Unidades adimensionales: hbar = 1, masa = 1")
    print(f"Radio: {RADIO}")
    print(f"Altura de barrera: {BARRERA}")
    print(f"Energia del pozo finito:   {energia:.6f}")
    print(f"Energia del pozo infinito: {referencia:.6f}")

    # Elegir una region que muestre la cola exterior.
    kappa = np.sqrt(2 * MASA * (BARRERA - energia)) / HBAR
    r_max = RADIO + 8 / kappa
    r = np.linspace(0, r_max, 1500)

    u = funcion_radial(r, RADIO, BARRERA, energia)

    # Comparar energias para diferentes radios.
    radios = np.linspace(0.2, 3.0, 200)

    energias_finitas = np.array([
        energia_finita(radio, BARRERA)
        for radio in radios
    ])

    energias_infinitas = energia_infinita(radios)

    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))

    ejes[0].plot(
        radios,
        energias_finitas,
        label="Pozo finito",
    )

    ejes[0].plot(
        radios,
        energias_infinitas,
        "--",
        label="Pozo infinito",
    )

    ejes[0].axhline(
        BARRERA,
        color="gray",
        linestyle=":",
        label="Altura de barrera V0",
    )

    ejes[0].set_xlabel("Radio R")
    ejes[0].set_ylabel("Energia")
    ejes[0].set_ylim(0, 1.3 * BARRERA)
    ejes[0].set_title("Energia fundamental contra radio")
    ejes[0].legend()

    ejes[1].plot(r, np.abs(u)**2)

    ejes[1].axvline(
        RADIO,
        color="gray",
        linestyle="--",
        label="Frontera del pozo",
    )

    ejes[1].set_xlabel("Distancia al centro r")
    ejes[1].set_ylabel("Densidad de probabilidad radial |u(r)|²")
    ejes[1].set_title(f"Estado fundamental: R={RADIO}, V0={BARRERA}")
    ejes[1].legend()

    for eje in ejes:
        eje.grid(alpha=0.3)

    fig.suptitle("Pozo esferico finito: unidades adimensionales")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
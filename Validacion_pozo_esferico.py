from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from Pozo_Esferico_finito import energia_finita, energia_infinita


# ============================================================
# VALIDACION: LIMITE DEL POZO ESFERICO INFINITO
# ============================================================

# Mantenemos fijo el radio y aumentamos la altura de la barrera.
# Utilizamos las mismas unidades adimensionales del codigo anterior.
RADIO_PRUEBA = 1.0

BARRERAS = np.array([
    20.0,
    100.0,
    1000.0,
    10000.0,
    1000000.0,
    100000000.0,
])

# Criterio numerico elegido para esta comprobacion:
# error relativo menor que 0.001, equivalente a 0.1 %.
TOLERANCIA_RELATIVA = 1e-3


def main():
    # Resultado analitico que sirve como referencia.
    referencia = energia_infinita(RADIO_PRUEBA)

    # Resolver el pozo finito para cada altura de barrera.
    energias = np.array([
        energia_finita(RADIO_PRUEBA, barrera)
        for barrera in BARRERAS
    ])

    # Diferencia respecto a la referencia infinita.
    errores_relativos = np.abs(energias - referencia) / referencia
    errores_porcentaje = 100 * errores_relativos

    print("VALIDACION DEL POZO ESFERICO FINITO")
    print(f"Radio fijo: {RADIO_PRUEBA}")
    print(f"Energia del pozo infinito: {referencia:.8f}")
    print()

    print(f"{'V0':>12} {'E finito':>15} {'Error (%)':>15}")

    for barrera, energia, error in zip(
        BARRERAS, energias, errores_porcentaje
    ):
        print(f"{barrera:12.1e} {energia:15.8f} {error:15.6f}")

    # Comprobar que existen los estados calculados.
    assert np.all(np.isfinite(energias)), (
        "Falta algun estado ligado."
    )

    # La energia ligada debe estar entre el fondo y la barrera.
    assert np.all(energias > 0)
    assert np.all(energias < BARRERAS)

    # Para barreras finitas, la energia debe ser menor
    # que la referencia del pozo infinito.
    assert np.all(energias < referencia)

    # Al subir la barrera, la energia debe aumentar.
    assert np.all(np.diff(energias) > 0), (
        "La energia no aumenta al elevar la barrera."
    )

    # La diferencia respecto al infinito debe disminuir.
    assert np.all(np.diff(errores_relativos) < 0), (
        "El error no disminuye al elevar la barrera."
    )

    # La barrera mas alta debe satisfacer el criterio elegido.
    assert errores_relativos[-1] < TOLERANCIA_RELATIVA, (
        "El ultimo resultado no alcanza la tolerancia."
    )

    print("\nValidacion: CORRECTA")
    print(
        "Error final:",
        f"{errores_porcentaje[-1]:.6f} %",
    )

    # Guardar la tabla de validacion.
    carpeta = (
        Path(__file__).resolve().parent / "resultados_pozo_finito"
    )
    carpeta.mkdir(exist_ok=True)

    tabla = np.column_stack([
        BARRERAS,
        energias,
        np.full_like(energias, referencia),
        errores_porcentaje,
    ])

    np.savetxt(
        carpeta / "validacion_limite_infinito.csv",
        tabla,
        delimiter=",",
        header="V0,E_finito,E_infinito,error_porcentaje",
        comments="",
    )

    # Mostrar la aproximacion de la energia y la reduccion del error.
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))

    ejes[0].semilogx(
        BARRERAS,
        energias,
        "o-",
        label="Pozo finito",
    )

    ejes[0].axhline(
        referencia,
        color="gray",
        linestyle="--",
        label="Pozo infinito",
    )

    ejes[0].set_xlabel("Altura de barrera V0")
    ejes[0].set_ylabel("Energia fundamental")
    ejes[0].set_title("Aproximacion al resultado infinito")
    ejes[0].legend()

    ejes[1].loglog(
        BARRERAS,
        errores_porcentaje,
        "o-",
        label="Error relativo",
    )

    ejes[1].axhline(
        100 * TOLERANCIA_RELATIVA,
        color="gray",
        linestyle="--",
        label="Criterio elegido: 0.1 %",
    )

    ejes[1].set_xlabel("Altura de barrera V0")
    ejes[1].set_ylabel("Error relativo (%)")
    ejes[1].set_title("Disminucion del error")
    ejes[1].legend()

    for eje in ejes:
        eje.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Validacion del pozo esferico finito\n"
        "Radio fijo R=1; hbar=1; masa=1"
    )

    fig.tight_layout()

    fig.savefig(
        carpeta / "validacion_limite_infinito.png",
        dpi=200,
    )

    print(f"Resultados guardados en: {carpeta}")

    plt.show()


if __name__ == "__main__":
    main()
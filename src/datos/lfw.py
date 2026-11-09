"""Carga de LFW. La selección de identidades la completa el alumno."""

from __future__ import annotations

from collections import Counter
import numpy as np
from sklearn.datasets import fetch_lfw_people

from src.config import DIR_CACHE
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros


class CargadorLFW(CargadorRostros):
    """Fotografías de LFW con mayor diversidad de pose, luz y fondo.

    TODO(alumno): implementar ``cargar`` y devolver un ``ConjuntoRostros``.
    Cuando este método deje de lanzar ``EtapaPendienteAlumno``,
    ``python main.py lfw`` reutiliza el mismo bucle de experimentos que Olivetti.

    Pasos esperados:

    1. ``fetch_lfw_people(min_faces_per_person=20, resize=0.5, data_home=...)``.
    2. Conservar como máximo 50 identidades, las de mayor frecuencia.
    3. Reindexar ``y`` con ``np.unique(..., return_inverse=True)`` y guardar los nombres.
    4. Leer alto y ancho desde ``images.shape``.
    5. Llamar a ``validar_conjunto`` antes de retornar.
    """

    nombre = "lfw"

    def cargar(self) -> ConjuntoRostros:
        rostros = fetch_lfw_people(
            min_faces_per_person=20,
            resize=0.5,
            data_home=str(DIR_CACHE),
            download_if_missing=True,
        )
        images = np.asarray(rostros.images)
        X = np.asarray(rostros.data, dtype=float)
        y = np.asarray(rostros.target, dtype=int)
        target_names = list(rostros.target_names)

        # 2. Conservar como máximo 50 identidades, las de mayor frecuencia.
        conteo = Counter(y)
        top50_clases = [cls for cls, _ in conteo.most_common(50)]
        mascara = np.isin(y, top50_clases)

        images = images[mascara]
        X = X[mascara]
        y_filtrado = y[mascara]

        # 3. Reindexar ``y`` con ``np.unique(..., return_inverse=True)`` y guardar los nombres.
        clases_unicas, y_reindexado = np.unique(y_filtrado, return_inverse=True)
        nombres = [target_names[cls] for cls in clases_unicas]

        alto, ancho = int(images.shape[1]), int(images.shape[2])

        conjunto = ConjuntoRostros(
            nombre=self.nombre,
            images=images,
            X=X,
            y=y_reindexado,
            nombres_identidades=nombres,
            alto=alto,
            ancho=ancho,
        )
        validar_conjunto(conjunto)
        return conjunto

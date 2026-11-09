"""Fisherfaces: PCA comprime y LDA discrimina. Lo implementa el alumno."""

from __future__ import annotations

import numpy as np

from src.embeddings.base import ExtractorEmbedding
from src.embeddings.lda import ExtractorLDA
from src.embeddings.pca import ExtractorPCA
from src.excepciones import EtapaPendienteAlumno


class ExtractorFisherfaces(ExtractorEmbedding):
    """Proyección z = W_LDAᵀ W_PCAᵀ (x − μ), ajustada solo con TRAIN.

    TODO(alumno): implementar ``fit`` y ``transform``.

    Orden obligatorio:

    1. Ajustar PCA con ``d_pca`` solo sobre ``X_train`` (ahí se estima μ).
    2. Ajustar LDA con ``d_lda`` sobre ese PCA de TRAIN, usando ``y_train``.
    3. Transformar TEST con las dos proyecciones, sin volver a ajustar.

    Contrato: ``d_lda <= min(C - 1, d_pca)``. Revise también el rango y
    la cantidad de muestras. Puede reutilizar ``ExtractorPCA`` y ``ExtractorLDA``.
    """

    nombre = "fisherfaces"

    def __init__(self, d_pca: int, d_lda: int) -> None:
        if d_pca < 1 or d_lda < 1:
            raise ValueError("Fisherfaces requiere d_pca y d_lda positivos.")
        if d_lda > d_pca:
            raise ValueError(
                f"d_lda={d_lda} supera d_pca={d_pca}. "
                "Debe cumplirse d_lda <= min(C-1, d_pca)."
            )
        self.d_pca = int(d_pca)
        self.d_lda = int(d_lda)
        self._pca = ExtractorPCA(n_components=self.d_pca)
        self._lda = ExtractorLDA(n_components=self.d_lda)

    @property
    def dimension(self) -> int:
        return self.d_lda

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorFisherfaces:
        if y is None:
            raise ValueError("Fisherfaces necesita y_train: LDA aprende con las identidades.")
        n_clases = int(np.unique(y).size)
        maximo = min(n_clases - 1, self.d_pca)
        if self.d_lda > maximo:
            raise ValueError(
                f"Fisherfaces con d_lda={self.d_lda} supera min(C-1, d_pca)={maximo}."
            )
        X_pca = self._pca.fit_transform(X)
        self._lda.fit(X_pca, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_pca = self._pca.transform(X)
        Z = self._lda.transform(X_pca)
        return self.comprobar(Z, X.shape[0])

    def _pendiente(self) -> EtapaPendienteAlumno:
        return EtapaPendienteAlumno(
            "embeddings.fisherfaces.ExtractorFisherfaces",
            "Ajuste PCA(d_pca) solo con X_train. Ajuste LDA(d_lda) sobre el PCA "
            "de TRAIN usando y_train. Pase TEST por las dos transformaciones, "
            f"sin reajustar. Aquí d_pca={self.d_pca} y d_lda={self.d_lda}; "
            "compruebe d_lda <= min(C-1, d_pca).",
        )

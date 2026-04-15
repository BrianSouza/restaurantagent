"""Clientes auxiliares para integração com ChromaDB."""

from chromadb.config import System
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetryClient(ProductTelemetryClient):
    """Implementação sem efeito para desabilitar envio de telemetria."""

    def __init__(self, system: System):
        super().__init__(system)

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None

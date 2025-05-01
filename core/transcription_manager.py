# transcription_manager.py (Refatorado para fornecer infraestrutura apenas)
import threading
from typing import Optional
import torch
from core.transcriber import Transcriber
from utils.logger import get_logger

logger = get_logger(__name__)

class TranscriptionManager:
    def __init__(self, max_concurrent: int = 2):
        self._lock = threading.Lock()
        self._semaforo = threading.Semaphore(max_concurrent)
        self._model_cache = {}  # Reaproveitamento de modelos por nome
        self._device = self._detectar_dispositivo()
        self._max_concurrent = max_concurrent
        logger.info(f"🧠 TranscriptionManager pronto com até {max_concurrent} transcrições simultâneas")

    def _detectar_dispositivo(self) -> str:
        if torch.cuda.is_available():
            try:
                mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
                if mem > 1500:
                    logger.info("⚡ Usando CUDA para transcrição")
                    return "cuda"
            except Exception as e:
                logger.warning(f"⚠️ Erro ao detectar GPU: {e}")
        logger.info("🖥️ Usando CPU para transcrição")
        return "cpu"

    def get_transcriber(self, model: str = "base") -> Optional[Transcriber]:
        self._semaforo.acquire()
        try:
            with self._lock:
                chave = f"{model}-{self._device}"
                if chave not in self._model_cache:
                    logger.debug(f"📦 Carregando novo modelo: {model} ({self._device})")
                    self._model_cache[chave] = Transcriber(modelo_whisper=model, device_override=self._device)
                return self._model_cache[chave]
        except Exception as e:
            logger.exception(f"❌ Erro ao obter transcriber: {e}")
            return None

    def liberar_transcriber(self):
        self._semaforo.release()

# Instância global
transcription_manager = TranscriptionManager(max_concurrent=2)
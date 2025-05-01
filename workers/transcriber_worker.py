# transcriber_worker.py (Otimizado para GPU)
from PyQt6.QtCore import QObject, pyqtSignal, QRunnable
import os
import sys
import traceback
from typing import Optional, Callable, Union
from utils.logger import get_logger
import torch

# Adiciona o diretório atual ao caminho do Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = get_logger(__name__)

class TranscriberSignals(QObject):
    resultado = pyqtSignal(dict)
    erro = pyqtSignal(dict)
    status = pyqtSignal(str)
    progresso = pyqtSignal(int, int)

class TranscriberWorker(QRunnable):
    def __init__(
        self,
        caminho_video: str,
        idioma: Optional[str] = None,
        cancelado: Optional[Union[Callable[[], bool], object]] = None,
        device_id: int = 0,
        **kwargs
    ):
        super().__init__()
        self.caminho_video = caminho_video
        self.idioma = idioma
        self.cancelado = cancelado
        self.device_id = device_id
        self.signals = TranscriberSignals()
        self.setAutoDelete(True)  # Limpa automaticamente após execução

    def run(self):
        logger.debug(f"[TranscriberWorker] 🚀 Iniciado para: {self.caminho_video}")
        resultado = {}

        try:
            if not os.path.exists(self.caminho_video):
                raise FileNotFoundError(f"❌ Arquivo não encontrado: {self.caminho_video}")

            # Configura GPU específica se disponível
            if torch.cuda.is_available():
                torch.cuda.set_device(self.device_id)
                logger.info(f"🎮 Usando GPU {self.device_id}: {torch.cuda.get_device_name()}")
            else:
                logger.warning("⚠️ GPU não disponível, usando CPU")

            transcritor = self._criar_transcritor()

            self.signals.status.emit("🎯 Iniciando transcrição...")
            resultado = transcritor.transcrever_audio(audio_path=self.caminho_video)

            if not resultado.get('sucesso', False):
                raise Exception(resultado.get('erro', 'Erro desconhecido na transcrição'))

            # Salva o arquivo de texto
            pasta_saida = os.path.dirname(self.caminho_video)
            nome_base = os.path.splitext(os.path.basename(self.caminho_video))[0]
            nome_arquivo = f"{nome_base}_transcricao.txt"

            from utils.utils import salvar_txt
            if salvar_txt(nome_arquivo, resultado['texto'], pasta_saida):
                logger.info(f"✅ Transcrição salva em: {os.path.join(pasta_saida, nome_arquivo)}")
                resultado['arquivo_transcricao'] = os.path.join(pasta_saida, nome_arquivo)
            else:
                logger.error("❌ Erro ao salvar arquivo de transcrição")

            # Atualiza o progresso para 100% quando finalizar
            self._atualizar_progresso(100)
            
            # Emite o resultado antes de qualquer limpeza
            self.signals.resultado.emit(resultado)
            
            # Limpa recursos
            if hasattr(transcritor, 'liberar_gpu'):
                transcritor.liberar_gpu(self.device_id)
            del transcritor

        except Exception as e:
            erro_msg = traceback.format_exc()
            logger.exception(f"[TranscriberWorker] ❌ Falha crítica na transcrição:\n{erro_msg}")
            
            # Emite o erro antes de qualquer limpeza
            self.signals.erro.emit({
                "erro": str(e),
                "traceback": erro_msg,
                "video_path": self.caminho_video,
                "sucesso": False
            })
            
            # Limpa recursos mesmo em caso de erro
            if 'transcritor' in locals():
                if hasattr(transcritor, 'liberar_gpu'):
                    transcritor.liberar_gpu(self.device_id)
                del transcritor

    def _criar_transcritor(self):
        try:
            from core.transcriber import Transcriber
            device_str = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
            transcritor = Transcriber(device_override=device_str)
            logger.debug(f"[TranscriberWorker] ✅ Transcriber instanciado com {device_str}")
            return transcritor
        except Exception as e:
            logger.exception("[TranscriberWorker] ❌ Erro ao instanciar Transcriber")
            raise

    def _atualizar_progresso(self, pct: int):
        """Callback para atualizar o progresso da transcrição"""
        self.signals.progresso.emit(pct, 100)  # Total sempre 100 para porcentagem
        self.signals.status.emit(f"🎯 Transcrevendo... {pct}%")

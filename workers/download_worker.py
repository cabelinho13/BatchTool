# download_worker.py (Otimizado para CPU)
from PyQt6.QtCore import QObject, pyqtSignal, QRunnable
from core.downloader import baixar_video_e_thumbnail
from core.processor import reencodar_video, reencodar_imagem, calcular_md5
from utils.logger import get_logger
from utils.helpers import verificar_flag_cancelado
import traceback
import os
import time
import psutil

logger = get_logger(__name__)

class WorkerSignals(QObject):
    progresso = pyqtSignal(int, int)
    status = pyqtSignal(str)
    status_linha = pyqtSignal(int, str)
    erro = pyqtSignal(object)
    finalizado = pyqtSignal(dict)

class DownloadWorker(QRunnable):
    def __init__(self, link, caminho_saida, row=-1, logar_callback=None):
        super().__init__()
        self.link = link
        self.caminho_saida = caminho_saida
        self.row = row
        self.signals = WorkerSignals()
        self.flag_cancelado = {"valor": False}
        self.caminho_video_final = None
        self.caminho_video = None
        self.caminho_thumb = None
        self.logar_callback = logar_callback or (lambda msg: logger.info(msg))
        
        # Configuração de prioridade para CPU
        self.setAutoDelete(True)  # Limpa automaticamente após execução

    def run(self) -> None:
        # Define afinidade do processo atual para CPU
        try:
            process = psutil.Process()
            # Usa apenas cores físicos, não lógicos
            cores_fisicos = list(range(psutil.cpu_count(logical=False)))
            process.cpu_affinity(cores_fisicos)
        except Exception as e:
            logger.warning(f"⚠️ Não foi possível definir afinidade CPU: {e}")

        logger.debug(f"[DownloadWorker] 🚀 Iniciando linha {self.row} | link: {self.link}")
        inicio_total = time.time()

        try:
            if self._checar_cancelamento(): return
            self._status(self.row, "🔗 Iniciando processamento...")

            self._executar_etapa("⬇️ Baixando vídeo...", self._baixar)
            self._executar_etapa("🎮 Reencodificando vídeo...", self._reencodar)

            if not self.caminho_video_final or not os.path.exists(self.caminho_video_final):
                raise FileNotFoundError("❌ Caminho do vídeo final é inválido ou não existe.")

            duracao = round(time.time() - inicio_total, 1)
            md5_final = calcular_md5(self.caminho_video_final)

            self._status(self.row, f"✅ Download completo 🕒 {duracao}s | MD5: {md5_final}")

            self.signals.finalizado.emit({
                "row": self.row,
                "video_path": self.caminho_video_final,
                "thumb_path": self.caminho_thumb,
                "hash": md5_final,
                "link": self.link
            })

        except Exception as e:
            msg = f"❌ Erro ao processar vídeo: {e}"
            self._status(self.row, msg)
            logger.exception(f"[DownloadWorker] 💥 Erro fatal na linha {self.row}: {e}")
            self._registrar_erro(traceback.format_exc())
            self.signals.erro.emit({"row": self.row, "erro": str(e)})

    def _progresso(self, atual: int, total: int):
        self.signals.progresso.emit(atual, total)

    def cancelar(self):
        self.flag_cancelado["valor"] = True
        self.logar_callback(f"🚩 Cancelamento solicitado para vídeo {self.link}")

    def _checar_cancelamento(self) -> bool:
        if self.flag_cancelado["valor"] or verificar_flag_cancelado(self.flag_cancelado):
            self._status(self.row, "🚩 Cancelado")
            return True
        return False

    def _status(self, row: int, mensagem: str):
        logger.debug(f"[DownloadWorker] STATUS ({row}): {mensagem}")
        self.logar_callback(mensagem)
        self.signals.status_linha.emit(row, mensagem)

    def _executar_etapa(self, mensagem, func):
        if self._checar_cancelamento(): return
        self._status(self.row, mensagem)
        func()
        if self._checar_cancelamento(): return

    def _baixar(self):
        self.caminho_video, self.caminho_thumb = baixar_video_e_thumbnail(
            self.link, self.caminho_saida, progresso_callback=self.logar_callback
        )

        if not self.caminho_video or not os.path.isfile(self.caminho_video):
            raise RuntimeError("❌ Falha no download do vídeo.")

        self._status(self.row, "📅 Download concluído.")

    def _reencodar(self):
        md5_antes = calcular_md5(self.caminho_video)
        self.logar_callback(f"🔍 MD5 antes do vídeo: {md5_antes}")

        self.caminho_video_final = reencodar_video(self.caminho_video)

        if not self.caminho_video_final:
            raise RuntimeError("❌ Reencodificação falhou.")

        md5_depois = calcular_md5(self.caminho_video_final)
        self.logar_callback(f"🧪 MD5 depois do vídeo: {md5_depois}")
        self._status(self.row, "✅ Vídeo reencodificado.")

        if self._checar_cancelamento(): return

        if self._thumb_valida():
            self._reencodar_thumb()

    def _thumb_valida(self) -> bool:
        return (
            self.caminho_thumb
            and os.path.exists(self.caminho_thumb)
            and self.caminho_thumb.lower().endswith((".jpg", ".jpeg", ".webp", ".png"))
        )

    def _reencodar_thumb(self):
        try:
            self._status(self.row, "🖼️ Reencodificando thumbnail...")

            md5_antes = calcular_md5(self.caminho_thumb)
            self.logar_callback(f"🔍 MD5 antes da imagem: {md5_antes}")

            reencodar_imagem(self.caminho_thumb)

            md5_depois = calcular_md5(self.caminho_thumb)
            self.logar_callback(f"🧪 MD5 depois da imagem: {md5_depois}")
            self._status(self.row, "✅ Thumbnail pronta.")
        except Exception as e:
            self.logar_callback(f"❌ Falha ao reencodar thumbnail: {e}")
            logger.exception("[DownloadWorker] Erro ao reencodar imagem")

    def _registrar_erro(self, erro_detalhado: str):
        try:
            log_path = os.path.join(self.caminho_saida, "erro_downloadworker.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[{self.link}]\n{erro_detalhado}\n{'='*80}\n")
        except Exception as e:
            logger.error(f"❌ Falha ao registrar erro no arquivo: {e}")

# worker_manager.py (Finalizado: sinais do TranscriberWorker corrigidos e idioma autodetectado)
import os
import queue
import threading
import time
import torch
from typing import NamedTuple, Callable
from functools import partial
from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool
from workers.download_worker import DownloadWorker
from workers.transcriber_worker import TranscriberWorker, TranscriberSignals
from utils.logger import get_logger
import psutil
import pynvml

logger = get_logger(__name__)

class TarefaInfo(NamedTuple):
    link: str
    pasta: str
    index: int
    logar_callback: Callable[[str], None]
    progresso_callback: Callable[[int, int], None]
    usar_transcricao: bool = True

class ResourceManager:
    def __init__(self):
        self.cpu_pool = QThreadPool()
        self.gpu_pool = QThreadPool()
        
        # Configuração CPU
        cpu_count = psutil.cpu_count(logical=False)  # Apenas cores físicos
        self.cpu_pool.setMaxThreadCount(max(1, cpu_count - 1))  # Deixa 1 core livre
        
        # Configuração GPU
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            try:
                pynvml.nvmlInit()
                self.gpu_count = torch.cuda.device_count()
                # Limita workers GPU baseado na memória disponível
                self.gpu_pool.setMaxThreadCount(min(2, self.gpu_count))
            except Exception as e:
                logger.warning(f"⚠️ Erro ao inicializar NVML: {e}")
                self.has_gpu = False
        else:
            self.gpu_pool.setMaxThreadCount(0)
        
        logger.info(f"🖥️ CPU Workers: {self.cpu_pool.maxThreadCount()}")
        logger.info(f"🎮 GPU Available: {self.has_gpu} | Workers: {self.gpu_pool.maxThreadCount()}")

    def submit_cpu_task(self, worker):
        """Submete tarefa para processamento em CPU"""
        self.cpu_pool.start(worker)

    def submit_gpu_task(self, worker):
        """Submete tarefa para processamento em GPU"""
        if self.has_gpu:
            self.gpu_pool.start(worker)
        else:
            logger.warning("⚠️ GPU não disponível, usando CPU para transcrição")
            self.cpu_pool.start(worker)

class WorkerManager(QObject):
    progresso_global = pyqtSignal(int, int)
    status_global = pyqtSignal(str)
    status_linha_global = pyqtSignal(int, str)
    item_concluido = pyqtSignal(int)
    popup_erro = pyqtSignal(str)
    resultado_transcricao = pyqtSignal(dict)

    def __init__(self, max_threads=None):
        super().__init__()
        self.fila = queue.PriorityQueue()
        self.max_threads = max_threads or self._definir_threads_ideais()
        self.encerrar = False
        self.workers_ativos = set()
        self.lock = threading.Lock()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(self.max_threads)
        logger.debug(f"🥵 WorkerManager inicializado com {self.max_threads} threads disponíveis.")

    def _definir_threads_ideais(self):
        cpu_threads = os.cpu_count() or 4
        if torch.cuda.is_available():
            logger.info("🚀 CUDA disponível! Aproveitando GPU.")
            return min(cpu_threads, 8)
        return min(cpu_threads, 4)

    def iniciar(self):
        logger.debug("📱 Iniciando gerenciador de tarefas em thread daemon...")
        threading.Thread(target=self._gerenciar_fila, daemon=True).start()

    def _gerenciar_fila(self):
        while not self.encerrar:
            with self.lock:
                if len(self.workers_ativos) >= self.max_threads:
                    time.sleep(0.2)
                    continue
            try:
                _, tarefa = self.fila.get(timeout=0.2)
                self._iniciar_worker(tarefa)
            except queue.Empty:
                continue

    def _iniciar_worker(self, tarefa: TarefaInfo):
        worker = DownloadWorker(
            link=tarefa.link,
            caminho_saida=tarefa.pasta,
            row=tarefa.index,
            logar_callback=tarefa.logar_callback,
        )

        logger.debug(f"🚀 Iniciando worker para linha {tarefa.index}: {tarefa.link} | usar_transcricao={tarefa.usar_transcricao}")

        worker.signals.progresso.connect(self.progresso_global)
        worker.signals.status_linha.connect(self.status_linha_global)

        worker.signals.finalizado.connect(lambda dados: self._tarefa_concluida(tarefa.index, dados, tarefa.usar_transcricao))
        worker.signals.erro.connect(lambda args: self._tarefa_falhou(args.get("row", tarefa.index), args.get("erro", "Erro desconhecido")))

        with self.lock:
            self.workers_ativos.add(tarefa.index)

        self.thread_pool.start(worker)
        logger.debug(f"📥 Worker adicionado à thread_pool para linha {tarefa.index}")

    def adicionar_tarefa(self, link, pasta, index, logar_callback, progresso_callback, usar_transcricao=True):
        tarefa = TarefaInfo(link, pasta, index, logar_callback, progresso_callback, usar_transcricao)
        self.fila.put((0, tarefa))
        logger.debug(f"📅 Tarefa adicionada na fila: linha {index} | {link}")

    def _tarefa_concluida(self, index, dados, usar_transcricao):
        logger.debug(f"[WorkerManager] 🔄 _tarefa_concluida chamada para linha {index}")
        self._remover_worker(index)

        self.item_concluido.emit(index)
        self.status_global.emit(f"✅ Linha {index + 1} concluída.")

        if usar_transcricao:
            caminho_video = dados.get("video_path")
            if caminho_video and os.path.exists(caminho_video):
                transcriber = TranscriberWorker(
                    caminho_video=caminho_video,
                    idioma=None  # autodetectar idioma
                )
                transcriber.signals.resultado.connect(self.resultado_transcricao.emit)
                transcriber.signals.status.connect(lambda msg: self.status_linha_global.emit(index, msg))
                transcriber.signals.erro.connect(lambda erro: self._tarefa_falhou(index, erro.get("erro", "Erro na transcrição")))
                self.thread_pool.start(transcriber)
            else:
                msg = f"❌ Caminho do vídeo inválido (linha {index}): {caminho_video}"
                logger.error(msg)
                self.popup_erro.emit(msg)

        self.status_global.emit(f"⏳ {self.fila.qsize()} tarefas restantes.")

    def _tarefa_falhou(self, index, mensagem):
        logger.error(f"❌ Falha no worker da linha {index}: {mensagem}")
        self.status_linha_global.emit(index, "❌ Falhou")
        self.status_global.emit(f"❌ Falha na linha {index + 1}: {mensagem}")
        self.popup_erro.emit(mensagem)
        self._remover_worker(index)

    def _remover_worker(self, index):
        with self.lock:
            if index in self.workers_ativos:
                self.workers_ativos.remove(index)
        logger.debug(f"🗑️ Worker da linha {index} removido")

    def cancelar_todas(self):
        self.encerrar = True
        logger.warning("🚩 Cancelando todas as tarefas e limpando fila...")
        with self.lock:
            self.workers_ativos.clear()
        with self.fila.mutex:
            self.fila.queue.clear()

    def tem_tarefas_ativas(self) -> bool:
        with self.lock:
            return bool(self.workers_ativos or not self.fila.empty())

# Singleton para gerenciamento de recursos
resource_manager = ResourceManager()

worker_manager = WorkerManager()

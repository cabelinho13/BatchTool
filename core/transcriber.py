import os
import subprocess
import sys
import torch
import tempfile
from typing import Optional, Callable, Union, List, Dict, Set, Tuple, Any
from faster_whisper import WhisperModel
from utils.utils import verificar_formato_arquivo, salvar_txt, gerar_copia_temporaria, salvar_transcricao
from utils.helpers import limpar_nome_arquivo
from utils.logger import get_logger
from threading import Lock, Thread
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import numpy as np
from dataclasses import dataclass, field, asdict
from threading import Event
import time
from datetime import datetime, timedelta
import psutil
from pathlib import Path
import gc
import signal
import json
from contextlib import contextmanager
import shutil
import ffmpeg
import threading

# Initialize logger first
logger = get_logger(__name__)

# Make webrtcvad optional
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    logger.warning("⚠️ webrtcvad não disponível. VAD otimizado desativado.")

@dataclass
class TranscriptionTask:
    video_path: str
    language: Optional[str]
    callback: Optional[Callable[[int], None]]
    cancel_event: Optional[Event]
    task_id: int

@dataclass
class TranscriptionMetrics:
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = None
    processing_time: float = 0.0
    audio_duration: float = 0.0
    model_inference_time: float = 0.0
    audio_conversion_time: float = 0.0
    gpu_memory_used: float = 0.0
    device_used: str = ""
    success: bool = False
    error_message: str = ""

class MemoryOptimizer:
    def __init__(self):
        self.initial_memory = None
        self.initial_gpu_memory = None
        self._collect_initial_stats()
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def _collect_initial_stats(self):
        """Coleta estatísticas iniciais de memória"""
        try:
            self.initial_memory = psutil.Process().memory_info().rss
            if torch.cuda.is_available():
                self.initial_gpu_memory = torch.cuda.memory_allocated()
                # Configura para liberar cache imediatamente
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.8)  # Limita uso de memória GPU
        except Exception as e:
            logger.warning(f"⚠️ Erro ao coletar estatísticas iniciais: {e}")
    
    @contextmanager
    def optimize_context(self):
        """Context manager para otimização de memória durante operações críticas"""
        try:
            self._pre_operation_cleanup()
            yield
        finally:
            self._post_operation_cleanup()
    
    def _pre_operation_cleanup(self):
        """Limpeza antes de operações principais"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            logger.warning(f"⚠️ Erro na limpeza prévia: {e}")
    
    def optimize_for_inference(self, model):
        """Otimiza o modelo para inferência"""
        try:
            with self.optimize_context():
                if hasattr(model, 'eval'):
                    model.eval()
                
                if torch.cuda.is_available():
                    # Otimizações específicas para GPU
                    if hasattr(model, 'half'):
                        model.half()  # Converte para FP16
                    
                    # Configura para computação determinística
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = True
                    
                    # Otimiza alocação de memória
                    torch.cuda.set_per_process_memory_fraction(0.8)
        except Exception as e:
            logger.error(f"❌ Erro ao otimizar modelo: {e}")
            raise
                
    def _post_operation_cleanup(self):
        """Limpeza após operações"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Registra pico de memória
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                logger.debug(f"📊 Pico de memória GPU: {peak_memory:.2f}MB")
        except Exception as e:
            logger.warning(f"⚠️ Erro na limpeza pós-operação: {e}")

    def get_memory_stats(self) -> Dict:
        """Retorna estatísticas de uso de memória"""
        stats = {
            'system_memory': {
                'initial': self.initial_memory / 1024**2,
                'current': psutil.Process().memory_info().rss / 1024**2
            }
        }
        
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'initial': self.initial_gpu_memory / 1024**2,
                'current': torch.cuda.memory_allocated() / 1024**2,
                'peak': torch.cuda.max_memory_allocated() / 1024**2,
                'cached': torch.cuda.memory_reserved() / 1024**2
            }

        return stats

class ResourceMonitor:
    @staticmethod
    def get_system_resources() -> Dict:
        """Retorna informações sobre recursos do sistema"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        gpu_info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    gpu_info[i] = {
                        'name': torch.cuda.get_device_name(i),
                        'memory_allocated': torch.cuda.memory_allocated(i) / 1024**2,
                        'memory_reserved': torch.cuda.memory_reserved(i) / 1024**2,
                        'max_memory_allocated': torch.cuda.max_memory_allocated(i) / 1024**2
                    }
                except Exception as e:
                    logger.warning(f"Erro ao obter informações da GPU {i}: {e}")

        return {
            'memory': {
                'total': memory.total / 1024**2,
                'available': memory.available / 1024**2,
                'percent': memory.percent
            },
            'cpu': {
                'percent': cpu_percent,
                'cores': psutil.cpu_count()
            },
            'gpu': gpu_info
        }

class CacheManager:
    def __init__(self, cache_dir: str = "cache", max_cache_size_mb: int = 1000):
        self.cache_dir = cache_dir
        self.max_cache_size_mb = max_cache_size_mb
        self.backup_dir = os.path.join(cache_dir, "backup")
        self._lock = threading.Lock()
        self._initialize_cache()

    def _initialize_cache(self):
        """Inicializa o diretório de cache"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.backup_dir, exist_ok=True)
            self.force_cleanup_cache()
        except Exception as e:
            logger.error(f"Erro ao inicializar cache: {e}")

    def force_cleanup_cache(self):
        """Limpa forçadamente todo o cache"""
        try:
            with self._lock:
                # Primeiro, faz backup dos arquivos importantes
                self._backup_important_files()
                
                # Limpa o cache
                for file in os.scandir(self.cache_dir):
                    if file.is_file() and not file.name.startswith('.'):
                        try:
                            os.remove(file.path)
                            logger.debug(f"Arquivo removido do cache: {file.path}")
                        except Exception as e:
                            logger.warning(f"Erro ao remover arquivo do cache: {e}")
                
                # Limpa memória
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                logger.info("✅ Cache limpo com sucesso")
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {e}")

    def _backup_important_files(self):
        """Faz backup dos arquivos importantes antes da limpeza"""
        try:
            important_extensions = {'.txt', '.json', '.log'}
            for file in os.scandir(self.cache_dir):
                if file.is_file() and any(file.name.endswith(ext) for ext in important_extensions):
                    backup_path = os.path.join(self.backup_dir, file.name)
                    shutil.copy2(file.path, backup_path)
                    logger.debug(f"Arquivo backup criado: {backup_path}")
        except Exception as e:
            logger.error(f"Erro ao criar backup: {e}")

    def _cleanup_cache(self):
        """Remove arquivos antigos do cache se necessário"""
        try:
            with self._lock:
                total_size = 0
                files = []
                
                # Coleta informações dos arquivos
                for file in os.scandir(self.cache_dir):
                    if file.is_file() and not file.name.startswith('.'):
                        size = file.stat().st_size / (1024 * 1024)  # Tamanho em MB
                        files.append((file.path, size, file.stat().st_mtime))
                        total_size += size
                
                # Remove arquivos mais antigos se necessário
                if total_size > self.max_cache_size_mb:
                    files.sort(key=lambda x: x[2])  # Ordena por data de modificação
                    for file_path, size, _ in files:
                        try:
                            # Verifica integridade do arquivo antes de remover
                            if self._verify_file_integrity(file_path):
                                os.remove(file_path)
                                total_size -= size
                                logger.debug(f"Arquivo removido do cache: {file_path}")
                                if total_size <= self.max_cache_size_mb:
                                    break
                        except Exception as e:
                            logger.warning(f"Erro ao remover arquivo do cache: {e}")
                            
                # Limpa memória após operação
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                            
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {e}")

    def _verify_file_integrity(self, file_path: str) -> bool:
        """Verifica a integridade do arquivo antes de removê-lo"""
        try:
            if not os.path.exists(file_path):
                return False
                
            # Verifica se o arquivo pode ser lido
            with open(file_path, 'rb') as f:
                f.read(1)
                
            # Verifica tamanho
            if os.path.getsize(file_path) == 0:
                return False
                
            return True
        except Exception:
            return False

    def _validar_audio(self, audio_path: str) -> bool:
        """Valida um arquivo de áudio usando ffmpeg"""
        try:
            # Verifica se o arquivo existe e tem tamanho
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                return False
            
            # Usa ffmpeg para validar o arquivo
            try:
                probe = ffmpeg.probe(audio_path)
                audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
                
                if not audio_stream:
                    return False
                
                # Verifica taxa de amostragem
                sample_rate = int(audio_stream.get('sample_rate', 0))
                if sample_rate < 8000 or sample_rate > 48000:
                    return False
                
                # Verifica número de canais
                channels = int(audio_stream.get('channels', 0))
                if channels not in [1, 2]:
                    return False
                
                # Verifica duração
                duration = float(audio_stream.get('duration', 0))
                if duration <= 0:
                    return False
                
                return True
            except ffmpeg.Error:
                return False
        except Exception as e:
            logger.error(f"Erro ao validar áudio: {e}")
            return False

    def get_cached_audio(self, video_path: str) -> Optional[str]:
        """Retorna o caminho do áudio em cache se existir"""
        try:
            with self._lock:
                video_hash = limpar_nome_arquivo(video_path)
                cached_path = os.path.join(self.cache_dir, f"{video_hash}.wav")
                
                if os.path.exists(cached_path):
                    if self._validar_audio(cached_path):
                        return cached_path
                    else:
                        try:
                            os.remove(cached_path)
                            logger.debug(f"Arquivo de cache inválido removido: {cached_path}")
                        except Exception as e:
                            logger.warning(f"Erro ao remover arquivo de cache inválido: {e}")
                return None
                    
        except Exception as e:
            logger.error(f"Erro ao verificar cache: {e}")
            return None

    def cache_audio(self, video_path: str, audio_path: str) -> str:
        """Adiciona um arquivo de áudio ao cache"""
        try:
            with self._lock:
                video_hash = limpar_nome_arquivo(video_path)
                cached_path = os.path.join(self.cache_dir, f"{video_hash}.wav")
                
                # Verifica se o arquivo de origem é válido
                if not self._validar_audio(audio_path):
                    raise ValueError("Arquivo de áudio inválido")
                
                # Copia o arquivo para o cache
                shutil.copy2(audio_path, cached_path)
                
                # Valida o arquivo copiado
                if not self._validar_audio(cached_path):
                    os.remove(cached_path)
                    raise ValueError("Falha na validação do arquivo em cache")
                
                return cached_path
                
        except Exception as e:
            logger.error(f"Erro ao adicionar ao cache: {e}")
            if os.path.exists(cached_path):
                try:
                    os.remove(cached_path)
                except:
                    pass
            raise

class Transcriber:
    def __init__(self, modelo_whisper="base", device_override: Optional[str] = None,
                 max_retries: int = 3, retry_delay: int = 5):
        self.modelo_whisper = modelo_whisper
        self.device = self._get_device(device_override)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model = None
        self.metrics = []
        self._metrics_lock = threading.Lock()
        self.memory_optimizer = MemoryOptimizer()
        self.resource_monitor = ResourceMonitor()
        self.cache_manager = CacheManager()
        
        # Tenta inicializar VAD
        self.vad = None
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(3)  # Modo mais agressivo
            logger.info("✅ WebRTCVAD inicializado com sucesso")
        except ImportError:
            logger.warning("⚠️ webrtcvad não disponível. VAD otimizado desativado.")
        
        self._load_model()
        
        # Inicializa monitores
        self.performance_monitor = PerformanceMonitor()

    def _get_device(self, device_override: Optional[str] = None) -> str:
        """Determina o dispositivo a ser usado"""
        if device_override:
            if device_override.startswith("cuda:"):
                # Converte cuda:0 para cuda
                return "cuda"
            return device_override
        
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Carrega o modelo Whisper"""
        try:
            logger.info(f"Carregando modelo Whisper {self.modelo_whisper} em {self.device}")
            
            # Verifica se o dispositivo está disponível
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("⚠️ GPU solicitada mas não disponível, usando CPU")
                self.device = "cpu"
            
            # Configura o tipo de computação baseado no dispositivo
            compute_type = "float16" if self.device == "cuda" else "int8"
            
            # Carrega o modelo com tratamento de erro específico
            try:
                self.model = WhisperModel(
                    self.modelo_whisper,
                    device=self.device,
                    compute_type=compute_type,
                    cpu_threads=4 if self.device == "cpu" else 0,
                    num_workers=2,
                    download_root="models"
                )
                logger.info(f"✅ Modelo carregado com sucesso em {self.device}")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar modelo: {e}")
                if "CUDA" in str(e) or "GPU" in str(e):
                    logger.warning("⚠️ Tentando carregar em CPU devido a erro CUDA")
                    self.device = "cpu"
                    compute_type = "int8"
                    self.model = WhisperModel(
                        self.modelo_whisper,
                        device="cpu",
                        compute_type=compute_type,
                        cpu_threads=4,
                        num_workers=2,
                        download_root="models"
                    )
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"❌ Erro fatal ao carregar modelo: {e}")
            raise

    def converter_audio_wav(self, audio_path: str) -> Optional[str]:
        """Converte um arquivo de áudio para formato WAV usando ffmpeg"""
        try:
            # Verifica se o arquivo já está em formato WAV
            if audio_path.lower().endswith('.wav'):
                return audio_path
            
            # Gera um nome de arquivo temporário
            temp_wav = os.path.join(tempfile.gettempdir(), f"temp_{os.path.basename(audio_path)}.wav")
            
            # Comando ffmpeg para conversão
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y', temp_wav
            ]
            
            # Executa o comando
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Erro na conversão do áudio: {result.stderr}")
                return None
            
            return temp_wav
            
        except Exception as e:
            logger.error(f"Erro ao converter áudio: {str(e)}")
            return None

    def _process_audio_with_vad(self, audio_path: str) -> np.ndarray:
        """Processa o áudio com VAD se disponível"""
        try:
            if self.vad is None:
                return self._load_audio_default(audio_path)
            
            # Carrega o áudio
            audio = self._load_audio_default(audio_path)
            if audio is None:
                return None
            
            # Converte para formato compatível com VAD
            audio_samples = (audio * 32768).astype(np.int16)
            frame_duration = 30  # ms
            samples_per_frame = int(16000 * frame_duration / 1000)
            
            # Processa frames com VAD
            frames = []
            for i in range(0, len(audio_samples), samples_per_frame):
                frame = audio_samples[i:i + samples_per_frame]
                if len(frame) == samples_per_frame:
                    if self.vad.is_speech(frame.tobytes(), 16000):
                        frames.append(frame)
            
            # Reconstrói o áudio apenas com frames de fala
            if frames:
                processed_audio = np.concatenate(frames).astype(np.float32) / 32768
                return processed_audio
            else:
                logger.warning("⚠️ Nenhum frame de fala detectado, usando áudio original")
                return audio
                
        except Exception as e:
            logger.error(f"❌ Erro no processamento VAD: {str(e)}")
            return self._load_audio_default(audio_path)

    def _load_audio_default(self, audio_path: str) -> Optional[np.ndarray]:
        """Carrega áudio usando ffmpeg otimizado"""
        try:
            # Configura ffmpeg para leitura otimizada
            stream = (
                ffmpeg
                .input(audio_path)
                .output(
                    'pipe:',
                    format='f32le',
                    acodec='pcm_f32le',
                    ac=1,
                    ar=16000,
                    loglevel='error',
                    threads=0  # Usa todos os threads disponíveis
                )
            )
            
            # Lê o áudio em chunks para melhor performance
            audio_chunks = []
            process = stream.run_async(pipe_stdout=True, pipe_stderr=True)
            
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                audio_chunks.append(np.frombuffer(chunk, np.float32))
            
            process.wait()
            
            if process.returncode != 0:
                logger.error(f"❌ Erro ffmpeg: {process.stderr.read().decode()}")
                return None
            
            # Concatena os chunks
            audio = np.concatenate(audio_chunks)
            
            # Normaliza o áudio
            if len(audio) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar áudio: {str(e)}")
            return None

    def transcrever_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcreve um arquivo de áudio usando o modelo Whisper"""
        try:
            # Verifica se o arquivo existe
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")
            
            # Converte o áudio para WAV se necessário
            audio_path = self.converter_audio_wav(audio_path)
            if audio_path is None:
                raise ValueError("Falha na conversão do áudio para WAV")
            
            # Processa o áudio com VAD se disponível
            audio = self._process_audio_with_vad(audio_path)
            if audio is None:
                raise ValueError("Falha ao processar o áudio")
            
            # Transcreve o áudio
            segments, info = self.model.transcribe(audio_path)
            
            # Monta o texto completo
            texto_completo = " ".join(segment.text for segment in segments)
            
            # Registra métricas
            with self._metrics_lock:
                self.metrics.append({
                    'timestamp': datetime.now(),
                    'duracao': info.duration,  # Duração em segundos
                    'tamanho_arquivo': os.path.getsize(audio_path),
                    'sucesso': True
                })
            
            return {
                'texto': texto_completo.strip(),
                'idioma': info.language,
                'duracao': info.duration,
                'sucesso': True
            }
            
        except Exception as e:
            error_category = self._categorize_error(str(e))
            logger.error(f"Erro na transcrição: {str(e)} (Categoria: {error_category})")
            
            with self._metrics_lock:
                self.metrics.append({
                    'timestamp': datetime.now(),
                    'erro': str(e),
                    'categoria_erro': error_category,
                    'sucesso': False
                })
            
            return {
                'erro': str(e),
                'categoria_erro': error_category,
                'sucesso': False
            }

    def liberar_gpu(self, gpu_id: Optional[int] = None):
        """Libera memória da GPU"""
        try:
            with self.memory_optimizer.optimize_context():
                if torch.cuda.is_available():
                    if gpu_id is not None:
                        torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    logger.debug("✅ Memória GPU liberada")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao liberar GPU: {e}")

    def _categorize_error(self, error_message: str) -> str:
        """Categoriza erros para melhor análise"""
        error_message = error_message.lower()
        
        categories = {
            'audio': ['audio', 'som', 'wav', 'conversão', 'ffmpeg'],
            'model': ['modelo', 'whisper', 'inferência', 'gpu', 'cuda'],
            'memory': ['memória', 'memory', 'out of memory', 'oom'],
            'file': ['arquivo', 'file', 'path', 'caminho', 'não encontrado'],
            'network': ['internet', 'network', 'conexão', 'download'],
            'permission': ['permissão', 'permission', 'acesso negado'],
            'format': ['formato', 'format', 'codec', 'incompatível'],
            'timeout': ['timeout', 'tempo limite', 'time limit'],
            'unknown': ['desconhecido', 'unknown']
        }
        
        for category, keywords in categories.items():
            if any(keyword in error_message for keyword in keywords):
                return category
        return 'unknown'

    def get_performance_stats(self) -> Dict:
        """Retorna estatísticas de desempenho das transcrições"""
        with self._metrics_lock:
            if not self.metrics:
                return {"message": "Nenhuma transcrição realizada ainda"}

            successful = [m for m in self.metrics if m['sucesso']]
            failed = [m for m in self.metrics if not m['sucesso']]

            stats = {
                "total_transcriptions": len(self.metrics),
                "successful_transcriptions": len(successful),
                "failed_transcriptions": len(failed),
                "success_rate": len(successful) / len(self.metrics) if self.metrics else 0,
                "average_processing_time": sum(m['duracao'] for m in successful) / len(successful) if successful else 0,
                "total_audio_processed": sum(m['duracao'] for m in successful) if successful else 0,
                "average_speed_ratio": sum(m['duracao'] / m['duracao'] for m in successful) if successful else 0,
                "error_analysis": self._analyze_common_errors(failed) if failed else {},
                "device_usage": {
                    "cpu": self.device == "cpu",
                    "gpu": self.device == "cuda",
                    "average_gpu_memory": sum(m['tamanho_arquivo'] for m in successful) / len(successful) if successful else 0
                }
            }

            return stats

    def _analyze_common_errors(self, failed_transcriptions: List[Dict]) -> Dict:
        """Analisa os erros mais comuns nas transcrições falhas"""
        error_categories = {}

        for metric in failed_transcriptions:
            category = self._categorize_error(metric['erro'])
            if category not in error_categories:
                error_categories[category] = {
                    "count": 0,
                    "examples": []
                }
            
            error_categories[category]["count"] += 1
            if len(error_categories[category]["examples"]) < 3:  # Limita a 3 exemplos
                error_categories[category]["examples"].append(metric['erro'])

        return error_categories

    def get_resource_usage(self) -> Dict:
        """Retorna informações sobre uso de recursos do sistema"""
        return self.resource_monitor.get_system_resources()

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.last_update = self.start_time
        self.processed_duration = 0
        self.total_audio_size = 0
        self.total_processing_time = 0
        self.gpu_memory_peaks = []
        self.cpu_usage = []
        self.memory_usage = []
        
    def update(self, audio_duration: float, processing_time: float):
        """Atualiza métricas de performance"""
        current_time = time.time()
        self.processed_duration += audio_duration
        self.total_processing_time += processing_time
        
        # Coleta métricas do sistema
        if torch.cuda.is_available():
            self.gpu_memory_peaks.append(torch.cuda.max_memory_allocated() / 1024**2)
        
        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(psutil.virtual_memory().percent)
        
        self.last_update = current_time
        
    def get_stats(self) -> Dict:
        """Retorna estatísticas de performance"""
        stats = {
            'total_duration': self.processed_duration,
            'total_processing_time': self.total_processing_time,
            'realtime_factor': self.total_processing_time / self.processed_duration if self.processed_duration > 0 else 0,
            'avg_cpu_usage': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0
        }
        
        if self.gpu_memory_peaks:
            stats.update({
                'max_gpu_memory': max(self.gpu_memory_peaks),
                'avg_gpu_memory': np.mean(self.gpu_memory_peaks)
            })
            
        return stats

def load_audio_file(file_path: str) -> np.ndarray:
    """Carrega um arquivo de áudio usando ffmpeg"""
    try:
        # Verifica se o arquivo existe e tem tamanho
        if not os.path.exists(file_path):
            logger.error("❌ Arquivo de áudio não encontrado")
            return None
            
        if os.path.getsize(file_path) == 0:
            logger.error("❌ Arquivo de áudio vazio")
            return None
            
        # Verifica formato do arquivo
        try:
            probe = ffmpeg.probe(file_path)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            if not audio_stream:
                logger.error("❌ Arquivo não contém stream de áudio")
                return None
                
            # Configura parâmetros baseados no stream
            sample_rate = int(audio_stream.get('sample_rate', 16000))
            channels = int(audio_stream.get('channels', 1))
            
            # Carrega áudio com ffmpeg
            out, _ = (
                ffmpeg
                .input(file_path)
                .output(
                    'pipe:',
                    format='f32le',
                    acodec='pcm_f32le',
                    ac=1,  # Força mono
                    ar=16000,  # Força 16kHz
                    loglevel='error'  # Reduz logs
                )
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Converte para numpy array
            audio = np.frombuffer(out, np.float32)
            
            # Valida o áudio carregado
            if len(audio) == 0:
                logger.error("❌ Áudio carregado está vazio")
                return None
                
            # Normaliza o áudio
            audio = audio / np.max(np.abs(audio))
            
            return audio
            
        except ffmpeg.Error as e:
            logger.error(f"❌ Erro ffmpeg ao carregar áudio: {e.stderr.decode()}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Erro ao carregar áudio: {e}")
        return None

class TranscriberPool:
    def __init__(self, model_name: str = "base", max_workers: int = 2):
        self.model_name = model_name
        self.max_workers = max_workers
        self.transcribers = []
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.workers = []
        self.stop_event = Event()
        self.resource_monitor = ResourceMonitor()
        self._initialize_transcribers()
        self._start_workers()

    def _initialize_transcribers(self):
        """Inicializa os transcribers com distribuição de GPU"""
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if gpu_count > 0:
            # Distribui workers entre GPUs disponíveis
            for i in range(self.max_workers):
                gpu_id = i % gpu_count
                device = f"cuda:{gpu_id}"
                try:
                    transcriber = Transcriber(
                        modelo_whisper=self.model_name,
                        device_override=device
                    )
                    self.transcribers.append(transcriber)
                except Exception as e:
                    logger.error(f"Erro ao inicializar transcriber em {device}: {e}")
                    # Fallback para CPU
                    transcriber = Transcriber(
                        modelo_whisper=self.model_name,
                        device_override="cpu"
                    )
                    self.transcribers.append(transcriber)
        else:
            # Modo CPU-only
            for _ in range(self.max_workers):
                transcriber = Transcriber(
                    modelo_whisper=self.model_name,
                    device_override="cpu"
                )
                self.transcribers.append(transcriber)

    def _worker(self, worker_id: int):
        """Função executada por cada worker"""
        transcriber = self.transcribers[worker_id]
        
        while not self.stop_event.is_set():
            try:
                # Tenta obter uma tarefa com timeout
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                
                # Monitora recursos antes do processamento
                resources_before = self.resource_monitor.get_system_resources()
                
                # Processa a tarefa
                try:
                    result = transcriber.transcrever_audio(task.video_path)
                    
                    # Monitora recursos após processamento
                    resources_after = self.resource_monitor.get_system_resources()
                    
                    # Adiciona informações de recursos ao resultado
                    result['recursos'] = {
                        'antes': resources_before,
                        'depois': resources_after,
                        'worker_id': worker_id,
                        'device': transcriber.device
                    }
                    
                except Exception as e:
                    logger.error(f"Erro no worker {worker_id}: {e}")
                    result = {
                        'erro': str(e),
                        'texto': '',
                        'progresso': 0
                    }
                
                # Adiciona o resultado à fila
                self.result_queue.put((task.task_id, result))
                
                # Limpa recursos após processamento
                transcriber.liberar_gpu()
                gc.collect()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Erro fatal no worker {worker_id}: {e}")
                if 'task' in locals():
                    self.result_queue.put((task.task_id, {
                        'erro': f"Erro fatal: {str(e)}",
                        'texto': '',
                        'progresso': 0
                    }))

    def _start_workers(self):
        """Inicia os workers"""
        for i in range(self.max_workers):
            worker = Thread(
                target=self._worker,
                args=(i,),
                daemon=True,
                name=f"TranscriberWorker-{i}"
            )
            worker.start()
            self.workers.append(worker)

    def process_videos(self, video_paths: List[str], batch_size: int = None) -> List[dict]:
        """Processa múltiplos vídeos em paralelo com suporte a batch"""
        if batch_size is None:
            batch_size = len(video_paths)
        
        all_results = []
        for i in range(0, len(video_paths), batch_size):
            batch = video_paths[i:i + batch_size]
            results = self._process_batch(batch)
            all_results.extend(results)
        
        return all_results

    def _process_batch(self, video_paths: List[str]) -> List[dict]:
        """Processa um batch de vídeos"""
        results = [None] * len(video_paths)
        tasks = []
        
        # Cria as tarefas
        for i, video_path in enumerate(video_paths):
            task = TranscriptionTask(
                video_path=video_path,
                language=None,
                callback=None,
                cancel_event=None,
                task_id=i
            )
            tasks.append(task)
            self.task_queue.put(task)
        
        # Aguarda resultados com timeout adaptativo
        completed = 0
        timeout_base = 30  # timeout base por tarefa em segundos
        while completed < len(tasks):
            try:
                # Calcula timeout baseado no número de tarefas restantes
                timeout = timeout_base * (len(tasks) - completed)
                task_id, result = self.result_queue.get(timeout=timeout)
                results[task_id] = result
                completed += 1
            except Empty:
                logger.warning(f"Timeout aguardando resultados após {completed}/{len(tasks)} completados")
                break
        
        return results

    def shutdown(self):
        """Encerra o pool de workers"""
        logger.info("Iniciando shutdown do TranscriberPool")
        self.stop_event.set()
        
        # Aguarda workers terminarem
        for i, worker in enumerate(self.workers):
            try:
                worker.join(timeout=5)
                if worker.is_alive():
                    logger.warning(f"Worker {i} não encerrou no timeout")
            except Exception as e:
                logger.error(f"Erro ao encerrar worker {i}: {e}")
        
        # Limpa recursos
        for transcriber in self.transcribers:
            try:
                transcriber.liberar_gpu()
            except Exception as e:
                logger.error(f"Erro ao liberar recursos: {e}")
        
        # Força limpeza de memória
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("TranscriberPool encerrado")

    def get_pool_stats(self) -> Dict:
        """Retorna estatísticas do pool"""
        stats = {
            'workers': len(self.workers),
            'workers_alive': sum(1 for w in self.workers if w.is_alive()),
            'queue_size': self.task_queue.qsize(),
            'devices': [t.device for t in self.transcribers],
            'resources': self.resource_monitor.get_system_resources()
        }
        return stats

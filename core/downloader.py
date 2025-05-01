import os
import yt_dlp
import time
import errno
from glob import glob
from typing import Optional, Tuple, Callable, Dict, Any
from utils.utils import normalizar_nome_arquivo, obter_cookies_automaticamente
from utils.logger import get_logger
import browser_cookie3
import tempfile
import concurrent.futures
from threading import Lock

logger = get_logger(__name__)

COOKIES_PATH = os.path.join("config", "cookies3.txt")


def is_valid_netscape_cookie_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line.startswith("# Netscape HTTP Cookie File"):
                return False
            second_line = f.readline()
            if "\t" not in second_line:
                logger.warning("⚠️ cookies.txt sem separadores TAB esperados.")
                return False
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao validar formato de cookies: {e}")
        return False


def notificar_usuario(msg: str, callback: Optional[Callable[[str], None]] = None) -> None:
    logger.info(msg)
    print(msg)
    if callback:
        callback(msg)


def tentar_download(link: str, ydl_opts: dict) -> dict:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=True)
        logger.debug(f"🎯 yt-dlp salvou como: {info.get('_filename')}")
        return info


def capturar_cookies_automaticamente(cookies_path: str, notificar: Callable[[str], None]) -> Optional[object]:
    notificar("🌐 Tentando extrair cookies automaticamente via navegador...")

    try:
        os.makedirs(os.path.dirname(cookies_path), exist_ok=True)
    except Exception:
        pass

    cookies = obter_cookies_automaticamente()

    if isinstance(cookies, str) and cookies.endswith(".txt"):
        with open(cookies, "r", encoding="utf-8") as fsrc, open(cookies_path, "w", encoding="utf-8") as fdst:
            fdst.write(fsrc.read())
        logger.info("✅ cookies.txt copiado para uso no yt-dlp.")
        return cookies_path

    elif hasattr(cookies, "__iter__"):
        logger.info("✅ Cookies capturados automaticamente (objeto).")
        return cookies

    else:
        raise RuntimeError("❌ Falha ao capturar cookies automaticamente ou via fallback.")


def baixar_video_e_thumbnail(
    link: str,
    pasta_destino: str,
    progresso_callback: Optional[Callable[[str], None]] = None,
    cookies_path: str = COOKIES_PATH
) -> Tuple[str, Optional[str]]:

    def notificar(msg: str):
        notificar_usuario(msg, progresso_callback)

    logger.debug(f"📅 Iniciando download do link: {link}")
    notificar(f"📅 Iniciando download do link: {link}")

    ydl_opts = {
        "outtmpl": os.path.join(pasta_destino, "%(title)s_%(upload_date)s.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "writethumbnail": True,
        "writeautomaticsub": False,
        "quiet": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
        "cachedir": False,
        "check_certificate": False,
        "nooverwrites": True,
        "cookiesfrombrowser": ("firefox",),
    }

    info = None

    try:
        notificar("🔐 Tentando download autenticado via Firefox...")
        info = tentar_download(link, ydl_opts)
        notificar("✅ Download autenticado com sucesso.")
    except Exception as e:
        logger.error(f"❌ Falha ao baixar vídeo com cookies do navegador: {e}")

        # ✅ Verifica se arquivos .part estão travados na pasta
        parte_em_uso = [f for f in os.listdir(pasta_destino) if f.endswith(".part")]
        if parte_em_uso:
            notificar("🕒 .part detectado, aguardando liberação para renomear...")
            logger.debug(f"⏳ Arquivos .part detectados: {parte_em_uso}")
            time.sleep(3)

        raise RuntimeError(f"❌ Falha ao baixar com cookies automáticos do Firefox: {e}")

    if not info:
        raise RuntimeError("❌ yt-dlp não retornou informações sobre o vídeo.")

    caminho_video = info.get("_filename")
    if not caminho_video or not os.path.exists(caminho_video):
        logger.warning("⚠️ _filename ausente ou arquivo não localizado. Buscando vídeo .mp4 na pasta...")

        mp4s = sorted(
            glob(os.path.join(pasta_destino, "*.mp4")),
            key=os.path.getmtime,
            reverse=True
        )
        if mp4s:
            caminho_video = mp4s[0]
            logger.debug(f"✅ Vídeo mais recente detectado: {caminho_video}")
        else:
            arquivos = os.listdir(pasta_destino)
            logger.error(f"❌ Nenhum .mp4 encontrado. Arquivos na pasta: {arquivos}")
            raise RuntimeError("❌ Falha no download do vídeo. Nenhum .mp4 localizado.")

    caminho_video = normalizar_nome_arquivo(caminho_video)

    caminho_thumb = None
    for ext in [".jpg", ".webp", ".png"]:
        tentativa = os.path.splitext(caminho_video)[0] + ext
        if os.path.exists(tentativa):
            caminho_thumb = tentativa
            break

    return caminho_video, caminho_thumb


class VideoDownloader:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._cookie_lock = Lock()
        self._download_lock = Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._setup_yt_dlp_options()

    def _setup_yt_dlp_options(self):
        """Configura as opções otimizadas do yt-dlp"""
        self.yt_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': '%(title)s.%(ext)s',
            'noplaylist': True,
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'logtostderr': False,
            'quiet': True,
            'no_warnings': True,
            'default_search': 'auto',
            'source_address': '0.0.0.0',
            'fragment_retries': 10,
            'retries': 10,
            'file_access_retries': 5,
            'buffersize': 1024*1024,  # 1MB buffer
            'http_chunk_size': 10485760,  # 10MB chunks
        }

    def download_video(self, url: str, output_path: str) -> Dict[str, Any]:
        """Download de vídeo com suporte a autenticação e otimizações"""
        try:
            logger.info(f"📅 Iniciando download do link: {url}")
            
            with self._cookie_lock:
                cookies = self._get_browser_cookies()
            
            if cookies:
                self.yt_opts['cookiefile'] = cookies
                logger.info("🔐 Usando cookies do navegador para autenticação")
            
            with self._download_lock:
                with yt_dlp.YoutubeDL(self.yt_opts) as ydl:
                    try:
                        # Extrai informações primeiro
                        info = ydl.extract_info(url, download=False)
                        video_title = info.get('title', 'video')
                        
                        # Configura o caminho de saída
                        output_file = os.path.join(output_path, f"{video_title}.mp4")
                        self.yt_opts['outtmpl'] = output_file
                        
                        # Faz o download com as opções otimizadas
                        ydl.download([url])
                        
                        return {
                            'success': True,
                            'path': output_file,
                            'title': video_title
                        }
                        
                    except Exception as e:
                        logger.error(f"❌ Erro no download: {str(e)}")
                        return {
                            'success': False,
                            'error': str(e)
                        }
                        
        except Exception as e:
            logger.error(f"❌ Erro fatal no download: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _get_browser_cookies(self) -> Optional[str]:
        """Obtém cookies do navegador de forma segura"""
        try:
            temp_cookie_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
            
            try:
                firefox_cookies = browser_cookie3.firefox()
                with open(temp_cookie_file.name, 'w') as f:
                    for cookie in firefox_cookies:
                        if 'youtube.com' in cookie.domain:
                            f.write(f"{cookie.domain}\tTRUE\t{cookie.path}\t"
                                  f"{'TRUE' if cookie.secure else 'FALSE'}\t{cookie.expires}\t"
                                  f"{cookie.name}\t{cookie.value}\n")
                return temp_cookie_file.name
            except:
                logger.warning("⚠️ Não foi possível obter cookies do Firefox")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erro ao obter cookies: {str(e)}")
            return None

    def download_multiple(self, urls: list, output_path: str) -> list:
        """Download paralelo de múltiplos vídeos"""
        futures = []
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for url in urls:
                future = executor.submit(self.download_video, url, output_path)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Erro em download paralelo: {str(e)}")
                    results.append({'success': False, 'error': str(e)})
        
        return results

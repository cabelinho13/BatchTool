import os
import re
import shutil
import datetime
import subprocess
from urllib.parse import urlparse, parse_qs
from functools import lru_cache
from typing import Optional, Tuple
from utils.logger import get_logger
import yt_dlp
import unicodedata

logger = get_logger(__name__)

# ─────────────── 📁 NOMES & PASTAS ───────────────

def gerar_nome_pasta_saida(link: str, slug: Optional[str] = None) -> str:
    video_id = parse_qs(urlparse(link).query).get("v", [""])[0] or "video"
    data = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"{data}_{slug or video_id}"

def normalizar_nome_arquivo(caminho: str) -> str:
    try:
        pasta, nome = os.path.split(caminho)

        # 🔒 Limpeza reforçada
        nome = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
        nome = re.sub(r"[^\w\s.-]", "_", nome)
        nome = re.sub(r'\s+', '_', nome).strip('_')
        if not nome:
            nome = "arquivo"

        novo_caminho = os.path.join(pasta, nome)
        if novo_caminho != caminho and os.path.exists(caminho):
            base, ext = os.path.splitext(nome)
            contador = 1
            while os.path.exists(novo_caminho):
                novo_caminho = os.path.join(pasta, f"{base}_{contador}{ext}")
                contador += 1
            os.rename(caminho, novo_caminho)
            logger.debug(f"🖍️ Renomeado para: {novo_caminho}")

        return novo_caminho
    except Exception as e:
        logger.error(f"❌ Erro ao normalizar nome de arquivo: {e}")
        return caminho


def gerar_copia_temporaria(caminho_original: str) -> Optional[str]:
    try:
        import tempfile
        base = os.path.basename(normalizar_nome_arquivo(caminho_original))
        temp_dir = tempfile.gettempdir()
        caminho_temp = os.path.join(temp_dir, base)
        shutil.copyfile(caminho_original, caminho_temp)
        logger.debug(f"📁 Cópia temporária criada: {caminho_temp}")
        return caminho_temp
    except Exception as e:
        logger.error(f"❌ Erro ao criar cópia temporária: {e}")
        return None

# ─────────────── 🔍 VERIFICAÇÕES ───────────────

FORMATOS_VALIDOS = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wav']

def verificar_formato_arquivo(caminho_video: str) -> bool:
    try:
        if not os.path.exists(caminho_video):
            logger.error(f"❌ Arquivo não encontrado: {caminho_video}")
            return False
            
        _, extensao = os.path.splitext(caminho_video)
        extensao = extensao.lower()
        
        if extensao not in FORMATOS_VALIDOS:
            logger.error(f"❌ Formato inválido: {extensao}")
            return False
            
        # Verifica se o arquivo tem conteúdo
        if os.path.getsize(caminho_video) == 0:
            logger.error(f"❌ Arquivo vazio: {caminho_video}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao verificar formato do arquivo: {e}")
        return False

@lru_cache(maxsize=1)
def get_ffmpeg_path() -> Tuple[str, str]:
    local_ffmpeg = os.path.abspath("ffmpeg.exe")
    local_ffprobe = os.path.abspath("ffprobe.exe")

    if os.path.exists(local_ffmpeg) and os.path.exists(local_ffprobe):
        return local_ffmpeg, local_ffprobe

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if ffmpeg_path and ffprobe_path:
        return ffmpeg_path, ffprobe_path

    raise FileNotFoundError(
        f"❌ ffmpeg.exe e/ou ffprobe.exe não encontrados.\n"
        f"Tentativas locais: {local_ffmpeg}, {local_ffprobe}\n"
        f"Variáveis de ambiente também foram verificadas."
    )

def obter_duracao_em_segundos(caminho: str) -> float:
    try:
        _, ffprobe = get_ffmpeg_path()
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", caminho],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return float(result.stdout.decode().strip() or 0.0)
    except Exception as e:
        logger.warning(f"⚠️ Não foi possível obter a duração do vídeo: {e}")
        return 0.0

# ─────────────── 💾 SALVAR ARQUIVOS ───────────────

def salvar_txt(nome_arquivo: str, conteudo: str, pasta_saida: str) -> bool:
    try:
        os.makedirs(pasta_saida, exist_ok=True)
        caminho_txt = os.path.join(pasta_saida, nome_arquivo)
        
        # Formata o texto
        linhas = conteudo.split('.')
        texto_formatado = []
        for linha in linhas:
            linha = linha.strip()
            if linha:
                texto_formatado.append(f"{linha}.")
        
        with open(caminho_txt, "w", encoding="utf-8") as f:
            f.write("\n\n".join(texto_formatado))
            
        logger.info(f"📄 Arquivo salvo: {caminho_txt}")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao salvar {nome_arquivo}: {e}")
        return False

def salvar_transcricao(texto: str, pasta_saida: str) -> bool:
    return salvar_txt("transcricao.txt", texto, pasta_saida)

# ─────────────── 🌐 COOKIES VIA NAVEGADOR ───────────────

def obter_cookies_automaticamente():
    logger.info("🌐 Obtendo cookies diretamente do navegador Firefox.")
    return {"cookiesfrombrowser": ("firefox",)}

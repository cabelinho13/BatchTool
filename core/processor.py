import os
import uuid
import ffmpeg
import shutil
import subprocess
import hashlib
from utils.logger import get_logger
from utils.utils import normalizar_nome_arquivo, get_ffmpeg_path
from typing import Optional

logger = get_logger(__name__)

def get_application_path() -> str:
    import sys
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def localizar_exiftool() -> str:
    caminho_global = shutil.which("exiftool")
    if caminho_global:
        logger.info(f"🛠️ exiftool global detectado: {caminho_global}")
        return caminho_global

    caminho_local = os.path.join(get_application_path(), "exiftool.exe")
    if os.path.exists(caminho_local):
        logger.info(f"🛠️ exiftool local detectado: {caminho_local}")
        return caminho_local

    raise FileNotFoundError("❌ exiftool não encontrado.")

def remover_arquivo_com_log(path: str) -> None:
    try:
        os.remove(path)
        logger.debug(f"🗑️ Arquivo removido: {path}")
    except Exception as e:
        logger.warning(f"⚠️ Falha ao remover {path}: {e}")

def calcular_md5(path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def reencodar_video(caminho_video: str) -> str:
    from ffmpeg import input as ffmpeg_input
    logger.info(f"🎞️ Reencodificando vídeo: {caminho_video}")
    
    # ✅ Sanitiza nome
    caminho_video = normalizar_nome_arquivo(caminho_video)

    if not os.path.exists(caminho_video):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_video}")

    nome_base, _ = os.path.splitext(caminho_video)
    novo_nome = f"{nome_base}_mod.mp4"

    ffmpeg_path, _ = get_ffmpeg_path()
    if not ffmpeg_path:
        raise RuntimeError("❌ ffmpeg não encontrado.")

    if os.path.exists(novo_nome):
        remover_arquivo_com_log(novo_nome)

    md5_antes = calcular_md5(caminho_video)
    logger.debug(f"🔍 MD5 antes do vídeo: {md5_antes}")
    logger.debug(f"⚙️ Comando ffmpeg para reencodificação: {caminho_video} -> {novo_nome}")

    try:
        (
            ffmpeg_input(caminho_video)
            .output(novo_nome, codec="copy", metadata=f"comment={uuid.uuid4()}")
            .overwrite_output()
            .run(cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else "sem stderr"
        logger.error(f"❌ ffmpeg erro:\n{stderr}")
        raise RuntimeError("❌ Falha na reencodificação do vídeo.")
    except Exception as e:
        logger.exception("❌ Erro inesperado durante reencodificação com ffmpeg.")
        raise

    if not os.path.exists(novo_nome):
        raise RuntimeError("❌ Arquivo final do vídeo não foi criado.")

    md5_depois = calcular_md5(novo_nome)
    logger.debug(f"🧪 MD5 depois do vídeo: {md5_depois}")

    if md5_antes == md5_depois:
        logger.warning("⚠️ MD5 não mudou! O vídeo pode não ter sido modificado.")
    else:
        logger.info("✅ O vídeo foi modificado com sucesso.")

    logger.info(f"✅ Reencodificação finalizada: {novo_nome}")
    remover_arquivo_com_log(caminho_video)
    return novo_nome

def reencodar_imagem(caminho_imagem: str) -> str:
    logger.info(f"🖼️ Regravando imagem com exiftool: {caminho_imagem}")
    exiftool_path = localizar_exiftool()

    saida = caminho_imagem.replace(".webp", ".jpg") if caminho_imagem.endswith(".webp") else caminho_imagem

    md5_antes = calcular_md5(saida)
    logger.debug(f"🔍 MD5 antes: {md5_antes}")

    comando = [exiftool_path, '-comment=Processed by VBT', '-overwrite_original', saida]

    try:
        resultado = subprocess.run(comando, capture_output=True, text=True)
        if resultado.returncode != 0:
            logger.error(f"❌ Erro no exiftool: {resultado.stderr}")
            raise RuntimeError("❌ Falha ao reescrever imagem com exiftool.")
    except Exception as e:
        logger.error(f"❌ Erro ao executar exiftool: {e}")
        raise

    md5_depois = calcular_md5(saida)
    logger.debug(f"🧪 MD5 depois: {md5_depois}")

    if md5_antes == md5_depois:
        logger.warning("⚠️ MD5 não mudou! A imagem pode não ter sido modificada corretamente.")
    else:
        logger.info("✅ A imagem foi modificada com sucesso (hash alterado).")

    return saida

def remover_metadados_simples(caminho_entrada: str) -> str:
    caminho_saida = os.path.splitext(caminho_entrada)[0] + "_clean.mp4"

    comando = [
        "ffmpeg",
        "-y",
        "-i", caminho_entrada,
        "-map_metadata", "-1",
        "-c:v", "copy",
        "-c:a", "copy",
        caminho_saida
    ]

    try:
        subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao remover metadados com ffmpeg: {e}")
        raise RuntimeError("❌ Falha ao remover metadados do vídeo.")

    return caminho_saida

import os
import sys
import shutil
import zipfile
import logging

from typing import Set, Dict
from datetime import datetime, timedelta
from logging import Logger

_loggers_cache: Dict[str, Logger] = {}


def criar_formatter() -> logging.Formatter:
    return logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s")


def configurar_logger(nome_arquivo: str = "video_batch_tool.log", level: int = logging.DEBUG) -> None:
    data_str = datetime.now().strftime("%Y-%m-%d")
    base_dir = os.environ.get("BATCHTOOL_LOG_DIR", "logs")
    pasta_logs = os.path.join(base_dir, data_str)

    try:
        os.makedirs(pasta_logs, exist_ok=True)
        caminho_log_geral = os.path.join(pasta_logs, nome_arquivo)
        caminho_log_erros = os.path.join(pasta_logs, "errors.log")

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        handler_paths: Set[str] = {
            os.path.abspath(getattr(h, 'baseFilename', ''))
            for h in root_logger.handlers
            if hasattr(h, 'baseFilename')
        }

        if os.path.abspath(caminho_log_geral) not in handler_paths:
            formatter = criar_formatter()

            file_handler = logging.FileHandler(caminho_log_geral, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            error_handler = logging.FileHandler(caminho_log_erros, mode='a', encoding='utf-8')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)

            # Console handler (stdout)
            if not any(isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) == sys.stdout
                       for h in root_logger.handlers):
                console_handler = logging.StreamHandler(sys.stdout)  # ou sys.stderr se preferir
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)

            logging.debug(f"📁 Logs salvos em: {os.path.abspath(caminho_log_geral)}")
            logging.debug(f"📁 Logs de erro em: {os.path.abspath(caminho_log_erros)}")

    except Exception as e:
        fallback_msg = f"[logger] ❌ Falha ao configurar log: {e}"
        logging.basicConfig(
            level=logging.ERROR,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.error(fallback_msg)


def get_logger(nome_modulo: str) -> Logger:
    if not nome_modulo:
        nome_modulo = __name__
    if nome_modulo not in _loggers_cache:
        _loggers_cache[nome_modulo] = logging.getLogger(nome_modulo)
    return _loggers_cache[nome_modulo]


def limpar_logs_antigos(dias: int = 7, compactar: bool = True) -> None:
    base_path = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(base_path):
        return

    limite_data = datetime.now() - timedelta(days=dias)

    for nome in os.listdir(base_path):
        caminho = os.path.join(base_path, nome)
        if not os.path.isdir(caminho):
            continue

        try:
            data_log = datetime.strptime(nome, "%Y-%m-%d")
            if data_log < limite_data:
                if compactar:
                    zip_path = f"{caminho}.zip"
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, _, files in os.walk(caminho):
                            for file in files:
                                abs_path = os.path.join(root, file)
                                arcname = os.path.relpath(abs_path, start=caminho)
                                zipf.write(abs_path, arcname)
                    shutil.rmtree(caminho)
                    if os.path.exists(zip_path):
                        logging.debug(f"📦 Log compactado: {zip_path} ({os.path.getsize(zip_path)} bytes)")
                else:
                    shutil.rmtree(caminho)
                    logging.debug(f"📁 Log excluído: {caminho}")
        except Exception as e:
            logging.error(f"❌ Erro ao limpar logs antigos: {e}")

import os
import re
import sys
import subprocess
from pathlib import Path
from PyQt6.QtGui import QTextCharFormat, QTextCursor, QColor
from PyQt6.QtCore import QTimer
from utils.logger import get_logger

logger = get_logger(__name__)

# 🎨 Cores para cada emoji utilizado nos logs
CORES_LOG = {
    "🧠": "#DDA0DD", "📄": "#98FB98", "🌍": "#00FF7F", "🕒": "#FFD700",
    "🎞️": "#FFB6C1", "⬇️": "#87CEFA", "📥": "#87CEFA", "🔗": "#00FA9A",
    "🖼️": "#1E90FF", "🛑": "#FF6347", "✅": "#4CAF50", "❌": "#FF4500",
    "🟡": "#FFD700", "🟢": "#00FF00", "🔴": "#FF0000"
}

def limpar_nome(nome: str) -> str:
    """Sanitiza o nome de um arquivo ou pasta para torná-lo compatível com o sistema de arquivos."""
    nome = nome.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    nome = re.sub(r'[<>:"/\\|?*]', '_', nome)
    nome = re.sub(r'\s+', ' ', nome)
    nome = nome.strip()
    return nome[:50] if nome else "sem_nome"

def limpar_nome_arquivo(nome: str) -> str:
    """Limpa o nome base de um arquivo (sem extensão), aplicando regras de normalização."""
    nome = Path(nome).stem
    nome = limpar_nome(nome)
    return re.sub(r'[^a-zA-Z0-9_\- ]+', '_', nome)

def logar(console_widget, mensagem: str) -> None:
    """
    Imprime mensagens no console visual com destaque por emojis + fallback no terminal.
    Seguro para chamadas em threads (via QTimer.singleShot).
    """
    print(mensagem)  # Terminal fallback

    if not console_widget:
        return

    def atualizar_console():
        cursor = console_widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()

        for emoji, cor in CORES_LOG.items():
            if emoji in mensagem:
                fmt.setForeground(QColor(cor))
                break
        else:
            fmt.setForeground(QColor("#AAAAAA"))  # Cor padrão

        cursor.insertText(mensagem + "\n", fmt)
        console_widget.setTextCursor(cursor)
        console_widget.ensureCursorVisible()

    QTimer.singleShot(0, atualizar_console)

def abrir_pasta(path: str):
    """Abre o diretório especificado no gerenciador de arquivos do sistema operacional."""
    try:
        if not os.path.exists(path):
            logger.warning(f"⚠️ Caminho não existe: {path}")
            return

        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        logger.exception(f"❌ Erro ao abrir pasta: {e}")

def verificar_flag_cancelado(flag_dict: dict) -> bool:
    """Verifica de forma segura e serializável se a tarefa foi marcada como cancelada."""
    return flag_dict.get("valor", False)

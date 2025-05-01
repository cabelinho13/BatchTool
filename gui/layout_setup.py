from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QLineEdit, QTableWidget, QProgressBar, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon  # ✅ Novo import

def setup_ui(self, widgets):
    self.setWindowTitle("Video Batch Tool - GUI")
    self.setWindowIcon(QIcon("assets/icon.ico"))  # ✅ Define o ícone da janela
    self.setMinimumSize(1000, 600)

    container = QWidget()
    layout_principal = QVBoxLayout(container)

    # Nome do canal
    label_canal = QLabel("📺 Nome do canal (para nome das pastas):")
    input_canal = QLineEdit()
    input_canal.setPlaceholderText("Ex: MEU CANAL DO YOUTUBE")

    # Links
    label_links = QLabel("📋 Cole abaixo os links dos vídeos do YouTube (um por linha):")
    text_links = QPlainTextEdit()
    text_links.setPlaceholderText("https://www.youtube.com/watch?v=...\nhttps://youtu.be/...")

    # Botões principais
    btn_processar = QPushButton("🚀 Processar vídeos")
    btn_cancelar = QPushButton("🛑 Parar")
    btn_limpar = QPushButton("🧹 Limpar")
    btn_reprocessar = QPushButton("🔁 Reprocessar falhas")

    layout_botoes = QHBoxLayout()
    layout_botoes.addWidget(btn_processar)
    layout_botoes.addWidget(btn_cancelar)
    layout_botoes.addWidget(btn_limpar)
    layout_botoes.addWidget(btn_reprocessar)

    # Tabela de status
    tabela_status = QTableWidget()
    tabela_status.setColumnCount(3)
    tabela_status.setHorizontalHeaderLabels(["#", "Link", "Status"])
    tabela_status.setColumnWidth(0, 40)
    tabela_status.setColumnWidth(1, 580)
    tabela_status.setColumnWidth(2, 220)
    tabela_status.verticalHeader().setVisible(False)
    tabela_status.horizontalHeader().setStretchLastSection(True)
    tabela_status.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
    tabela_status.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    tabela_status.setStyleSheet("""
        QTableWidget::item { padding: 6px; }
        QProgressBar { border: 1px solid #888; border-radius: 5px; min-height: 16px; }
        QProgressBar::chunk { background-color: #00cc66; }
    """)

    # Console de log
    console = QPlainTextEdit()
    console.setReadOnly(True)
    console.setStyleSheet("""
        QPlainTextEdit {
            background-color: #1e1e1e;
            color: #00FF7F;
            font-family: Consolas, monospace;
            font-size: 13px;
            border: 1px solid #444;
            border-radius: 5px;
        }
    """)

    # Abrir pasta saída
    btn_abrir_saida = QPushButton("📂 Abrir pasta de saída")

    # Label de status visual
    lbl_status = QLabel("")
    lbl_status.setStyleSheet("color: gray; font-weight: bold;")
    lbl_status.setAlignment(Qt.AlignmentFlag.AlignLeft)

    # Montagem do layout
    layout_principal.addWidget(label_canal)
    layout_principal.addWidget(input_canal)
    layout_principal.addWidget(label_links)
    layout_principal.addWidget(text_links)
    layout_principal.addLayout(layout_botoes)
    layout_principal.addWidget(tabela_status)
    layout_principal.addWidget(console)
    layout_principal.addWidget(btn_abrir_saida)
    layout_principal.addWidget(lbl_status)

    self.setCentralWidget(container)

    # Tema CSS geral
    self.setStyleSheet("""
        QWidget {
            background-color: #2b2b2b;
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
            font-size: 14px;
        }

        QPushButton {
            background-color: #3c3f41;
            color: #ffffff;
            border: 1px solid #5c5c5c;
            border-radius: 5px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #505357;
        }
        QPushButton:pressed {
            background-color: #2c2f33;
        }

        QLineEdit, QPlainTextEdit {
            background-color: #1e1e1e;
            color: #dcdcdc;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 4px;
        }

        QTableWidget {
            background-color: #1f1f1f;
            gridline-color: #444;
            border: none;
        }

        QHeaderView::section {
            background-color: #3c3f41;
            color: #dcdcdc;
            padding: 4px;
            border: 1px solid #444;
        }

        QProgressBar {
            border: 1px solid #888;
            border-radius: 5px;
            text-align: center;
            background-color: #2e2e2e;
        }

        QProgressBar::chunk {
            background-color: #00cc66;
        }
    """)

    widgets["input_canal"] = input_canal
    widgets["text_links"] = text_links
    widgets["btn_processar"] = btn_processar
    widgets["btn_cancelar"] = btn_cancelar
    widgets["btn_limpar"] = btn_limpar
    widgets["btn_reprocessar"] = btn_reprocessar
    widgets["btn_abrir_saida"] = btn_abrir_saida
    widgets["tabela_status"] = tabela_status
    widgets["console"] = console
    widgets["lbl_status"] = lbl_status

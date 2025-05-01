# splash_screen.py (Melhorias aplicadas)
from PyQt6.QtWidgets import QSplashScreen
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QPen, QPolygon
from PyQt6.QtCore import Qt, QTimer, QRectF, QPoint
from utils.logger import get_logger

logger = get_logger(__name__)

class SplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(520, 320)

        self.bg_color = QColor(20, 20, 20)
        self.text_color = QColor("#00FF7F")
        self.border_color = QColor("#444444")
        self.message = "🔄 Inicializando..."
        self.font = QFont("Consolas", 14, QFont.Weight.Bold)

        self._step = 0
        self._dots = ["", ".", "..", "..."]
        self._loading = True

        self.setPixmap(QPixmap(self.size()))

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animar)
        self._timer.start(300)  # animação mais suave

        self._barra_progressiva = 0
        self._barra_timer = QTimer(self)
        self._barra_timer.timeout.connect(self._simular_barra)
        self._barra_timer.start(20)  # simulação mais fluida

    def _animar(self):
        if self._loading:
            self._step = (self._step + 1) % len(self._dots)
            self.update()

    def _simular_barra(self):
        if self._loading:
            self._barra_progressiva += 1
            if self._barra_progressiva > 100:
                self._barra_timer.stop()
                self._barra_progressiva = 100
                logger.debug("✅ Barra de progresso finalizada.")
            self.update()

    def update_text(self, text: str):
        self.message = text
        self.update()

    def finish(self, widget):
        self._loading = False
        self._timer.stop()
        self._barra_timer.stop()
        if widget is None:
            logger.error("❌ Widget principal é None ao encerrar splash.")
        else:
            logger.debug("🧊 Splash screen finalizada.")
        super().finish(widget)

    def drawContents(self, painter: QPainter):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)

        # 🎨 LOGO CIRCULAR com símbolo play estilizado
        logo_center = self.rect().center()
        logo_radius = 40

        pen = QPen(self.text_color)
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(logo_center, logo_radius, logo_radius)

        # ▶ símbolo de play
        painter.setBrush(self.text_color)
        triangle = QPolygon([
            logo_center + QPoint(-10, -15),
            logo_center + QPoint(20, 0),
            logo_center + QPoint(-10, 15)
        ])
        painter.drawPolygon(triangle)

        # 🪧 Nome do app
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        painter.drawText(QRectF(0, 100, self.width(), 40), Qt.AlignmentFlag.AlignCenter, "Video Batch Tool")

        # 📜 Mensagem de progresso
        painter.setPen(self.text_color)
        painter.setFont(self.font)
        full_text = f"{self.message}{self._dots[self._step]}"
        painter.drawText(QRectF(0, 140, self.width(), 40), Qt.AlignmentFlag.AlignCenter, full_text)

        # 📊 Barra de progresso
        bar_width = self.width() - 80
        bar_x = 40
        bar_y = self.height() - 60
        bar_height = 18

        painter.setPen(self.border_color)
        painter.drawRect(bar_x, bar_y, bar_width, bar_height)

        fill_width = int((self._barra_progressiva / 100) * bar_width)
        if fill_width > 2:
            painter.fillRect(bar_x + 1, bar_y + 1, fill_width - 2, bar_height - 2, self.text_color)

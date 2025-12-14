import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QLabel, QComboBox, QLineEdit, QTextEdit, QTabWidget,
    QMessageBox, QGroupBox, QGridLayout, QFrame, QListWidget,
    QAbstractItemView, QSpinBox, QDoubleSpinBox, QFormLayout,
    QCheckBox, QGraphicsDropShadowEffect, QScrollArea, QSizePolicy,
    QStackedWidget, QProgressBar
)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QLinearGradient, QBrush, QIcon, QPainter
from algorithms import (
    KMeans, KMedoids, AGNES, DIANA, DBSCAN,
    SimpleImputer, MinMaxScaler, StandardScaler,
    KNN, GaussianNaiveBayes, train_test_split,
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score
)


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE - Midnight Aurora Theme
# ═══════════════════════════════════════════════════════════════════════════════
COLORS = {
    'bg_dark': '#0f0f1a',
    'bg_card': '#1a1a2e',
    'bg_elevated': '#252542',
    'bg_input': '#16162a',
    'border': '#2d2d4a',
    'border_light': '#3d3d5c',
    'text_primary': '#e8e8f0',
    'text_secondary': '#9090a8',
    'text_muted': '#606078',
    'accent_primary': '#7c3aed',      # Violet
    'accent_secondary': '#06b6d4',    # Cyan
    'accent_tertiary': '#f472b6',     # Pink
    'accent_success': '#10b981',      # Emerald
    'accent_warning': '#f59e0b',      # Amber
    'accent_danger': '#ef4444',       # Red
    'gradient_start': '#7c3aed',
    'gradient_end': '#06b6d4',
}


class GlowEffect(QGraphicsDropShadowEffect):
    """Custom glow effect for buttons and cards"""
    def __init__(self, color='#7c3aed', blur=20, parent=None):
        super().__init__(parent)
        self.setBlurRadius(blur)
        self.setColor(QColor(color))
        self.setOffset(0, 0)


class ModernCard(QFrame):
    """Reusable card component with subtle glow"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modernCard")
        self.setStyleSheet(f"""
            QFrame#modernCard {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 16px;
            }}
        """)


class AccentButton(QPushButton):
    """Primary action button with gradient and glow"""
    def __init__(self, text, icon_text="", parent=None):
        super().__init__(parent)
        display_text = f"{icon_text}  {text}" if icon_text else text
        self.setText(display_text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['accent_primary']}, stop:1 {COLORS['accent_secondary']});
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #8b5cf6, stop:1 #22d3ee);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6d28d9, stop:1 #0891b2);
            }}
            QPushButton:disabled {{
                background: {COLORS['bg_elevated']};
                color: {COLORS['text_muted']};
            }}
        """)


class SecondaryButton(QPushButton):
    """Secondary/ghost button"""
    def __init__(self, text, icon_text="", parent=None):
        super().__init__(parent)
        display_text = f"{icon_text}  {text}" if icon_text else text
        self.setText(display_text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_elevated']};
                border-color: {COLORS['border_light']};
                color: {COLORS['text_primary']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['bg_input']};
            }}
        """)


class IconButton(QPushButton):
    """Small icon button for toolbars"""
    def __init__(self, icon_text, tooltip="", parent=None):
        super().__init__(icon_text, parent)
        self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(40, 40)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_elevated']};
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_primary']};
                color: white;
                border-color: {COLORS['accent_primary']};
            }}
        """)


class NavButton(QPushButton):
    """Navigation tab button"""
    def __init__(self, text, icon_text="", parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(70)
        self.setMinimumWidth(90)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_muted']};
                border: none;
                border-radius: 12px;
                padding: 12px 8px;
                font-size: 14px;
                font-weight: 500;
                font-family: 'Segoe UI', sans-serif;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_elevated']};
                color: {COLORS['text_secondary']};
            }}
            QPushButton:checked {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(124, 58, 237, 0.2), stop:1 rgba(6, 182, 212, 0.2));
                color: {COLORS['accent_secondary']};
                border: 1px solid rgba(6, 182, 212, 0.3);
            }}
        """)


class StatCard(QFrame):
    """Mini stat display card"""
    def __init__(self, title, value, icon="📊", color=COLORS['accent_primary'], parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 8px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        
        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 15px; font-weight: 500;")
        header.addWidget(title_label)
        header.addStretch()
        
        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 24px; font-weight: 700;")
        
        layout.addLayout(header)
        layout.addWidget(self.value_label)
    
    def set_value(self, value):
        self.value_label.setText(str(value))


class DataAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.filtered_df = None
        self.current_page = 0
        self.setup_theme()
        self.init_ui()

    def setup_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['bg_dark']};
            }}
            QWidget {{
                font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
                color: {COLORS['text_primary']};
            }}
            QLabel {{
                color: {COLORS['text_secondary']};
                font-size: 13px;
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {COLORS['bg_input']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 13px;
                color: {COLORS['text_primary']};
                selection-background-color: {COLORS['accent_primary']};
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {COLORS['accent_primary']};
            }}
            QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
                border-color: {COLORS['border_light']};
            }}
            QComboBox {{
                background-color: {COLORS['bg_input']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 13px;
                color: {COLORS['text_primary']};
                min-height: 20px;
            }}
            QComboBox:hover {{
                border-color: {COLORS['border_light']};
            }}
            QComboBox:focus {{
                border-color: {COLORS['accent_primary']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {COLORS['text_secondary']};
                margin-right: 10px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_elevated']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                selection-background-color: {COLORS['accent_primary']};
                outline: none;
            }}
            QTableWidget {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                gridline-color: {COLORS['border']};
                selection-background-color: rgba(124, 58, 237, 0.3);
                selection-color: {COLORS['text_primary']};
                alternate-background-color: {COLORS['bg_elevated']};
            }}
            QTableWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {COLORS['border']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_elevated']};
                color: {COLORS['text_secondary']};
                padding: 12px 8px;
                border: none;
                border-bottom: 2px solid {COLORS['accent_primary']};
                font-weight: 600;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            QTextEdit {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 16px;
                font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
                font-size: 13px;
                color: {COLORS['text_primary']};
                selection-background-color: {COLORS['accent_primary']};
            }}
            QGroupBox {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                margin-top: 24px;
                padding: 20px 16px 16px 16px;
                font-weight: 600;
                color: {COLORS['text_primary']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 16px;
                top: 8px;
                padding: 0 8px;
                background-color: {COLORS['bg_card']};
                color: {COLORS['accent_secondary']};
                font-size: 13px;
                font-weight: 600;
            }}
            QScrollBar:vertical {{
                background: {COLORS['bg_dark']};
                width: 8px;
                border-radius: 4px;
                margin: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['border_light']};
                border-radius: 4px;
                min-height: 40px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLORS['accent_primary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollBar:horizontal {{
                background: {COLORS['bg_dark']};
                height: 8px;
                border-radius: 4px;
                margin: 4px;
            }}
            QScrollBar::handle:horizontal {{
                background: {COLORS['border_light']};
                border-radius: 4px;
                min-width: 40px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {COLORS['accent_primary']};
            }}
            QListWidget {{
                background-color: {COLORS['bg_input']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 4px;
                color: {COLORS['text_primary']};
                outline: none;
            }}
            QListWidget::item {{
                padding: 8px 12px;
                border-radius: 6px;
                margin: 2px;
            }}
            QListWidget::item:selected {{
                background-color: rgba(124, 58, 237, 0.3);
                color: {COLORS['accent_secondary']};
            }}
            QListWidget::item:hover {{
                background-color: {COLORS['bg_elevated']};
            }}
            QCheckBox {{
                color: {COLORS['text_secondary']};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid {COLORS['border']};
                background-color: {COLORS['bg_input']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['accent_primary']};
                border-color: {COLORS['accent_primary']};
            }}
            QCheckBox::indicator:hover {{
                border-color: {COLORS['accent_primary']};
            }}
            QToolTip {{
                background-color: {COLORS['bg_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
            }}
        """)

    def init_ui(self):
        self.setWindowTitle("◈ DataForge Studio")
        self.setGeometry(50, 50, 1500, 950)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ═══════════════════════════════════════════════════════════════════════
        # SIDEBAR NAVIGATION
        # ═══════════════════════════════════════════════════════════════════════
        sidebar = QFrame()
        sidebar.setFixedWidth(100)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border-right: 1px solid {COLORS['border']};
            }}
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 16, 8, 16)
        sidebar_layout.setSpacing(8)

        # Logo
        logo_label = QLabel("◈")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet(f"""
            font-size: 32px;
            color: {COLORS['accent_primary']};
            padding: 16px 0;
        """)
        sidebar_layout.addWidget(logo_label)

        # Navigation buttons
        self.nav_buttons = []
        nav_items = [
            "Data",
            "Stats",
            "Charts",
            "Process",
            "Filter",
            "Cluster",
            "Classify",
        ]

        for text in nav_items:
            btn = NavButton(text)
            btn.clicked.connect(lambda checked, t=text: self.switch_page(t))
            self.nav_buttons.append(btn)
            sidebar_layout.addWidget(btn)

        self.nav_buttons[0].setChecked(True)
        sidebar_layout.addStretch()

        # Theme indicator
        theme_label = QLabel("🌙")
        theme_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        theme_label.setStyleSheet("font-size: 20px; padding: 16px 0;")
        sidebar_layout.addWidget(theme_label)

        main_layout.addWidget(sidebar)

        # ═══════════════════════════════════════════════════════════════════════
        # MAIN CONTENT AREA
        # ═══════════════════════════════════════════════════════════════════════
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(20)

        # Header
        header = self.create_header()
        content_layout.addWidget(header)

        # Stacked pages
        self.pages = QStackedWidget()
        self.create_data_page()
        self.create_stats_page()
        self.create_viz_page()
        self.create_preprocessing_page()
        self.create_filter_page()
        self.create_clustering_page()
        self.create_classification_page()

        content_layout.addWidget(self.pages)
        main_layout.addWidget(content_container)

    def create_header(self):
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['bg_card']}, stop:1 {COLORS['bg_elevated']});
                border: 1px solid {COLORS['border']};
                border-radius: 16px;
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 16, 24, 16)

        # Title section
        title_section = QVBoxLayout()
        title_section.setSpacing(4)
        
        title_label = QLabel("DataForge Studio")
        title_label.setStyleSheet(f"""
            font-size: 22px;
            font-weight: 700;
            color: {COLORS['text_primary']};
            letter-spacing: -0.5px;
        """)
        
        subtitle_label = QLabel("Advanced Data Analysis & Machine Learning Platform")
        subtitle_label.setStyleSheet(f"""
            font-size: 12px;
            color: {COLORS['text_muted']};
            letter-spacing: 0.5px;
        """)
        
        title_section.addWidget(title_label)
        title_section.addWidget(subtitle_label)
        header_layout.addLayout(title_section)
        header_layout.addStretch()

        # File info badge
        self.file_badge = QFrame()
        self.file_badge.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_input']};
                border: 1px solid {COLORS['border']};
                border-radius: 20px;
                padding: 4px;
            }}
        """)
        badge_layout = QHBoxLayout(self.file_badge)
        badge_layout.setContentsMargins(12, 6, 12, 6)
        badge_layout.setSpacing(8)
        
        file_icon = QLabel("📄")
        self.file_name_label = QLabel("No file loaded")
        self.file_name_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; font-weight: 500;")
        badge_layout.addWidget(file_icon)
        badge_layout.addWidget(self.file_name_label)
        
        header_layout.addWidget(self.file_badge)

        # Load button
        self.load_btn = AccentButton("Open Dataset", "📁")
        self.load_btn.clicked.connect(self.load_csv)
        header_layout.addWidget(self.load_btn)

        return header

    def switch_page(self, page_name):
        page_map = {
            "Data": 0, "Stats": 1, "Charts": 2, "Process": 3,
            "Filter": 4, "Cluster": 5, "Classify": 6
        }
        idx = page_map.get(page_name, 0)
        self.pages.setCurrentIndex(idx)
        
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == idx)

    def create_data_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)
        
        self.rows_card = StatCard("Rows", "—")
        self.cols_card = StatCard("Columns", "—")
        self.missing_card = StatCard("Missing", "—")
        self.numeric_card = StatCard("Numeric", "—")
        
        stats_row.addWidget(self.rows_card)
        stats_row.addWidget(self.cols_card)
        stats_row.addWidget(self.missing_card)
        stats_row.addWidget(self.numeric_card)
        layout.addLayout(stats_row)

        # Table card
        table_card = ModernCard()
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(20, 20, 20, 20)
        
        table_header = QHBoxLayout()
        table_title = QLabel("Data Preview")
        table_title.setStyleSheet(f"font-size: 20px; font-weight: 600; color: {COLORS['text_primary']};")
        table_header.addWidget(table_title)
        table_header.addStretch()
        
        self.preview_info = QLabel("Showing first 100 rows")
        self.preview_info.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 14px;")
        table_header.addWidget(self.preview_info)
        table_layout.addLayout(table_header)

        self.table_widget = QTableWidget()
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setShowGrid(False)
        self.table_widget.verticalHeader().setVisible(False)
        table_layout.addWidget(self.table_widget)

        layout.addWidget(table_card)
        self.pages.addWidget(page)

    def create_stats_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Controls
        controls_card = ModernCard()
        controls_layout = QHBoxLayout(controls_card)
        controls_layout.setContentsMargins(20, 16, 20, 16)

        col_label = QLabel("Select Column")
        col_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 500;")
        self.stats_column_combo = QComboBox()
        self.stats_column_combo.setMinimumWidth(250)
        self.stats_column_combo.currentTextChanged.connect(self.update_statistics)
        
        controls_layout.addWidget(col_label)
        controls_layout.addWidget(self.stats_column_combo)
        controls_layout.addStretch()
        
        calc_btn = AccentButton("Calculate All", "📊")
        calc_btn.clicked.connect(self.calculate_all_statistics)
        controls_layout.addWidget(calc_btn)
        
        layout.addWidget(controls_card)

        # Results
        results_card = ModernCard()
        results_layout = QVBoxLayout(results_card)
        results_layout.setContentsMargins(20, 20, 20, 20)
        
        results_title = QLabel("📈  Statistical Analysis")
        results_title.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 12px;")
        results_layout.addWidget(results_title)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlaceholderText("Select a column to view statistics...")
        results_layout.addWidget(self.stats_text)
        
        layout.addWidget(results_card)
        self.pages.addWidget(page)

    def create_viz_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Sidebar
        sidebar = ModernCard()
        sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(16)

        sidebar_title = QLabel("Chart Settings")
        sidebar_title.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {COLORS['text_primary']};")
        sidebar_layout.addWidget(sidebar_title)

        # Chart type
        type_label = QLabel("Chart Type")
        type_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 500; margin-top: 8px;")
        sidebar_layout.addWidget(type_label)
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Histogram", 
            "Scatter Plot", 
            "Box Plot", 
            "Line Plot", 
            "Bar Chart",
            "Violin Plot",
            "Density Plot",
            "Correlation Heatmap"
        ])
        self.plot_type_combo.currentTextChanged.connect(self.update_plot_controls)
        sidebar_layout.addWidget(self.plot_type_combo)

        # X axis
        x_label = QLabel("X Axis")
        x_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 500; margin-top: 8px;")
        sidebar_layout.addWidget(x_label)
        self.x_column_combo = QComboBox()
        sidebar_layout.addWidget(self.x_column_combo)

        # Y axis
        self.y_label = QLabel("Y Axis")
        self.y_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 500; margin-top: 8px;")
        sidebar_layout.addWidget(self.y_label)
        self.y_column_combo = QComboBox()
        sidebar_layout.addWidget(self.y_column_combo)

        sidebar_layout.addStretch()

        plot_btn = AccentButton("Generate", "✨")
        plot_btn.clicked.connect(self.generate_plot)
        sidebar_layout.addWidget(plot_btn)

        save_btn = SecondaryButton("Export", "💾")
        save_btn.clicked.connect(self.save_plot)
        sidebar_layout.addWidget(save_btn)

        layout.addWidget(sidebar)

        # Chart area
        chart_card = ModernCard()
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(20, 20, 20, 20)

        self.figure = Figure(figsize=(10, 7), facecolor=COLORS['bg_card'])
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)

        layout.addWidget(chart_card)
        self.pages.addWidget(page)

    def create_preprocessing_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Sidebar
        sidebar = ModernCard()
        sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(12)

        sidebar_title = QLabel("⚙️  Preprocessing")
        sidebar_title.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {COLORS['text_primary']};")
        sidebar_layout.addWidget(sidebar_title)

        # Columns
        cols_label = QLabel("Select Columns")
        cols_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 500; margin-top: 8px;")
        sidebar_layout.addWidget(cols_label)
        
        self.prep_columns_list = QListWidget()
        self.prep_columns_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.prep_columns_list.setMaximumHeight(180)
        sidebar_layout.addWidget(self.prep_columns_list)

        # Operation
        op_label = QLabel("Operation")
        op_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 500; margin-top: 8px;")
        sidebar_layout.addWidget(op_label)
        
        self.prep_action_combo = QComboBox()
        self.prep_action_combo.addItems([
            "Impute Missing (Mean)", 
            "Impute Missing (Median)", 
            "Normalize (Min-Max)", 
            "Standardize (Z-Score)"
        ])
        sidebar_layout.addWidget(self.prep_action_combo)

        sidebar_layout.addStretch()

        run_btn = AccentButton("Apply", "▶️")
        run_btn.clicked.connect(self.run_preprocessing)
        sidebar_layout.addWidget(run_btn)

        reset_btn = SecondaryButton("Reset", "↺")
        reset_btn.clicked.connect(self.reset_data)
        sidebar_layout.addWidget(reset_btn)

        layout.addWidget(sidebar)

        # Table
        table_card = ModernCard()
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(20, 20, 20, 20)

        table_title = QLabel("📋  Transformed Data")
        table_title.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 12px;")
        table_layout.addWidget(table_title)

        self.prep_table = QTableWidget()
        self.prep_table.setAlternatingRowColors(True)
        self.prep_table.setShowGrid(False)
        self.prep_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.prep_table)

        layout.addWidget(table_card)
        self.pages.addWidget(page)

    def create_filter_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Filter bar
        filter_card = ModernCard()
        filter_layout = QHBoxLayout(filter_card)
        filter_layout.setContentsMargins(20, 16, 20, 16)
        filter_layout.setSpacing(12)

        filter_icon = QLabel("🔍")
        filter_icon.setStyleSheet("font-size: 20px;")
        filter_layout.addWidget(filter_icon)

        self.filter_column_combo = QComboBox()
        self.filter_column_combo.setMinimumWidth(180)
        self.filter_column_combo.setPlaceholderText("Select column...")
        filter_layout.addWidget(self.filter_column_combo)

        self.filter_condition_combo = QComboBox()
        self.filter_condition_combo.addItems([">", "<", ">=", "<=", "==", "!=", "contains"])
        self.filter_condition_combo.setFixedWidth(100)
        filter_layout.addWidget(self.filter_condition_combo)

        self.filter_value_input = QLineEdit()
        self.filter_value_input.setPlaceholderText("Enter value...")
        self.filter_value_input.setMinimumWidth(200)
        filter_layout.addWidget(self.filter_value_input)

        filter_layout.addStretch()

        apply_btn = AccentButton("Apply", "✓")
        apply_btn.clicked.connect(self.apply_filter)
        filter_layout.addWidget(apply_btn)

        clear_btn = SecondaryButton("Clear", "✕")
        clear_btn.clicked.connect(self.reset_filter)
        filter_layout.addWidget(clear_btn)

        layout.addWidget(filter_card)

        # Info row
        info_row = QHBoxLayout()
        self.filter_info_label = QLabel("No active filters")
        self.filter_info_label.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 13px;
            padding: 8px 0;
        """)
        info_row.addWidget(self.filter_info_label)
        info_row.addStretch()
        
        export_btn = SecondaryButton("Export Results", "📤")
        export_btn.clicked.connect(self.export_filtered_data)
        info_row.addWidget(export_btn)
        layout.addLayout(info_row)

        # Table
        table_card = ModernCard()
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(20, 20, 20, 20)

        self.filtered_table = QTableWidget()
        self.filtered_table.setAlternatingRowColors(True)
        self.filtered_table.setShowGrid(False)
        self.filtered_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.filtered_table)

        layout.addWidget(table_card)
        self.pages.addWidget(page)

    def create_clustering_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Sidebar
        sidebar = ModernCard()
        sidebar.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(12)

        sidebar_title = QLabel("Clustering")
        sidebar_title.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {COLORS['text_primary']};")
        sidebar_layout.addWidget(sidebar_title)

        # Algorithm
        algo_label = QLabel("Algorithm")
        algo_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 500; margin-top: 8px;")
        sidebar_layout.addWidget(algo_label)
        
        self.cluster_algo_combo = QComboBox()
        self.cluster_algo_combo.addItems(["K-Means", "K-Medoids", "AGNES (Hierarchical)", "DIANA (Hierarchical)", "DBSCAN"])
        self.cluster_algo_combo.currentTextChanged.connect(self.update_cluster_params)
        sidebar_layout.addWidget(self.cluster_algo_combo)

        # Parameters
        params_frame = QFrame()
        params_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_input']};
                border-radius: 8px;
                padding: 4px;
            }}
        """)
        params_layout = QFormLayout(params_frame)
        params_layout.setContentsMargins(12, 12, 12, 12)
        params_layout.setSpacing(10)

        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 20)
        self.k_spin.setValue(3)
        self.k_label = QLabel("Clusters (k):")
        self.k_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

        self.linkage_combo = QComboBox()
        self.linkage_combo.addItems(["single", "complete", "average"])
        self.linkage_label = QLabel("Linkage:")
        self.linkage_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setValue(0.5)
        self.eps_spin.setDecimals(2)
        self.eps_label = QLabel("Eps:")
        self.eps_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setValue(5)
        self.min_samples_label = QLabel("Min Samples:")
        self.min_samples_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

        params_layout.addRow(self.k_label, self.k_spin)
        params_layout.addRow(self.linkage_label, self.linkage_combo)
        params_layout.addRow(self.eps_label, self.eps_spin)
        params_layout.addRow(self.min_samples_label, self.min_samples_spin)

        sidebar_layout.addWidget(params_frame)

        # Elbow method checkbox (only for K-Means)
        self.elbow_checkbox = QCheckBox("Find optimal k (Elbow Method)")
        self.elbow_checkbox.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.elbow_checkbox.setToolTip("Automatically determine optimal number of clusters using Elbow Method")
        sidebar_layout.addWidget(self.elbow_checkbox)

        # Features
        feat_label = QLabel("Features (2+)")
        feat_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 500; margin-top: 8px;")
        sidebar_layout.addWidget(feat_label)
        
        self.cluster_feature_list = QListWidget()
        self.cluster_feature_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.cluster_feature_list.setMaximumHeight(140)
        sidebar_layout.addWidget(self.cluster_feature_list)

        sidebar_layout.addStretch()

        # Elbow method button
        self.elbow_btn = SecondaryButton("Find Optimal k")
        self.elbow_btn.clicked.connect(self.run_elbow_method)
        self.elbow_btn.setToolTip("Run Elbow Method to find optimal number of clusters")
        sidebar_layout.addWidget(self.elbow_btn)

        run_btn = AccentButton("Run Clustering", "▶️")
        run_btn.clicked.connect(self.run_clustering)
        sidebar_layout.addWidget(run_btn)

        layout.addWidget(sidebar)

        # Chart
        chart_card = ModernCard()
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(20, 20, 20, 20)

        chart_title = QLabel("Cluster Visualization")
        chart_title.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 12px;")
        chart_layout.addWidget(chart_title)

        self.cluster_figure = Figure(figsize=(10, 7), facecolor=COLORS['bg_card'])
        self.cluster_canvas = FigureCanvas(self.cluster_figure)
        chart_layout.addWidget(self.cluster_canvas)

        layout.addWidget(chart_card)
        self.pages.addWidget(page)
        self.update_cluster_params()

    def create_classification_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Sidebar
        sidebar = ModernCard()
        sidebar.setFixedWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(12)

        sidebar_title = QLabel("🤖  Classification")
        sidebar_title.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {COLORS['text_primary']};")
        sidebar_layout.addWidget(sidebar_title)

        # Algorithm
        algo_label = QLabel("Algorithm")
        algo_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 500; margin-top: 8px;")
        sidebar_layout.addWidget(algo_label)
        
        self.class_algo_combo = QComboBox()
        self.class_algo_combo.addItems(["K-Nearest Neighbors (KNN)", "Gaussian Naive Bayes"])
        self.class_algo_combo.currentTextChanged.connect(self.update_classification_params)
        sidebar_layout.addWidget(self.class_algo_combo)

        # Target
        target_label = QLabel("Target Variable (y)")
        target_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 500; margin-top: 8px;")
        sidebar_layout.addWidget(target_label)
        self.target_combo = QComboBox()
        sidebar_layout.addWidget(self.target_combo)

        # Parameters
        params_frame = QFrame()
        params_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_input']};
                border-radius: 8px;
            }}
        """)
        params_layout = QFormLayout(params_frame)
        params_layout.setContentsMargins(12, 12, 12, 12)
        params_layout.setSpacing(10)

        self.knn_k_spin = QSpinBox()
        self.knn_k_spin.setRange(1, 20)
        self.knn_k_spin.setValue(3)
        self.knn_k_label = QLabel("Neighbors (k):")
        self.knn_k_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

        self.find_optimal_k_check = QCheckBox("Find optimal k (1-10)")

        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        self.split_spin.setDecimals(2)
        split_label = QLabel("Test Size:")
        split_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

        params_layout.addRow(self.knn_k_label, self.knn_k_spin)
        params_layout.addRow("", self.find_optimal_k_check)
        params_layout.addRow(split_label, self.split_spin)

        sidebar_layout.addWidget(params_frame)

        # Features
        feat_label = QLabel("Features (X)")
        feat_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 500; margin-top: 8px;")
        sidebar_layout.addWidget(feat_label)
        
        self.class_feature_list = QListWidget()
        self.class_feature_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.class_feature_list.setMaximumHeight(120)
        sidebar_layout.addWidget(self.class_feature_list)

        sidebar_layout.addStretch()

        run_btn = AccentButton("Train & Evaluate", "▶️")
        run_btn.clicked.connect(self.run_classification)
        sidebar_layout.addWidget(run_btn)

        layout.addWidget(sidebar)

        # Results area
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(16)

        # Metrics card
        metrics_card = ModernCard()
        metrics_layout = QVBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(20, 20, 20, 20)

        metrics_title = QLabel("📊  Model Metrics")
        metrics_title.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 12px;")
        metrics_layout.addWidget(metrics_title)

        self.class_results_text = QTextEdit()
        self.class_results_text.setReadOnly(True)
        self.class_results_text.setPlaceholderText("Train a model to see results...")
        self.class_results_text.setMaximumHeight(200)
        metrics_layout.addWidget(self.class_results_text)

        results_layout.addWidget(metrics_card)

        # Chart card
        chart_card = ModernCard()
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(20, 20, 20, 20)

        self.class_figure = Figure(figsize=(10, 4), facecolor=COLORS['bg_card'])
        self.class_canvas = FigureCanvas(self.class_figure)
        chart_layout.addWidget(self.class_canvas)

        results_layout.addWidget(chart_card)
        layout.addWidget(results_container)

        self.pages.addWidget(page)
        self.update_classification_params()

    # ═══════════════════════════════════════════════════════════════════════════
    # LOGIC METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def update_cluster_params(self):
        algo = self.cluster_algo_combo.currentText()
        is_kmeans = "K-Means" in algo
        is_kmeans_or_medoids = "K-Means" in algo or "K-Medoids" in algo or "DIANA" in algo
        is_agnes = "AGNES" in algo
        is_dbscan = "DBSCAN" in algo

        self.k_label.setVisible(is_kmeans_or_medoids or is_agnes)
        self.k_spin.setVisible(is_kmeans_or_medoids or is_agnes)
        self.linkage_label.setVisible(is_agnes)
        self.linkage_combo.setVisible(is_agnes)
        self.eps_label.setVisible(is_dbscan)
        self.eps_spin.setVisible(is_dbscan)
        self.min_samples_label.setVisible(is_dbscan)
        self.min_samples_spin.setVisible(is_dbscan)
        
        # Elbow method only for K-Means
        self.elbow_checkbox.setVisible(is_kmeans)
        self.elbow_btn.setVisible(is_kmeans)

    def update_classification_params(self):
        algo = self.class_algo_combo.currentText()
        is_knn = "KNN" in algo
        self.knn_k_label.setVisible(is_knn)
        self.knn_k_spin.setVisible(is_knn)
        self.find_optimal_k_check.setVisible(is_knn)

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.filtered_df = self.df.copy()

                filename = file_path.split('/')[-1]
                self.file_name_label.setText(filename)
                self.file_badge.setStyleSheet(f"""
                    QFrame {{
                        background-color: rgba(124, 58, 237, 0.15);
                        border: 1px solid rgba(124, 58, 237, 0.3);
                        border-radius: 20px;
                    }}
                """)

                self.display_data()
                self.populate_controls()
                self.update_stat_cards()

                self.show_toast("Success", f"Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")
            except Exception as e:
                self.show_error("Error", f"Failed to load: {str(e)}")

    def update_stat_cards(self):
        if self.filtered_df is None:
            return
        self.rows_card.set_value(f"{len(self.filtered_df):,}")
        self.cols_card.set_value(len(self.filtered_df.columns))
        self.missing_card.set_value(f"{self.filtered_df.isnull().sum().sum():,}")
        self.numeric_card.set_value(len(self.filtered_df.select_dtypes(include=[np.number]).columns))

    def reset_data(self):
        if self.df is not None:
            self.filtered_df = self.df.copy()
            self.display_data()
            self.populate_controls()
            self.update_stat_cards()
            self.show_toast("Reset", "Data reset to original")

    def display_data(self):
        if self.filtered_df is None:
            return

        df = self.filtered_df.head(100)
        self._populate_table(self.table_widget, df)
        self._populate_table(self.prep_table, df)

        self.preview_info.setText(f"Showing {len(df):,} of {len(self.filtered_df):,} rows")

    def _populate_table(self, table: QTableWidget, df: pd.DataFrame):
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns.tolist())
        for i in range(len(df)):
            for j in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(i, j, item)
        table.resizeColumnsToContents()

    def display_filtered_data(self):
        if self.filtered_df is None:
            return
        df = self.filtered_df.head(100)
        self._populate_table(self.filtered_table, df)

    def populate_controls(self):
        if self.filtered_df is None:
            return

        cols = self.filtered_df.columns.tolist()
        num_cols = self.filtered_df.select_dtypes(include=[np.number]).columns.tolist()

        self.stats_column_combo.clear()
        self.stats_column_combo.addItems(num_cols)

        self.x_column_combo.clear()
        self.y_column_combo.clear()
        self.x_column_combo.addItems(cols)
        self.y_column_combo.addItems(num_cols)

        self.filter_column_combo.clear()
        self.filter_column_combo.addItems(cols)

        self.cluster_feature_list.clear()
        self.cluster_feature_list.addItems(num_cols)

        self.prep_columns_list.clear()
        self.prep_columns_list.addItems(num_cols)

        self.target_combo.clear()
        self.target_combo.addItems(cols)
        self.class_feature_list.clear()
        self.class_feature_list.addItems(num_cols)

    def update_statistics(self):
        col = self.stats_column_combo.currentText()
        if not col or self.filtered_df is None:
            return
        data = self.filtered_df[col].dropna()

        text = f"""╔══════════════════════════════════════════════════╗
║  📊  STATISTICS FOR: {col[:30]:<28} ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  Count       │  {len(data):<32,} ║
║  Mean        │  {data.mean():<32.4f} ║
║  Median      │  {data.median():<32.4f} ║
║  Std Dev     │  {data.std():<32.4f} ║
║  Variance    │  {data.var():<32.4f} ║
║  Min         │  {data.min():<32.4f} ║
║  Max         │  {data.max():<32.4f} ║
║  Range       │  {(data.max() - data.min()):<32.4f} ║
║  Q1 (25%)    │  {data.quantile(0.25):<32.4f} ║
║  Q3 (75%)    │  {data.quantile(0.75):<32.4f} ║
║  IQR         │  {(data.quantile(0.75) - data.quantile(0.25)):<32.4f} ║
║  Skewness    │  {data.skew():<32.4f} ║
║  Kurtosis    │  {data.kurtosis():<32.4f} ║
║                                                  ║
╚══════════════════════════════════════════════════╝"""
        self.stats_text.setText(text)

    def calculate_all_statistics(self):
        if self.filtered_df is None:
            return
        num_cols = self.filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        text = "╔══════════════════════════════════════════════════╗\n"
        text += "║          📊  COMPLETE STATISTICAL SUMMARY        ║\n"
        text += "╠══════════════════════════════════════════════════╣\n\n"

        for col in num_cols:
            data = self.filtered_df[col].dropna()
            text += f"▸ {col}\n"
            text += f"  Mean: {data.mean():.3f}  │  Median: {data.median():.3f}  │  Std: {data.std():.3f}\n"
            text += f"  Min: {data.min():.3f}  │  Max: {data.max():.3f}  │  Range: {data.max()-data.min():.3f}\n\n"

        text += "╚══════════════════════════════════════════════════╝"
        self.stats_text.setText(text)

    def update_plot_controls(self):
        ptype = self.plot_type_combo.currentText()
        needs_y = "Scatter" in ptype or "Line" in ptype or "Bar" in ptype
        self.y_column_combo.setEnabled(needs_y)
        self.y_label.setVisible(needs_y)
        self.y_column_combo.setVisible(needs_y or "Heatmap" not in ptype or "Density" not in ptype)

    def _safe_convert_to_numeric(self, series):
        """Safely convert a pandas series to numeric, handling errors"""
        try:
            # Try direct conversion
            result = pd.to_numeric(series, errors='coerce')
            if result.isna().all():
                # If all NaN, try converting strings to numeric
                result = pd.to_numeric(series.astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
            # Ensure we return a numeric type that numpy can handle
            return result.astype(np.float64, errors='ignore')
        except Exception:
            try:
                result = pd.to_numeric(series, errors='coerce')
                return result.astype(np.float64, errors='ignore')
            except Exception:
                # Last resort: return as-is if conversion completely fails
                return series

    def generate_plot(self):
        if self.filtered_df is None:
            self.show_error("Warning", "No data loaded. Please load a dataset first.")
            return
        
        ptype = self.plot_type_combo.currentText()
        x_col = self.x_column_combo.currentText()
        
        if not x_col:
            self.show_error("Warning", "Please select an X-axis column.")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Style the plot
        ax.set_facecolor(COLORS['bg_elevated'])
        self.figure.patch.set_facecolor(COLORS['bg_card'])
        ax.tick_params(colors=COLORS['text_secondary'])
        ax.spines['bottom'].set_color(COLORS['border'])
        ax.spines['top'].set_color(COLORS['border'])
        ax.spines['left'].set_color(COLORS['border'])
        ax.spines['right'].set_color(COLORS['border'])
        ax.xaxis.label.set_color(COLORS['text_secondary'])
        ax.yaxis.label.set_color(COLORS['text_secondary'])
        ax.title.set_color(COLORS['text_primary'])

        try:
            # Extract and convert data safely
            x_data = self.filtered_df[x_col].dropna()
            
            # For numeric plots, convert to numeric
            if ptype in ["Histogram", "Box Plot", "Density Plot"] or "Scatter" in ptype or "Line" in ptype:
                x_data = self._safe_convert_to_numeric(x_data)
                x_data = x_data.dropna()
                
                if len(x_data) == 0:
                    self.show_error("Error", f"Column '{x_col}' cannot be converted to numeric values.")
                    return

            if "Histogram" in ptype:
                if not pd.api.types.is_numeric_dtype(x_data):
                    x_data = self._safe_convert_to_numeric(x_data).dropna()
                ax.hist(x_data, bins=30, 
                       color=COLORS['accent_primary'], edgecolor=COLORS['bg_card'], alpha=0.8)
                ax.set_title(f"Distribution of {x_col}", fontsize=14, fontweight='bold')
                ax.set_xlabel(x_col)
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.2, color=COLORS['border'])

            elif "Scatter" in ptype:
                y_col = self.y_column_combo.currentText()
                if not y_col:
                    self.show_error("Warning", "Please select a Y-axis column for scatter plot.")
                    return
                
                y_data = self.filtered_df[y_col].dropna()
                x_data_clean = self.filtered_df[x_col].dropna()
                y_data_clean = y_data.copy()
                
                # Convert both to numeric
                x_data_clean = self._safe_convert_to_numeric(x_data_clean).dropna()
                y_data_clean = self._safe_convert_to_numeric(y_data_clean).dropna()
                
                # Align indices
                common_idx = x_data_clean.index.intersection(y_data_clean.index)
                x_data_clean = x_data_clean.loc[common_idx]
                y_data_clean = y_data_clean.loc[common_idx]
                
                if len(x_data_clean) == 0 or len(y_data_clean) == 0:
                    self.show_error("Error", "Cannot create scatter plot: insufficient numeric data.")
                    return
                
                scatter = ax.scatter(x_data_clean, y_data_clean, 
                                   alpha=0.6, c=COLORS['accent_secondary'], s=50, edgecolors='white', linewidth=0.5)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} vs {x_col}", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.2, color=COLORS['border'])

            elif "Box" in ptype:
                if not pd.api.types.is_numeric_dtype(x_data):
                    x_data = self._safe_convert_to_numeric(x_data).dropna()
                bp = ax.boxplot([x_data], labels=[x_col], patch_artist=True)
                bp['boxes'][0].set_facecolor(COLORS['accent_primary'])
                bp['boxes'][0].set_alpha(0.7)
                for element in ['whiskers', 'caps', 'medians']:
                    for item in bp[element]:
                        item.set_color(COLORS['accent_secondary'])
                ax.set_title(f"Box Plot of {x_col}", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.2, color=COLORS['border'], axis='y')

            elif "Line" in ptype:
                y_col = self.y_column_combo.currentText()
                if not y_col:
                    self.show_error("Warning", "Please select a Y-axis column for line plot.")
                    return
                
                y_data = self.filtered_df[y_col].dropna()
                x_data_clean = self.filtered_df[x_col].dropna()
                y_data_clean = y_data.copy()
                
                # Convert both to numeric
                x_data_clean = self._safe_convert_to_numeric(x_data_clean).dropna()
                y_data_clean = self._safe_convert_to_numeric(y_data_clean).dropna()
                
                # Align indices and sort by x
                common_idx = x_data_clean.index.intersection(y_data_clean.index)
                x_data_clean = x_data_clean.loc[common_idx].sort_values()
                y_data_clean = y_data_clean.loc[x_data_clean.index]
                
                if len(x_data_clean) == 0 or len(y_data_clean) == 0:
                    self.show_error("Error", "Cannot create line plot: insufficient numeric data.")
                    return
                
                ax.plot(x_data_clean, y_data_clean, 
                       marker='o', color=COLORS['accent_primary'], linewidth=2, markersize=4, alpha=0.8)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} over {x_col}", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, color=COLORS['border'])

            elif "Bar" in ptype:
                y_col = self.y_column_combo.currentText()
                if not y_col:
                    self.show_error("Warning", "Please select a Y-axis column for bar chart.")
                    return
                
                # For bar charts, x can be categorical
                x_data = self.filtered_df[x_col].dropna()
                y_data = self._safe_convert_to_numeric(self.filtered_df[y_col]).dropna()
                
                # Align indices
                common_idx = x_data.index.intersection(y_data.index)
                x_data = x_data.loc[common_idx]
                y_data = y_data.loc[common_idx]
                
                if len(x_data) == 0 or len(y_data) == 0:
                    self.show_error("Error", "Cannot create bar chart: insufficient data.")
                    return
                
                # If x is categorical or has many unique values, take first 50
                if len(x_data.unique()) > 50:
                    x_data = x_data.head(50)
                    y_data = y_data.head(50)
                
                ax.bar(range(len(x_data)), y_data, 
                      color=COLORS['accent_primary'], alpha=0.8, edgecolor=COLORS['bg_card'])
                ax.set_xticks(range(len(x_data)))
                ax.set_xticklabels(x_data.astype(str), rotation=45, ha='right', fontsize=9)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} by {x_col}", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.2, color=COLORS['border'], axis='y')

            elif "Violin" in ptype:
                if not pd.api.types.is_numeric_dtype(x_data):
                    x_data = self._safe_convert_to_numeric(x_data).dropna()
                
                try:
                    # Try to use seaborn-style violin plot with matplotlib
                    parts = ax.violinplot([x_data], positions=[1], showmeans=True, showmedians=True)
                    for pc in parts['bodies']:
                        pc.set_facecolor(COLORS['accent_primary'])
                        pc.set_alpha(0.7)
                    parts['cmeans'].set_color(COLORS['accent_secondary'])
                    parts['cmedians'].set_color(COLORS['accent_warning'])
                    ax.set_xticks([1])
                    ax.set_xticklabels([x_col])
                    ax.set_title(f"Violin Plot of {x_col}", fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.2, color=COLORS['border'], axis='y')
                except Exception:
                    # Fallback to boxplot if violin plot fails
                    bp = ax.boxplot([x_data], labels=[x_col], patch_artist=True)
                    bp['boxes'][0].set_facecolor(COLORS['accent_primary'])
                    bp['boxes'][0].set_alpha(0.7)
                    ax.set_title(f"Box Plot of {x_col} (Violin not available)", fontsize=14, fontweight='bold')

            elif "Density" in ptype:
                if not pd.api.types.is_numeric_dtype(x_data):
                    x_data = self._safe_convert_to_numeric(x_data).dropna()
                
                # Use histogram with density=True as density plot (KDE approximation)
                # This doesn't require scipy - uses numpy and matplotlib only
                try:
                    counts, bins, patches = ax.hist(x_data, bins=50, density=True, 
                                                   color=COLORS['accent_primary'], 
                                                   edgecolor=COLORS['bg_card'], 
                                                   alpha=0.6)
                    # Add a simple smooth line overlay using numpy convolution for smoothing
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    # Simple moving average for smoothing
                    if len(counts) > 3:
                        kernel_size = min(5, len(counts) // 5)
                        if kernel_size > 1:
                            kernel = np.ones(kernel_size) / kernel_size
                            smoothed = np.convolve(counts, kernel, mode='same')
                            ax.plot(bin_centers, smoothed, color=COLORS['accent_secondary'], 
                                   linewidth=2, label='Smooth', alpha=0.8)
                    
                    ax.set_xlabel(x_col)
                    ax.set_ylabel("Density")
                    ax.set_title(f"Density Plot of {x_col}", fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.2, color=COLORS['border'])
                except Exception as e:
                    # Fallback to simple histogram
                    ax.hist(x_data, bins=30, density=True, 
                           color=COLORS['accent_primary'], edgecolor=COLORS['bg_card'], alpha=0.8)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel("Density")
                    ax.set_title(f"Density Plot of {x_col}", fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.2, color=COLORS['border'])

            elif "Heatmap" in ptype:
                num_cols = self.filtered_df.select_dtypes(include=[np.number]).columns
                if len(num_cols) < 2:
                    self.show_error("Warning", "Need at least 2 numeric columns for correlation heatmap.")
                    return
                
                # Ensure all are numeric
                numeric_df = self.filtered_df[num_cols].apply(pd.to_numeric, errors='coerce')
                corr = numeric_df.corr()
                
                # Remove columns/rows with all NaN
                corr = corr.dropna(axis=0, how='all').dropna(axis=1, how='all')
                
                if corr.empty:
                    self.show_error("Error", "Cannot compute correlation: no valid numeric data.")
                    return
                
                im = ax.imshow(corr.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
                ax.set_yticklabels(corr.columns, fontsize=9)
                
                # Add correlation values as text
                for i in range(len(corr.columns)):
                    for j in range(len(corr.columns)):
                        text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
                
                self.figure.colorbar(im, ax=ax, label='Correlation')
                ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')

            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            import traceback
            error_msg = f"Error generating plot: {str(e)}\n\n{traceback.format_exc()}"
            self.show_error("Error", error_msg)

    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.figure.savefig(path, facecolor=COLORS['bg_card'], edgecolor='none', dpi=150)
            self.show_toast("Saved", f"Chart exported to {path.split('/')[-1]}")

    def run_preprocessing(self):
        if self.filtered_df is None:
            return

        items = self.prep_columns_list.selectedItems()
        cols = [i.text() for i in items]
        action = self.prep_action_combo.currentText()

        if not cols:
            self.show_error("Warning", "Select columns first")
            return

        try:
            data = self.filtered_df[cols].values

            if "Impute Missing" in action:
                strategy = 'mean' if 'Mean' in action else 'median'
                imputer = SimpleImputer(strategy=strategy)
                new_data = imputer.fit_transform(data)
                self.filtered_df[cols] = new_data

            elif "Normalize" in action:
                scaler = MinMaxScaler()
                new_data = scaler.fit_transform(data)
                self.filtered_df[cols] = new_data

            elif "Standardize" in action:
                scaler = StandardScaler()
                new_data = scaler.fit_transform(data)
                self.filtered_df[cols] = new_data

            self.display_data()
            self.update_stat_cards()
            self.show_toast("Applied", f"{action} on {len(cols)} columns")

        except Exception as e:
            self.show_error("Error", f"Preprocessing failed: {str(e)}")

    def apply_filter(self):
        if self.df is None:
            return
        col = self.filter_column_combo.currentText()
        cond = self.filter_condition_combo.currentText()
        val_str = self.filter_value_input.text()

        try:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                val = float(val_str)
                ops = {
                    ">": lambda x: x > val,
                    "<": lambda x: x < val,
                    ">=": lambda x: x >= val,
                    "<=": lambda x: x <= val,
                    "==": lambda x: x == val,
                    "!=": lambda x: x != val,
                }
                self.filtered_df = self.df[ops[cond](self.df[col])]
            else:
                if cond == "contains":
                    self.filtered_df = self.df[self.df[col].astype(str).str.contains(val_str)]
                elif cond == "==":
                    self.filtered_df = self.df[self.df[col] == val_str]

            self.display_filtered_data()
            self.filter_info_label.setText(f"✓ Filtered: {len(self.filtered_df):,} rows match '{col} {cond} {val_str}'")
            self.filter_info_label.setStyleSheet(f"color: {COLORS['accent_success']}; font-size: 13px; font-weight: 500;")
        except Exception as e:
            self.show_error("Error", str(e))

    def reset_filter(self):
        if self.df is not None:
            self.filtered_df = self.df.copy()
            self.display_filtered_data()
            self.filter_info_label.setText("No active filters")
            self.filter_info_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 13px;")
            self.filter_value_input.clear()

    def export_filtered_data(self):
        if self.filtered_df is None or len(self.filtered_df) == 0:
            self.show_error("Warning", "No data to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Filtered Data", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.filtered_df.to_csv(file_path, index=False)
                self.show_toast("Exported", f"{len(self.filtered_df):,} rows saved")
            except Exception as e:
                self.show_error("Error", f"Export failed: {str(e)}")

    def run_clustering(self):
        if self.filtered_df is None:
            return
        items = self.cluster_feature_list.selectedItems()
        feats = [i.text() for i in items]

        if len(feats) < 2:
            self.show_error("Warning", "Select at least 2 features")
            return

        try:
            X = self.filtered_df[feats].dropna().values
            algo = self.cluster_algo_combo.currentText()

            labels = None
            if "K-Means" in algo:
                labels = KMeans(k=self.k_spin.value()).fit(X)
            elif "K-Medoids" in algo:
                labels = KMedoids(k=k_value).fit(X)
            elif "AGNES" in algo:
                labels = AGNES(k=k_value, linkage=self.linkage_combo.currentText()).fit(X)
            elif "DIANA" in algo:
                labels = DIANA(k=k_value).fit(X)
            elif "DBSCAN" in algo:
                labels = DBSCAN(eps=self.eps_spin.value(), min_samples=self.min_samples_spin.value()).fit(X)

            self.plot_clusters(X, labels, feats, algo)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            self.show_toast("Complete", f"Found {n_clusters} clusters")

        except Exception as e:
            self.show_error("Error", str(e))

    def plot_clusters(self, X, labels, feats, title):
        self.cluster_figure.clear()
        ax = self.cluster_figure.add_subplot(111)

        # Style
        ax.set_facecolor(COLORS['bg_elevated'])
        self.cluster_figure.patch.set_facecolor(COLORS['bg_card'])
        ax.tick_params(colors=COLORS['text_secondary'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['border'])
        ax.xaxis.label.set_color(COLORS['text_secondary'])
        ax.yaxis.label.set_color(COLORS['text_secondary'])
        ax.title.set_color(COLORS['text_primary'])

        unique = np.unique(labels)
        # Custom color palette for clusters
        cluster_colors = ['#7c3aed', '#06b6d4', '#f472b6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#22d3ee']

        for i, cluster in enumerate(unique):
            if cluster == -1:
                color = COLORS['text_muted']
                label = 'Noise'
            else:
                color = cluster_colors[i % len(cluster_colors)]
                label = f'Cluster {cluster}'

            mask = labels == cluster
            ax.scatter(X[mask, 0], X[mask, 1], c=color, s=60, alpha=0.7, 
                      label=label, edgecolors='white', linewidth=0.5)

        ax.set_xlabel(feats[0], fontsize=11)
        ax.set_ylabel(feats[1], fontsize=11)
        ax.set_title(f"{title} Results", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.2, color=COLORS['border'])

        self.cluster_figure.tight_layout()
        self.cluster_canvas.draw()

    def run_classification(self):
        if self.filtered_df is None:
            return

        target = self.target_combo.currentText()
        feat_items = self.class_feature_list.selectedItems()
        features = [i.text() for i in feat_items]

        if not features:
            self.show_error("Warning", "Select at least one feature")
            return

        try:
            df_clean = self.filtered_df.dropna(subset=features + [target])
            X = df_clean[features].values
            y = df_clean[target].values

            test_size = self.split_spin.value()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            algo = self.class_algo_combo.currentText()

            if "KNN" in algo:
                if self.find_optimal_k_check.isChecked():
                    self.run_knn_optimization(X_train, X_test, y_train, y_test)
                else:
                    k = self.knn_k_spin.value()
                    model = KNN(k=k)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    self.show_metrics(y_test, y_pred, f"KNN (k={k})")

            elif "Naive Bayes" in algo:
                model = GaussianNaiveBayes()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                self.show_metrics(y_test, y_pred, "Gaussian Naive Bayes")
                self.class_figure.clear()
                self.class_canvas.draw()

        except Exception as e:
            self.show_error("Error", f"Classification failed: {str(e)}")

    def show_metrics(self, y_true, y_pred, title):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm, classes = confusion_matrix(y_true, y_pred)

        text = f"""╔══════════════════════════════════════════════════╗
║  🤖  MODEL: {title[:38]:<38} ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  ▸ Accuracy     │  {acc:<30.4f} ║
║  ▸ Precision    │  {prec:<30.4f} ║
║  ▸ Recall       │  {rec:<30.4f} ║
║  ▸ F1 Score     │  {f1:<30.4f} ║
║                                                  ║
╠══════════════════════════════════════════════════╣
║  CONFUSION MATRIX                                ║
║  Classes: {str(classes):<40} ║
╚══════════════════════════════════════════════════╝

{np.array2string(cm, separator='  ')}"""

        self.class_results_text.setText(text)

    def run_knn_optimization(self, X_train, X_test, y_train, y_test):
        precisions = []
        accuracies = []
        ks = range(1, 11)

        for k in ks:
            model = KNN(k=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            precisions.append(precision_score(y_test, y_pred))
            accuracies.append(accuracy_score(y_test, y_pred))

        # Plot
        self.class_figure.clear()
        ax = self.class_figure.add_subplot(111)

        ax.set_facecolor(COLORS['bg_elevated'])
        self.class_figure.patch.set_facecolor(COLORS['bg_card'])
        ax.tick_params(colors=COLORS['text_secondary'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['border'])

        ax.plot(ks, precisions, marker='o', color=COLORS['accent_primary'], 
               linewidth=2, markersize=8, label='Precision')
        ax.plot(ks, accuracies, marker='s', color=COLORS['accent_secondary'], 
               linewidth=2, markersize=8, label='Accuracy')

        ax.set_xlabel('k (Neighbors)', color=COLORS['text_secondary'], fontsize=11)
        ax.set_ylabel('Score', color=COLORS['text_secondary'], fontsize=11)
        ax.set_title('KNN Performance vs k', color=COLORS['text_primary'], fontsize=14, fontweight='bold')
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.2, color=COLORS['border'])
        ax.legend(loc='lower right')

        self.class_figure.tight_layout()
        self.class_canvas.draw()

        best_k = ks[np.argmax(precisions)]
        best_acc = accuracies[np.argmax(precisions)]
        self.class_results_text.setText(f"""╔══════════════════════════════════════════════════╗
║  🎯  KNN OPTIMIZATION COMPLETE                   ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  ▸ Optimal k        │  {best_k:<26} ║
║  ▸ Best Precision   │  {max(precisions):<26.4f} ║
║  ▸ Accuracy at k={best_k:<2}  │  {best_acc:<26.4f} ║
║                                                  ║
╚══════════════════════════════════════════════════╝""")

    def show_toast(self, title, message):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {COLORS['bg_card']};
            }}
            QMessageBox QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QPushButton {{
                background-color: {COLORS['accent_primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: 500;
            }}
        """)
        msg.exec()

    def show_error(self, title, message):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {COLORS['bg_card']};
            }}
            QMessageBox QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QPushButton {{
                background-color: {COLORS['accent_danger']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: 500;
            }}
        """)
        msg.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = DataAnalysisApp()
    window.show()
    sys.exit(app.exec())

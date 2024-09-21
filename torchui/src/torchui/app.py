# TorchUI 基于图形界面的 PyTorch 测试工具

###### 版本号：1.0.0

"""
A basic tool to test PyTorch with GUI
"""

import importlib.metadata
import sys

import json
import pickle
import sqlite3
import re
import os
import numpy as np
import itertools
import math
import ollama
import json
import pickle
import sqlite3
import sys
import re
import os
import numpy as np
import itertools
import math
import mistune
import chardet
import torch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import ConnectionStyle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib import collections

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    # Backwards compatibility - importlib.metadata was added in Python 3.8
    import importlib_metadata


from datetime import datetime
from PySide6.QtGui import QAction, QFont, QGuiApplication, QKeySequence,QShortcut
from PySide6.QtWidgets import QComboBox,QAbstractItemView, QHBoxLayout, QLabel, QMainWindow, QApplication, QMenu, QSizePolicy, QTextBrowser, QTextEdit, QWidget, QToolBar, QFileDialog, QTableView, QVBoxLayout, QHBoxLayout, QWidget, QSlider,  QGroupBox , QLabel , QWidgetAction, QPushButton, QSizePolicy
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QVariantAnimation, Qt, QTranslator, QLocale, QLibraryInfo

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from PySide6.QtGui import QGuiApplication

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex

from scipy.stats import gmean

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] =  'truetype'

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)
working_directory = os.path.dirname(current_file_path)
# 改变当前工作目录
os.chdir(current_directory)



class TorchUI(QMainWindow):
    def __init__(self):
        super().__init__()  
        self.resize(1024, 600)  # 设置窗口尺寸为1024*600  
        self.init_ui()
        self.show()

    def init_ui(self):
        self.setWindowTitle("TorchUI: A basic tool to test PyTorch with GUI")
        self.main_frame = QWidget()
        self.toolbar = QToolBar()   
        self.toolbar.setStyleSheet("font-size: 14px")        
        self.addToolBar(self.toolbar)  

        self.action = QAction('New Task', self)              
        self.input_text_edit = QTextEdit()
        self.text_browser = QTextBrowser()         
        self.button = QPushButton("Send\nCtrl+Enter")                
        self.label = QLabel("Role", self)
        self.selector = QComboBox(self)        
        # 创建一个水平布局并添加表格视图和画布

        self.input_text_edit.setFixedHeight(100)  # 设置文本编辑框的高度为100
        self.button.setFixedHeight(100)  # 设置按钮的高度为50

        self.base_layout = QVBoxLayout()
        self.upper_layout = QHBoxLayout()
        self.lower_layout = QHBoxLayout()

        self.toolbar.addAction(self.action)
        self.toolbar.addWidget(self.label)
        self.toolbar.addWidget(self.selector)
        
        self.button.clicked.connect(self.Magic)
        self.action.triggered.connect(self.Magic)

        self.upper_layout.addWidget(self.text_browser)

        
        self.lower_layout.addWidget(self.input_text_edit)
        self.lower_layout.addWidget(self.button)

        self.base_layout.addLayout(self.upper_layout)
        self.base_layout.addLayout(self.lower_layout)

        self.main_frame.setLayout(self.base_layout)
        self.setCentralWidget(self.main_frame)

    def Magic(self):
        print("Magic")
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        # Define a simple neural network using nn.Sequential
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

        # Generate some random data
        x_train = torch.randn(100, 10)
        y_train = torch.randn(100, 1)
        x_test = torch.randn(20, 10)
        y_test = torch.randn(20, 1)

        # Create DataLoader
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

        # Initialize the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        text_to_show =  (f"Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

        # Training loop
        for epoch in range(100):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            text_to_show += f"Epoch {epoch+1}, Loss: {loss.item()}\n"

        # Testing the model
        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test)
            test_loss = criterion(test_outputs, y_test)
            print(f"Test Loss: {test_loss.item()}")
            text_to_show += f"Test Loss: {test_loss.item()}\n"


        self.text_browser.append(text_to_show)

def main():
    # Linux desktop environments use an app's .desktop file to integrate the app
    # in to their application menus. The .desktop file of this app will include
    # the StartupWMClass key, set to app's formal name. This helps associate the
    # app's windows to its menu item.
    #
    # For association to work, any windows of the app must have WMCLASS property
    # set to match the value set in app's desktop file. For PySide6, this is set
    # with setApplicationName().

    # Find the name of the module that was used to start the app
    app_module = sys.modules["__main__"].__package__
    # Retrieve the app's metadata
    metadata = importlib.metadata.metadata(app_module)

    QApplication.setApplicationName(metadata["Formal-Name"])

    app = QApplication(sys.argv)
    main_window = TorchUI()
    sys.exit(app.exec())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = TorchUI()
    main_window.show()  # 显示主窗口
    sys.exit(app.exec())
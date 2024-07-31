import os
import sys
import json
import time
import cv2
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt

from matplotlib.colors import rgb2hex
from PyQt5.QtGui import QImage, QPixmap, QIntValidator
from PyQt5.QtCore import pyqtSignal, QThread, Qt, QPoint
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QSpacerItem,
    QMenu,
    QLabel,
    QAction,
    QWidgetAction,
    QScrollArea,
    QPushButton,
    QLineEdit,
    QComboBox,
)

from video_controller import Video
from main_detection import Main_Detection


def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class MainWindow(QMainWindow):
    options_signal = pyqtSignal(np.ndarray)

    style = """
        QMainWindow {
            background-color: rgb(40,40,40);
        }

        QLabel#menuBar_button {
            background-color: rgb(40,40,40);
            color: white;
            padding: 10px;
            padding-top: 2px;
            padding-bottom: 5px;
        }
        QLabel#menuBar_button:hover {
            background-color: rgb(60,60,60);
            border-radius: 6px;
        }
        QLabel[mode="clicked"] {
            background-color: rgb(40,40,40);
        }

        QLabel#menuBar_button_enabled {
            color: white;
            padding: 10px;
            padding-top: 2px;
            padding-bottom: 5px;
            background-color: rgb(38,79,120);     /*rgb(236, 74, 74*/
        }

   
                              
        QMenu {
            background-color: rgb(40,40,40);
            color: white;
            border: 1px solid black;
            padding: 4px;
        }

        QMenu::item {
            border-radius: 6px;
            border-style: outset;
            padding: 4px;
            padding-right: 20px;
        }

        QMenu::item::selected {
            background-color: rgb(60,60,60);
        }
   
        
        QPushButton {
            background-color: rgb(40,40,40);
            color: rgb(255,255,255);
            border: 1px solid black;
                           
            border-radius: 3px;
            border-style: outset;
            padding: 10px;
        }
        QPushButton:hover {
            background-color: rgb(60,60,60);
        }
        QPushButton:pressed {
            background-color: rgb(50,50,50);
        }
        
        
        QLabel {
            color: rgb(255,255,255);
        }

        QLabel#config {
            padding: 4px;
            padding-left: 6px;
        }

        QLabel#config:hover {
            background-color: rgb(60,60,60);
            border-radius: 6px;
            border-style: outset;
        }

        QFrame#menuBar_line {
            background-color: rgb(60,60,60);
        }

        QLineEdit {
            background-color: rgb(50,50,50);
            color: white;
            padding: 3px;
            border: 1px solid #000;
            border-radius: 3px;
            border-style: outset;
        }
        QLineEdit:focus {
            border: 1px solid rgb(0, 153, 188);
        }

        """

    default_config_path = resource_path("./config/default_config.json")
    saved_config_path = resource_path("./config/saved_config.json")

    background_img_path = resource_path("./media/background2.png")
    # icon_img_path = resource_path("./media/icon2.png")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Op|qO - Objective facial Palsy QuantificatiOn")
        # self.setWindowIcon(QIcon(self.icon_img_path))

        self.setMouseTracking(True)

        self.label_to_modify = None
        self.prev_time = 0
        self.show_fps = False
        self.show_recording = False
        self.saved_config = {}
        self.movie_thread = MovieThread()
        self.plot_window = PlotWindow()
        self.plot_recorded_data = PlotRecordedData()
        self.options = []
        self.cameras = {}
        self.card_points = []
        self.vertical_real_length = 1

        self.plot_recorded_data.mouseClicked.connect(self.movie_thread.jumpt_to_frame)

        # Layoutsl
        vlayout = QVBoxLayout()
        menu_layout = QHBoxLayout()

        main_layout = QHBoxLayout()
        right_layout = QVBoxLayout()

        # Establecer márgenes a cero para eliminar la separación
        menu_layout.setAlignment(Qt.AlignLeft)

        menu_layout.setContentsMargins(0, 0, 0, 0)
        vlayout.setContentsMargins(0, 5, 0, 5)
        menu_layout.setSpacing(0)

        # Setup menubar.
        self.setup_menubar(menu_layout)

        # Setup main content.
        self.backg_img = QImage(self.background_img_path)
        self.movie_thread.ImageUpdate.connect(
            self.update_movie
        )  # If a signal with a frame is received, update image.
        self.movie_thread.finished.connect(self.stopped_video)
        self.movie_thread.PauseUpdate.connect(
            lambda: self.toggle_pause(self.pauseLabel, True)
        )
        self.options_signal.connect(self.movie_thread.receive_options)
        self.options_signal.connect(self.plot_window.set_lines)

        self.image = QLabel(self)
        scaled_img = self.backg_img.scaled(
            int(self.backg_img.width() * 0.55),
            int(self.backg_img.height() * 0.55),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image.setPixmap(QPixmap(scaled_img))
        self.image.resizeEvent = self.resize_image
        self.image.setBaseSize(
            int(self.backg_img.width() * 0.2), int(self.backg_img.height() * 0.2)
        )
        self.image.setMinimumSize(
            int(self.backg_img.width() * 0.1), int(self.backg_img.height() * 0.1)
        )
        self.image.setMouseTracking(True)
        self.image.mouseMoveEvent = self.image_mouse_moved_event
        self.image.mousePressEvent = self.image_clicked_event
        main_layout.addWidget(self.image)
        main_layout.setStretchFactor(self.image, 1)

        self.scroll = QScrollArea()
        self.setup_scroll()
        self.scroll.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding))
        right_layout.addWidget(self.scroll)

        buttons_layout = QHBoxLayout()
        btn = QPushButton("Add")
        btn.pressed.connect(self.add_scroll_option)
        btn.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

        buttons_layout.addWidget(btn)
        btn = QPushButton("Remove")
        btn.pressed.connect(self.remove_scroll_option)
        btn.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        buttons_layout.addWidget(btn)

        spacer = QSpacerItem(40, 0, QSizePolicy.Minimum)
        buttons_layout.addItem(spacer)

        btn = QPushButton("Save")
        btn.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        buttons_layout.addWidget(btn)
        textbox = QLineEdit(self, placeholderText=" Name of saved config...")
        textbox.setContentsMargins(0, 0, 20, 0)
        buttons_layout.addWidget(textbox)
        btn.pressed.connect(lambda: self.add_new_config(textbox.text()))

        right_layout.addLayout(buttons_layout)

        main_layout.addLayout(right_layout)
        main_layout.setContentsMargins(5, 0, 0, 5)

        vlayout.addLayout(menu_layout)
        vlayout.addLayout(main_layout)

        widget = QWidget()
        widget.setMouseTracking(True)
        widget.setLayout(vlayout)
        self.setCentralWidget(widget)

        self.setStyleSheet(self.style)

    def resize_image(self, event):
        if not self.movie_thread.ThreadActive:
            self.set_background_img()

        # Call the base class resizeEvent
        QLabel.resizeEvent(self.image, event)

    def image_clicked_event(self, event):
        if not self.movie_thread.isRunning():
            return

        if self.label_to_modify and self.movie_thread.landmark_id:
            id = str(self.movie_thread.landmark_id)
            self.label_to_modify.setText(id)

        elif self.settings_scale_manual.isChecked():
            x = event.pos().x()
            y = event.pos().y()

            img_H = self.image.pixmap().height()
            img_W = self.image.pixmap().width()
            shape = (img_H, img_W)

            offset_y = (self.image.height() - img_H) / 2
            y = int(y - offset_y)

            h, w = shape
            height, width = self.movie_thread.image.shape[:2]

            new_x = round(x / w * width)
            new_y = round(y / h * height)

            point = np.array([new_x, new_y])

            if len(self.card_points) > 1:
                self.card_points = []

            self.card_points.append(point)

            if len(self.card_points) > 1:
                p1, p2 = self.card_points
                card_length_px = np.linalg.norm(p1 - p2)  # card side length in pixels.

                self.movie_thread.pixel_to_mm_factor = (
                    self.movie_thread.card_length_mm / card_length_px
                )
                self.movie_thread.V_DIST_PX_old = self.movie_thread.V_DIST_PX

    def image_mouse_moved_event(self, event):
        if self.movie_thread.isRunning():
            x = event.pos().x()
            y = event.pos().y()

            img_H = self.image.pixmap().height()
            img_W = self.image.pixmap().width()

            offset_y = (self.image.height() - img_H) / 2
            y = int(y - offset_y)
            selected = (x, y)

            self.movie_thread.get_mouse_pos(selected, (img_H, img_W))

    def update_movie(self, cv2_img):
        if self.show_fps:
            cv2_img = self.print_fps(cv2_img)

        if self.movie_thread.record:
            cv2.circle(
                cv2_img,
                (cv2_img.shape[1] - 25, 25),
                int(min(cv2_img.shape[:2]) / 50),
                (0, 0, 255),
                -1,
            )

        if self.movie_thread.pause:
            cv2_img = self.print_pause(cv2_img)

        if self.settings_scale_manual.isChecked():
            cv2_img = self.print_card_line(cv2_img)

        image = QImage(
            cv2_img.data, cv2_img.shape[1], cv2_img.shape[0], QImage.Format_RGB888
        ).rgbSwapped()
        self.image.setPixmap(
            QPixmap(image).scaled(
                self.image.width(),
                self.image.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

        if self.plot_window.isVisible() and self.movie_thread.mode == "Play":
            self.plot_window.update_plot(self.movie_thread.measures)

    def set_background_img(self):
        img = QPixmap(self.backg_img)
        scaled_img = img.scaled(
            self.image.width(),
            self.image.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image.setPixmap(scaled_img)

    def print_fps(self, img):
        elapsed_time = time.time() - self.prev_time

        if elapsed_time != 0:
            fps = round(1 / elapsed_time, 2)
            self.prev_time = time.time()
            cv2.putText(
                img,
                str(fps),
                (7, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (100, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return img

    def print_pause(self, img):
        A = (img.shape[1] // 2 - 10, img.shape[0] // 2 + min(img.shape[:2]) // 40)
        B = (img.shape[1] // 2 - 5, img.shape[0] // 2 - min(img.shape[:2]) // 40)
        img = cv2.rectangle(img, A, B, color=(200, 200, 200), thickness=2)

        A = (img.shape[1] // 2 + 10, img.shape[0] // 2 + min(img.shape[:2]) // 40)
        B = (img.shape[1] // 2 + 5, img.shape[0] // 2 - min(img.shape[:2]) // 40)
        img = cv2.rectangle(img, A, B, color=(200, 200, 200), thickness=2)

        return img

    def print_card_line(self, img):
        for point in self.card_points:
            img = cv2.circle(img, point, 2, (0, 255, 200), -1)

        if len(self.card_points) > 1:
            img = cv2.line(
                img, self.card_points[0], self.card_points[1], (0, 255, 200), 1
            )

        return img

    def start_video(self, mode):
        if mode == "Play":
            if not self.movie_thread.isRunning() and not self.movie_thread.ThreadActive:
                self.movie_thread.start()

        elif mode == "Stop" and self.movie_thread.isRunning():
            self.movie_thread.stop()

    def stopped_video(self):
        self.set_background_img()

        self.recordLabel.setObjectName("menuBar_button")
        self.pauseLabel.setObjectName("menuBar_button")
        self.recordLabel.setStyleSheet(self.style)
        self.pauseLabel.setStyleSheet(self.style)

    def setup_scroll(self):
        self.vbox = QVBoxLayout()

        self.vbox.addStretch()
        for _ in range(2):
            self.add_scroll_option()

        widget = QWidget()
        widget.setLayout(self.vbox)
        widget.setStyleSheet(
            self.style
            + """
                            QWidget {
                                background-color: rgb(40,40,40);
                            }
                            QLineEdit {
                                background-color: rgb(50,50,50);
                                color: white;
                                padding: 3px;
                                border: 1px solid #000;
                                border-radius: 3px;
                                border-style: outset;
                            }
                            QLineEdit:focus {
                                
                                border: 1px solid rgb(0, 153, 188);
                            }

                            QComboBox, QAbstractItemView {
                                padding: 3px;
                                border: 1px solid #000;
                                border-radius: 3px;
                                border-style: outset;
                                color: white;
                                min-width: 3em;                     

                                background-color: rgb(50,50,50);
                                selection-background-color: rgb(40,40,40);
                            }
                             """
        )

        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(widget)

    def add_scroll_option(self):
        if self.vbox.count() >= 3:
            return

        options_layout = QHBoxLayout()
        options_layout.setContentsMargins(0, 10, 0, 10)

        btn = QPushButton(objectName="show_lines_btn")
        btn.setFixedSize(15, 15)
        options_layout.addWidget(btn)

        options_layout.addWidget(QLabel("Measurement point:", self))

        textbox = QLineEdit(placeholderText=" Point to measure...")
        textbox.setValidator(QIntValidator(0, 478))
        textbox.textChanged.connect(self.update_options)
        textbox.focusInEvent = lambda event, sender=textbox: self.text_focus(
            event, sender
        )
        textbox.focusOutEvent = lambda event, sender=textbox: self.text_focus(
            event, sender
        )
        textbox.setContentsMargins(0, 0, 20, 0)
        options_layout.addWidget(textbox)

        options_layout.addWidget(QLabel("axis:", self))

        menu = QComboBox(self)
        menu.addItems([" Vertical axis", " Horizontal axis"])
        menu.currentIndexChanged.connect(self.update_options)
        options_layout.addWidget(menu)

        self.vbox.insertLayout(len(self.vbox.children()), options_layout)

        id = self.vbox.count() - 2
        btn.pressed.connect(lambda: self.toggle_lines(id))
        color = rgb2hex(np.array(self.plot_window.cmap(id)))
        btn.setStyleSheet(
            "QPushButton:hover {background-color: rgb(100,100,100);}"
            + f"QPushButton {{background-color: {color}; padding: 0px;}}"
        )

        self.update_options()

    def remove_scroll_option(self):
        if self.vbox.count() >= 2:
            last = self.vbox.takeAt(self.vbox.count() - 2)

            while last.count():
                item = last.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

            self.update_options()

    def update_options(self):
        self.options = np.empty((0, 3), dtype=int)

        for item in self.vbox.children():
            point = item.itemAt(2).widget().text()
            point = int(point) if point else -1
            point = point if point < 478 else -1
            axis = int(item.itemAt(4).widget().currentIndex())
            A_axis, B_axis = [[10, 152], [33, 263]][axis]
            self.options = np.vstack([self.options, [point, A_axis, B_axis]])

        self.options_signal.emit(self.options)

    def toggle_lines(self, id):
        if len(self.plot_window.lines) - 1 >= id:
            visible = self.plot_window.lines[id].isVisible()
            self.plot_window.lines[id].setVisible(not visible)

    def text_focus(self, event, sender):
        if event.gotFocus():  # Focus in.
            self.label_to_modify = sender
            QLineEdit.focusInEvent(
                sender, event
            )  # Running the rest of the original function.
        else:  # Focuss out.
            self.label_to_modify = None
            QLineEdit.focusOutEvent(
                sender, event
            )  # Running the rest of the original function.

    def setup_menubar(self, menu_layout):
        # Options for the File menu:
        fileMenu = QMenu(self)
        fileLabel = QLabel("File", self, objectName="menuBar_button")
        fileLabel.mousePressEvent = lambda _: self.show_menu(fileMenu, fileLabel)

        self.file_video = QAction(
            " Open Video File...", fileMenu, triggered=self.file_actions
        )
        self.file_recorded = QAction(
            " Open Recorded File...", fileMenu, triggered=self.file_actions
        )

        fileMenu.addActions([self.file_video, self.file_recorded])
        menu_layout.addWidget(fileLabel)

        # Options for the Settings menu:
        settingsMenu = QMenu(self)
        settingsLabel = QLabel("Settings", self, objectName="menuBar_button")
        settingsLabel.mousePressEvent = lambda _: self.show_menu(
            settingsMenu, settingsLabel
        )

        self.settings_landmarks = QAction(
            " Show Facial Landmarks",
            settingsMenu,
            triggered=self.settings_actions,
            checkable=True,
        )
        self.settings_axis = QAction(
            " Show Reference Lines",
            settingsMenu,
            triggered=self.settings_actions,
            checkable=True,
            checked=True,
            enabled=False,
        )
        self.settings_scale_manual = QAction(
            " Set Measures Scale (Manual)",
            settingsMenu,
            triggered=self.settings_actions,
            checkable=True,
        )
        self.settings_scale_auto = QAction(
            " Set Measures Scale (Auto)",
            settingsMenu,
            triggered=self.settings_actions,
            checkable=True,
        )
        self.settings_fps = QAction(
            " Show FPS", settingsMenu, triggered=self.settings_actions, checkable=True
        )
        self.settings_contour = QAction(
            " Show Facial Contour",
            settingsMenu,
            triggered=self.settings_actions,
            checkable=True,
            checked=True,
        )
        self.settings_plot = QAction(
            "      Show Plot Window", settingsMenu, triggered=self.settings_actions
        )
        self.settings_blur = QAction(
            " Blur background",
            settingsMenu,
            triggered=self.settings_actions,
            checkable=True,
        )

        settingsMenu.addActions(
            [
                self.settings_landmarks,
                self.settings_axis,
                self.settings_scale_manual,
                self.settings_scale_auto,
            ]
        )
        settingsMenu.addSeparator()
        settingsMenu.addActions(
            [self.settings_fps, self.settings_contour, self.settings_plot]
        )
        settingsMenu.addSeparator()
        settingsMenu.addAction(self.settings_blur)
        settingsMenu.addSeparator()

        self.defaultConfigMenu = settingsMenu.addMenu("      Default Outcome Measures")
        self.savedConfigMenu = settingsMenu.addMenu("      Saved Outcome Measures")

        self.config_clear = QAction(
            " Clear Configuration", self, triggered=self.settings_actions
        )
        self.defaultConfigMenu.addAction(self.config_clear)
        self.savedConfigMenu.addAction(self.config_clear)
        self.defaultConfigMenu.addSeparator()
        self.savedConfigMenu.addSeparator()

        self.load_config("default")
        self.load_config("saved")

        menu_layout.addWidget(settingsLabel)

        # Options for the Camera menu:
        cameraMenu = QMenu(self)
        cameraLabel = QLabel("Use Camera", self, objectName="menuBar_button")
        cameraLabel.mousePressEvent = lambda _: self.find_cameras(
            cameraMenu, cameraLabel
        )
        menu_layout.addWidget(cameraLabel)

        # Options for Pause button:
        self.pauseLabel = QLabel("Pause", self, objectName="menuBar_button")
        self.pauseLabel.mousePressEvent = lambda _: self.toggle_pause(self.pauseLabel)
        self.pauseLabel.mouseReleaseEvent = lambda _: self.menuLabel_release(
            self.pauseLabel
        )
        menu_layout.addWidget(self.pauseLabel)

        # Options for Record button:
        self.recordLabel = QLabel("Record", self, objectName="menuBar_button")
        self.recordLabel.mousePressEvent = lambda _: self.toggle_record(
            self.recordLabel
        )
        self.recordLabel.mouseReleaseEvent = lambda _: self.menuLabel_release(
            self.recordLabel
        )
        menu_layout.addWidget(self.recordLabel)

    def show_menu(self, menu, label):
        menu.aboutToHide.connect(lambda: label.setStyleSheet(""))

        # Obtener la posición global del QLabel
        label_pos = label.mapToGlobal(QPoint(0, 0))

        # Mostrar el menú debajo del QLabel
        menu_pos = QPoint(label_pos.x(), label_pos.y() + label.height())
        menu.exec_(menu_pos)

    def file_actions(self):
        button = self.sender()

        if button.text() == self.file_video.text():
            self.movie_thread.stop()

            file_name, _ = QFileDialog.getOpenFileName(
                filter="Video files (*.mov *.mp4)"
            )
            if file_name:
                self.movie_thread.source = file_name
                self.toggle_pause(self.pauseLabel, True)

                time.sleep(0.2)  # For giving time to the thread to close.
                # if not self.movie_thread.isRunning() and not self.movie_thread.ThreadActive:
                self.start_video("Play")

        elif (
            button.text() == self.file_recorded.text()
            and not self.plot_recorded_data.isVisible()
        ):
            file_name, _ = QFileDialog.getOpenFileName(filter="CSV files (*.csv)")

            if file_name:
                # self.movie_thread.stop()
                self.plot_recorded_data.load_data(file_name)
                self.plot_recorded_data.show()

                # Opening recorded video.
                source = file_name.split(".")[0] + ".mp4"

                self.movie_thread.source = source
                self.movie_thread.video.open_video(source)
                self.movie_thread.record = False
                self.toggle_pause(self.pauseLabel, True)
                self.movie_thread.recorded_data = list()
                self.movie_thread.recorded_images = list()

                self.movie_thread.mode = "Re-Play"

                time.sleep(0.2)  # For giving time to the thread to close.
                self.movie_thread.new_frame = True
                self.start_video("Play")

        else:
            exit("[ERROR]: UNKNOWN FILE ACTION!")

    def settings_actions(self):
        button = self.sender()
        parent = self.sender().parent()

        if button.text() in [self.settings_landmarks.text(), self.settings_axis.text()]:
            opposite = set(
                [self.settings_landmarks.text(), self.settings_axis.text()]
            ) - set([button.text()])

            button.setEnabled(False)
            for i, action in enumerate(parent.actions()):
                if action.text() in opposite:
                    parent.actions()[i].setEnabled(True)
                    parent.actions()[i].setChecked(False)

            self.movie_thread.mode = (
                "Play" if button.text() == self.settings_axis.text() else "Landmarks"
            )

        elif button.text() == self.settings_scale_manual.text():
            pass

        elif button.text() == self.settings_fps.text():
            self.show_fps = not self.show_fps

        elif button.text() == self.settings_contour.text():
            self.movie_thread.show_contour = not self.movie_thread.show_contour

        elif button.text() == self.settings_blur.text():
            self.movie_thread.blur_background = not self.movie_thread.blur_background

        elif button.text() == self.settings_plot.text():
            if not self.plot_window.isVisible():
                self.plot_window.show()

        elif button.text() == self.settings_scale_auto.text():
            self.movie_thread.enable_card_detection = button.isChecked()

        elif button.text() == self.config_clear.text():
            self.set_config(["", ""], [0, 0])

    def find_cameras(self, cameraMenu, cameraLabel):
        self.cameras = self.movie_thread.video.find_cameras()

        # remove all the previous buttons in the menu.
        for action in cameraMenu.actions():
            cameraMenu.removeAction(action)

        # Add cameras to menu.
        for id, name in self.cameras.items():
            action = QAction(
                f" {id}: {name}", cameraMenu, triggered=self.camera_actions
            )
            cameraMenu.addAction(action)

        self.show_menu(cameraMenu, cameraLabel)

    def camera_actions(self):
        button = self.sender()
        id = 0
        self.movie_thread.stop()

        for key, name in self.cameras.items():
            if name == button.text()[4:]:
                id = key
                break

        self.movie_thread.source = id
        if not self.movie_thread.mode in ["Play", "Landmarks"]:
            self.movie_thread.mode = "Play"
        time.sleep(0.2)  # For giving time to the thread to close.
        self.start_video("Play")

    def toggle_record(self, button):
        button.setStyleSheet("background-color: rgb(40,40,40);")
        if self.movie_thread.mode == "Play" and self.movie_thread.ThreadActive:
            options = [n[0] for n in self.options if n[0] != -1]

            if not len(options):
                button.setStyleSheet(self.style)

                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)

                msg.setWindowTitle("No measures introduced!")
                msg.setText(
                    "You need to add some points to measure before pressing record!"
                )

                msg.setStandardButtons(QMessageBox.Ok)

                msg.exec_()

                return

            self.movie_thread.record = not self.movie_thread.record

            if self.movie_thread.record:
                button.setObjectName("menuBar_button_enabled")
            else:
                button.setObjectName("menuBar_button")

            if (
                not self.movie_thread.record and self.movie_thread.recorded_data
            ):  # If record is dissabled, save record.
                file_name, _ = QFileDialog.getSaveFileName(
                    self, "Save CSV and MP4 File", "readings"
                )
                self.movie_thread.save_recording(file_name)

    def toggle_pause(self, button, force=False):
        button.setStyleSheet("background-color: rgb(40,40,40);")
        if self.movie_thread.ThreadActive or force:
            self.movie_thread.pause = not self.movie_thread.pause

            if self.movie_thread.pause:
                button.setObjectName("menuBar_button_enabled")
            else:
                button.setObjectName("menuBar_button")

        if force:
            self.menuLabel_release(button)

    def menuLabel_release(self, button):
        button.setStyleSheet(self.style)

    def load_config(self, mode):
        if mode == "saved":
            path = self.saved_config_path
        else:
            path = self.default_config_path

        try:
            with open(path, "r") as f:
                config = json.load(f)

                if mode == "saved":
                    self.saved_config = config
                else:
                    self.default_config = config
        except (json.decoder.JSONDecodeError, FileNotFoundError):
            return

        for name in config:
            widget = QWidget(self)
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QLabel(name, widget, objectName="config"))

            # Crear las acciones con los widgets personalizados
            action = QWidgetAction(self, triggered=self.set_config)
            action.setDefaultWidget(widget)
            action.setText(name)
            action.setProperty("default", mode == "default")

            if mode == "saved":
                self.savedConfigMenu.addAction(action)
                btn = QPushButton("-", widget)
                btn.setFixedSize(15, 15)
                btn.setObjectName(name)
                btn.pressed.connect(self.remove_saved_config)
                layout.addWidget(btn)
            else:
                self.defaultConfigMenu.addAction(action)

    def set_config(self, landmarks=None, axis=None):
        if landmarks is None or axis is None:
            sender = self.sender()
            name = sender.text()

            if sender.property("default"):
                landmarks, axis = self.default_config[name]
            else:
                landmarks, axis = self.saved_config[name]

        while self.vbox.count() >= 2:
            self.remove_scroll_option()

        while self.vbox.count() <= len(landmarks):
            self.add_scroll_option()

            for i, item in enumerate(self.vbox.children()):
                item.itemAt(2).widget().setText(str(landmarks[i]))  # Id
                item.itemAt(4).widget().setCurrentIndex(axis[i])  # Axis

    def save_config(self):
        with open(self.saved_config_path, "w") as f:
            json.dump(self.saved_config, f)

    def add_new_config(self, name: str):
        if not name:
            name = "Option " + str(len(self.saved_config) + 1)

        if name in self.saved_config.keys():
            return

        options = [[], []]
        for item in self.vbox.children():
            point = item.itemAt(2).widget().text()
            point = int(point) if point and int(point) < 478 else ""
            axis = int(item.itemAt(4).widget().currentIndex())

            options[0].append(point)
            options[1].append(axis)

        # Update dict of configs.
        self.saved_config[name] = options.copy()

        # Add new config to menu.
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(name, widget, objectName="config"))
        btn = QPushButton("-", widget)
        btn.setFixedSize(15, 15)
        btn.pressed.connect(self.remove_saved_config)
        btn.setObjectName(name)
        layout.addWidget(btn)

        action = QWidgetAction(self, triggered=self.set_config)
        action.setDefaultWidget(widget)
        action.setText(name)
        action.setProperty("default", False)
        self.savedConfigMenu.addAction(action)

        self.save_config()

    def remove_saved_config(self):
        sender = self.sender()
        name = sender.objectName()

        actions = [action.text() for action in self.savedConfigMenu.actions()]
        id = actions.index(name)
        action = self.savedConfigMenu.actions()[id]

        widget = action.defaultWidget()
        layout = widget.layout()

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.savedConfigMenu.removeAction(action)
        self.saved_config.pop(name)

        self.save_config()

    def closeEvent(self, event):
        self.movie_thread.stop()
        self.plot_window.close()
        self.plot_recorded_data.close()
        event.accept()


class MovieThread(QThread, Main_Detection):
    ImageUpdate = pyqtSignal(np.ndarray)
    PauseUpdate = pyqtSignal()

    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self.mouse_pos = None
        self.mode = "Play"
        self.ThreadActive = False
        self.record = False
        self.recorded_data = list()
        self.recorded_images = list()
        self.pause = False
        self.video = Video(source)
        self.new_frame = False

    def run(self):
        self.ThreadActive = True

        ret = self.video.open_video(self.source)
        if not ret:
            self.stop()

        img = None
        ret = True

        while self.ThreadActive:
            if (
                img is not None and self.pause
            ):  # Update of image while app paused to allow resize of paused image.
                if self.new_frame:
                    _, img = self.video.get_frame()
                    self.new_frame = False

                self.ImageUpdate.emit(img.copy())
                time.sleep(0.1)
                continue

            ret, img = self.video.get_frame()

            if not ret:
                self.video.open_video(self.source)
                self.PauseUpdate.emit()
                continue

            if self.mode == "Re-Play":
                time.sleep(0.04)
            else:
                self.process_image(img)  # , int(cap.get(cv2.CAP_PROP_POS_MSEC)))

                if self.mode == "Play":
                    self.measure_user_exercises()

                    if self.blur_background:
                        self.draw_blurred_background()

                    img = self.draw_measurements_on_image()

                    if self.enable_card_detection:
                        img = self.draw_card_detection()

                    if self.record:
                        self.record_exercise(img.copy())

                elif self.mode == "Landmarks":
                    if self.blur_background:
                        self.draw_blurred_background()

                    img = self.draw_face_mesh()

                elif self.mode == "Card_auto":
                    None
                    # Get card detection.

            self.ImageUpdate.emit(img)

        self.stop()

    def stop(self):
        self.video.release()

        self.ThreadActive = False
        self.record = False
        self.pause = False
        self.recorded_data = list()
        self.recorded_images = list()

        self.quit()

    def get_mouse_pos(self, selected, shape):
        if not self.ThreadActive or self.mode != "Landmarks" or self.image is None:
            return

        x, y = selected

        h, w = shape
        height, width = self.image.shape[:2]

        new_x = x / w * width
        new_y = y / h * height

        self.selected = (new_x, new_y)

    def receive_options(self, event):
        self.options = [x for x in event.copy() if x[0] != 1]

    def record_exercise(self, img):
        if self.record and self.measures:
            self.recorded_data.append(self.measures.copy())
            self.recorded_images.append(img)

    def save_recording(self, file_name):
        if not self.record and self.recorded_data:
            if file_name:
                # Save recorded data.
                np.savetxt(file_name + ".csv", self.recorded_data, delimiter=",")

                # Save recorded video.
                fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                video_out = cv2.VideoWriter(
                    file_name + ".mp4", fourcc, 24.0, (640, 480)
                )
                for frame in self.recorded_images:
                    video_out.write(frame)

                # Save maximum of recorded data.
                maxims = np.max(self.recorded_data, axis=0).reshape(1, -1)
                path, name = file_name.rsplit("/", 1)
                np.savetxt(path + "/max_" + name + ".csv", maxims, delimiter=",")

            self.recorded_data = list()

    def jumpt_to_frame(self, frame_index):
        self.pause = True
        self.video.set_frame_index(frame_index)
        self.new_frame = True


class PlotWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-Time Plot")
        self.setGeometry(100, 100, 800, 400)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setTitle("Measurements")
        label_style = {"color": "#EEE", "font-size": "12pt"}
        self.plot_widget.setLabel("bottom", "Time", "s", **label_style)
        self.plot_widget.setLabel("left", "Distance", "mm", **label_style)
        self.plot_widget.getAxis("left").enableAutoSIPrefix(False)
        self.plot_widget.getAxis("bottom").enableAutoSIPrefix(False)

        self.plot_widget.getAxis("left").setTextPen("w")
        self.plot_widget.getAxis("bottom").setTextPen("w")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)

        self.cmap = plt.get_cmap("tab10")

        self.size_data = 400
        self.names = "Left", "Right"

    def set_lines(self, options):
        self.plot_widget.clear()

        lines_ids = [i for i, x in enumerate(options) if x[0] != -1]

        # Clear timestamp and data (data as list of lists).
        self.time = []
        self.data = [[] for _ in lines_ids]
        self.lines = []

        for i, id in enumerate(lines_ids):
            color = np.array(self.cmap(i)) * 255
            line = self.plot_widget.plot(
                name=self.names[id], pen=pg.mkPen(color=color, width=2)
            )
            self.lines.append(line)

    def update_plot(self, measures):
        if not measures or len(measures) != len(self.lines):
            return

        if not self.time:
            self.timer_start = time.time()
            self.time.append(0)
        else:
            self.time.append(time.time() - self.timer_start)

        if len(self.time) > self.size_data:
            self.time = self.time[1:]

        for i, line in enumerate(self.lines):
            self.data[i].append(measures[i])

            if len(self.data[i]) > self.size_data:
                self.data[i] = self.data[i][1:]

            line.setData(self.time, self.data[i])


class PlotRecordedData(PlotWindow):
    mouseClicked = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Recorded Data Plot")
        label_style = {"color": "#EEE", "font-size": "12pt"}
        self.plot_widget.setLabel("bottom", "Samples", **label_style)
        self.data = None
        self.desired_frame = None

        # Create a scatter plot item for the hover point
        self.hoverPoint = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush([100, 100, 100, 255])
        )

        # Create a text item for displaying coordinates
        self.hoverText = pg.TextItem(anchor=(0.5, -1.5), color="w")
        self.plot_widget.addItem(self.hoverText)

        # Enable mouse tracking
        self.plot_widget.setMouseTracking(
            True
        )  # Enable mouse tracking on the plot widget
        self.old_event = self.plot_widget.mouseMoveEvent
        self.plot_widget.mouseMoveEvent = self.custom_mouseMoveEvent

    def load_data(self, file_name):
        self.plot_widget.clear()

        self.data = np.loadtxt(file_name, delimiter=",", dtype=float).transpose()
        if not len(self.data):
            return

        if not isinstance(self.data[0], np.ndarray):
            self.data = [self.data]

        step = np.arange(len(self.data[0]))

        for id, line in enumerate(self.data):
            color = np.array(self.cmap(id)) * 255
            max_id = np.argmax(line)
            min_id = np.argmin(line)
            name = (
                f"{self.names[id]} - (max:{line[max_id]:.3f} , min:{line[min_id]:.3f})"
            )

            scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(color))
            scatter.addPoints([max_id, min_id], [line[max_id], line[min_id]])

            self.plot_widget.plot(
                step, line, name=name, pen=pg.mkPen(color=color, width=1)
            )
            self.plot_widget.addItem(scatter)

        self.plot_widget.getViewBox().autoRange()  # Autofocus on the data.

        self.plot_widget.addItem(self.hoverPoint)
        self.plot_widget.addItem(self.hoverText)

    def mousePressEvent(self, event):
        pos = event.pos()
        if self.plot_widget.sceneBoundingRect().contains(pos):
            if self.desired_frame is not None:
                self.mouseClicked.emit(self.desired_frame)

        super().mousePressEvent(event)

    def custom_mouseMoveEvent(self, event):
        pos = event.pos()
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x, y = np.clip(mouse_point.x(), 0, len(self.data[0]) - 1), mouse_point.y()
            # Calculate closest data point
            closest_distance = float("inf")
            closest_coordinates = None
            for y_data in self.data:
                index = round(x)
                distance = abs(y_data[index] - y)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_coordinates = (round(x), y_data[index])

            if closest_coordinates and closest_distance < 1:  # specify your radius here
                self.hoverPoint.setData(
                    [closest_coordinates[0]], [closest_coordinates[1]]
                )
                self.hoverText.setPos(closest_coordinates[0], closest_coordinates[1])
                self.hoverText.setHtml(
                    f"<div style='color: white;'>({closest_coordinates[0]:.2f}, {closest_coordinates[1]:.2f})</div>"
                )
                self.desired_frame = closest_coordinates[0]
            else:
                self.hoverPoint.setData([], [])  # Clear data if no close point
                self.hoverText.setText("")
                self.desired_frame = None

        self.old_event(event)


if __name__ == "__main__":
    app = QApplication(sys.argv + ["-platform", "windows:darkmode=1"])

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

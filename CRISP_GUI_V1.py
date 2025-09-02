# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# CRISP: Cremated Remains Inference of Sex Probabilities - GUI
# Created and maintained by Sophie Beitel @ cBits
# Protected under GPL-3.0
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

import os
import sys
from copy import deepcopy

import pandas as pd
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, \
    QHBoxLayout, QMainWindow, QFileDialog, QScrollArea, QMessageBox, QToolButton, QStackedLayout, \
    QDialog, QDialogButtonBox, QFormLayout, QTextEdit  # for graphic ui
from PyQt6.QtCore import Qt, QDir
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

import CRISP_CORE_V4 as crisp

title = "CRISP"
version = "0.1"
year = "2025"
license_type = "GPL-3.0"
dev_mode = True


def resource_path(relative_path):
    # Returns Path to resource reliably both when executed as .exe and as .py
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, figure=None):
        super().__init__(figure)


class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Additional Info required to export")

        self.first = QLineEdit(self)
        self.second = QLineEdit(self)
        self.third = QLineEdit(self)
        self.fourth = QTextEdit(self)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Site:", self.first)
        layout.addRow("Grave number:", self.second)
        layout.addRow("Individual:", self.third)
        layout.addRow("Additional notes (optional):", self.fourth)
        layout.addWidget(button_box)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

    def get_inputs(self):
        return self.first.text(), self.second.text(), self.third.text(), self.fourth.toPlainText()


class MainWindow(QMainWindow):
    window_height = 720  # 900 # = 1920:1080
    window_width = 1280  # 1600
    singleCase = []
    file_path = None
    save_path = None
    file_name = None
    warn_heading = None
    scroll_heading = None
    input_lines = []
    input_labels = []
    page_labels = []
    export_data = []    # holds raw data (dict), results (array, [male, female]), warnings (str) and plots (figure)
    copyright_notice = f"©{title} V-{version} {year}. Rights reserved under {license_type}."

    def __init__(self):
        super().__init__()

        # what is displayed in the window title:
        self.setWindowTitle(title + ' - V' + version)
        self.setGeometry(100, 100, self.window_width, self.window_height)  # first two is where, second two is size

        menu_bar = self.menuBar()
        self.status_bar = self.statusBar()  # displays the StatusTips of Actions

        help_menu = menu_bar.addMenu("&Help")

        manual_action = QAction("&Manual", self)
        manual_action.setStatusTip("Get the Manual")
        manual_action.triggered.connect(self.open_pdf_manual)

        about_action = QAction("&About", self)
        about_action.setStatusTip("About the Program")
        about_action.triggered.connect(self.about_popup)

        cite_action = QAction("&Reference", self)
        cite_action.setStatusTip("How to cite this software")
        cite_action.triggered.connect(self.cite_popup)

        help_menu.addAction(manual_action)
        help_menu.addAction(about_action)
        help_menu.addAction(cite_action)

        self.stacked = QStackedLayout()

        page_1 = QWidget(self)
        page_1.setLayout(self.generate_input_page_layout(True))

        self.stacked.addWidget(page_1)
        self.stacked.setCurrentIndex(0)

        # set the layout:
        container = QWidget()
        container.setLayout(self.stacked)
        self.setCentralWidget(container)

        # show your work!
        self.show()

    def generate_input_page_layout(self, buttons=False):
        # Layouts:
        big_box = QVBoxLayout()  # holds Header Label, Subheader HBox, Columns HBox, Buttons HBox
        subheader_hbox = QHBoxLayout()  # holds Subheader 1 and Subheader 2 HBoxes
        sub_box1 = QHBoxLayout()  # holds Subheader 1 Label
        sub_box2 = QHBoxLayout()  # holds Subheader 2 Label
        columns_hbox = QHBoxLayout()  # holds columns 1 through 4/5
        buttons_hbox = QHBoxLayout()  # holds clear case, clear file and calculate HBoxes
        button_box1 = QHBoxLayout()  # holds clear case button
        button_box2 = QHBoxLayout()  # holds clear file, calculate and next page buttons
        col1 = QVBoxLayout()  # holds 7 labels and 7 lines
        col2 = QVBoxLayout()  # holds 7 labels and 7 lines
        col3 = QVBoxLayout()  # holds 7 labels and 7 lines
        col4 = QVBoxLayout()  # holds File HBox, Template HBox and Warnings Label
        file_hbox = QHBoxLayout()  # holds File Button and File Label
        template_hbox = QHBoxLayout()  # holds Template Button and Template Label

        # Labels:
        heading = QLabel("CRISP: Cremated Remains Inference of Sex Probabilities")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setObjectName('heading')  # for formatting
        sub_heading1 = QLabel("Case-by-Case Input:")
        sub_heading2 = QLabel("Multiple Cases Input:")
        sub_heading1.setObjectName("subheading")  # for formatting
        sub_heading2.setObjectName("subheading")  # for formatting
        self.warn_heading = QLabel(wordWrap=True)
        self.warn_heading.setObjectName('warns')  # for formatting
        file_heading = QLabel(".xlss file")
        template_heading = QLabel("Get the .xlss template!")
        copyright_heading = QLabel(self.copyright_notice)
        copyright_heading.setAlignment(Qt.AlignmentFlag.AlignBottom)
        copyright_heading.setObjectName('copyright')  # for formatting

        # Make Warn heading scrollable if it gets too long:
        scrollable_layout = QVBoxLayout()  # layout for inside the scrollable bit
        scrollable_layout.addWidget(self.warn_heading)
        scrollable_layout.addStretch()
        scrollable_widget = QWidget()  # widget to hold the layout
        scrollable_widget.setLayout(scrollable_layout)
        scrollable_widget.setObjectName("warningareas")
        self.scroll_heading = QScrollArea()
        self.scroll_heading.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_heading.setWidgetResizable(True)
        self.scroll_heading.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_heading.verticalScrollBar().rangeChanged.connect(
            self.scroll_to_bottom)  # always jumps down if new line is added
        self.scroll_heading.setWidget(scrollable_widget)

        # Inputs and Input Labels:
        for n in crisp.parameter_table["metric_vars"]:
            self.input_lines.append(QLineEdit(placeholderText="Enter Measurement"))
            self.input_labels.append(QLabel(str(n)))

        # Buttons:
        clear_case = QPushButton("Clear Case", clicked=self.clear_case)
        clear_file = QPushButton("Clear File Path", clicked=self.clear_file)
        get_template = QPushButton("Template", clicked=self.template)
        choose_file = QPushButton("Browse...", clicked=self.choose_file)
        calculate = QPushButton("Calculate", clicked=self.calculate)

        # Fill Layouts:
        big_box.addWidget(heading)
        big_box.addLayout(subheader_hbox)
        big_box.addLayout(columns_hbox)
        big_box.addLayout(buttons_hbox)

        subheader_hbox.addLayout(sub_box1, 3)
        subheader_hbox.addLayout(sub_box2, 2)

        sub_box1.addWidget(sub_heading1)
        sub_box1.addStretch()

        sub_box2.addWidget(sub_heading2)
        sub_box2.addStretch()

        columns_hbox.addLayout(col1, 1)
        columns_hbox.addLayout(col2, 1)
        columns_hbox.addLayout(col3, 1)
        columns_hbox.addLayout(col4, 2)

        # Input lines:
        col1.addSpacing(25)
        col2.addSpacing(25)
        col3.addSpacing(25)
        for n in range(0, 7):
            col1.addWidget(self.input_labels[n])
            col1.addWidget(self.input_lines[n])
        for n in range(7, 14):
            col2.addWidget(self.input_labels[n])
            col2.addWidget(self.input_lines[n])
        for n in range(14, 19):
            col3.addWidget(self.input_labels[n])
            col3.addWidget(self.input_lines[n])
        col1.addStretch()
        col2.addStretch()
        col3.addStretch()

        col4.addSpacing(25)
        col4.addLayout(file_hbox)
        col4.addLayout(template_hbox)
        col4.addWidget(self.scroll_heading, 4)
        col4.addStretch(1)

        file_hbox.addWidget(choose_file)
        file_hbox.addWidget(file_heading)
        file_hbox.addStretch()

        template_hbox.addWidget(get_template)
        template_hbox.addWidget(template_heading)
        template_hbox.addStretch()

        buttons_hbox.addLayout(button_box1, 3)
        buttons_hbox.addLayout(button_box2, 2)

        button_box1.addWidget(copyright_heading)
        button_box1.addStretch()
        button_box1.addWidget(clear_case)

        button_box2.addWidget(clear_file)
        button_box2.addStretch()
        button_box2.addWidget(calculate)

        if buttons:
            buttons_box3 = QHBoxLayout()

            next_button = QToolButton()
            next_action = QAction()
            next_action.triggered.connect(self.next_page)
            next_button.setArrowType(Qt.ArrowType.RightArrow)
            next_button.setDefaultAction(next_action)

            buttons_box3.addStretch()
            buttons_box3.addWidget(next_button)

            button_box2.addLayout(buttons_box3)

        return big_box

    def generate_single_results_page_layout(self, plots=None, result: list = None, warnings: str = None):
        # Layouts:
        big_box = QVBoxLayout()  # holds Header Label, Subheader HBox and Body HBox
        subheader_hbox = QHBoxLayout()  # holds Subheader and Export Button
        body_hbox = QHBoxLayout() # holds plots VBox and Warnings VBox
        plots_vbox = QVBoxLayout()  # holds plots
        warnings_vbox = QVBoxLayout()  # holds Warning Label and Results
        prob_hbox = QHBoxLayout()  # holds F VBox and M VBox
        f_vbox = QVBoxLayout()  # holds F Label and F Probability Label
        m_vbox = QVBoxLayout()  # holds M Label and M Probability Label
        button_box = QHBoxLayout()  # holds start/prev/next Buttons
        bottom_box = QHBoxLayout() # holds buttons and copyright

        # Labels:
        heading = QLabel("CRISP: Cremated Remains Inference of Sex Probabilities")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setObjectName('heading')  # for formatting
        sub_heading = QLabel("Results")
        sub_heading.setObjectName("subheading")  # for formatting
        plots_label = QLabel("Plots go here (sorry)")
        warnings_heading = QLabel("Warnings")
        warnings_heading.setObjectName("prob")  # for formatting
        warnings_label = QLabel(wordWrap=True)
        warnings_label.setObjectName('warns')  # for formatting
        prob_heading = QLabel("Probability")
        prob_heading.setObjectName("prob")  # for formatting
        m_heading = QLabel("Male:")
        f_heading = QLabel("Female:")
        self.page_labels.append(QLabel())
        copyright_heading = QLabel(self.copyright_notice)
        copyright_heading.setAlignment(Qt.AlignmentFlag.AlignBottom)
        copyright_heading.setObjectName('copyright')  # for formatting

        # index 0 is male, index 1 is female
        m_prob = QLabel(f"{result[0] * 100:.2f}%")
        f_prob = QLabel(f"{result[1] * 100:.2f}%")

        # Make Warnings scrollable if it gets too long:
        scrollable_layout = QVBoxLayout()  # layout for inside the scrollable bit
        scrollable_layout.addWidget(warnings_label)
        scrollable_layout.addStretch()
        scrollable_widget = QWidget()  # widget to hold the layout
        scrollable_widget.setObjectName("warningareas")  # for formatting
        scrollable_widget.setLayout(scrollable_layout)
        scroll_heading2 = QScrollArea()
        scroll_heading2.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_heading2.setWidgetResizable(True)
        scroll_heading2.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_heading2.setWidget(scrollable_widget)

        warnings_label.setText(warnings)

        # Buttons:
        export_res = QPushButton("Export", clicked=lambda:self.export(len(self.page_labels)-1))
        # Lambda to hand over variables to func

        # Fill Layouts:
        big_box.addWidget(heading)
        big_box.addLayout(subheader_hbox)
        big_box.addLayout(body_hbox)
        big_box.addLayout(bottom_box)

        subheader_hbox.addWidget(sub_heading)
        subheader_hbox.addStretch()

        body_hbox.addLayout(plots_vbox, 4)
        body_hbox.addLayout(warnings_vbox, 1)

        try:
            plots_canvas = MplCanvas(plots)
            plots_vbox.addWidget(plots_canvas)
        except Exception as e:
            print(e)
            plots_vbox.addWidget(plots_label)

        warnings_vbox.addWidget(warnings_heading)
        warnings_vbox.addWidget(scroll_heading2, 1)
        warnings_vbox.addWidget(prob_heading)
        warnings_vbox.addLayout(prob_hbox)
        warnings_vbox.addWidget(export_res)
        warnings_vbox.addStretch(1)

        prob_hbox.addLayout(f_vbox)
        prob_hbox.addLayout(m_vbox)

        f_vbox.addWidget(f_heading)
        f_vbox.addWidget(f_prob)

        m_vbox.addWidget(m_heading)
        m_vbox.addWidget(m_prob)

        bottom_box.addWidget(copyright_heading)
        bottom_box.addStretch()
        bottom_box.addLayout(button_box)

        next_button = QToolButton()
        next_action = QAction()
        next_action.triggered.connect(self.next_page)
        next_button.setArrowType(Qt.ArrowType.RightArrow)
        next_button.setDefaultAction(next_action)

        prev_button = QToolButton()
        prev_action = QAction()
        prev_action.triggered.connect(self.prev_page)
        prev_button.setArrowType(Qt.ArrowType.LeftArrow)
        prev_button.setDefaultAction(prev_action)

        jump_button = QToolButton()
        jump_action = QAction()
        jump_action.triggered.connect(self.jump_to_start)
        jump_action.setText(" New Input")
        jump_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        jump_button.setArrowType(Qt.ArrowType.LeftArrow)
        jump_button.setDefaultAction(jump_action)

        button_box.addStretch()
        button_box.addWidget(self.page_labels[len(self.page_labels)-1])
        button_box.addWidget(jump_button)
        button_box.addWidget(prev_button)
        button_box.addWidget(next_button)

        return big_box

    def refresh_pages(self):
        # re-sets page labels to display correct maximum
        for i in range(0, len(self.page_labels)):
            self.page_labels[i].setText(f"Page {i+1} of {len(self.page_labels)}")

    def next_page(self):
        # navigates to the next page, if existing
        next_ind = self.stacked.currentIndex() + 1
        if next_ind <= self.stacked.count(): self.stacked.setCurrentIndex(next_ind)

    def prev_page(self):
        # navigates to previous page, if existing
        prev_ind = self.stacked.currentIndex() - 1
        if prev_ind >= 0: self.stacked.setCurrentIndex(prev_ind)

    def jump_to_start(self):
        self.stacked.setCurrentIndex(0)

    def scroll_to_bottom(self):
        # Set the scroll bar to its maximum value (auto-scroll to bottom when elements are added)
        self.scroll_heading.verticalScrollBar().setValue(self.scroll_heading.verticalScrollBar().maximum())

    def clear_case(self):
        # clear all input lines
        for n in self.input_lines:
            n.clear()
        self.warn_heading.setText(self.warn_heading.text() + "Case-by-Case Input cleared." + "\n")
        if dev_mode: print("cleared case")

    def clear_file(self):
        # clear the file path to the input file from memory, error message if there was none selected
        if self.file_path is not None:
            self.file_path = None
            self.warn_heading.setText(self.warn_heading.text() + "File selection cleared successfully\n")
            if dev_mode: print("cleared file path: " + str(self.file_path))
        else:
            self.warn_heading.setText(self.warn_heading.text() + "Error: No file selected.\n")
            if dev_mode: print("file path already clear")

    def calculate(self):
        # get the input data from single case input
        if len(self.singleCase) == 0:
            for n in self.input_lines:
                temp = n.text().strip()
                if temp != "":
                    try:
                        temp = temp.replace(",", ".")
                    except Exception as e:
                        if dev_mode: print(e)

                    try:
                        self.singleCase.append(float(temp))
                    except ValueError:
                        self.warn_heading.setText(self.warn_heading.text() + "Warning: Non-numeric data input will be"
                                                                             " ignored.\n")
                        self.singleCase.append(None)
                else: self.singleCase.append(None)
        else:
            for n in range(0, len(self.input_lines)):
                temp = self.input_lines[n].text().strip()
                if temp != "":
                    try:
                        temp = temp.replace(",", ".")
                    except Exception as e:
                        if dev_mode: print(e)

                    try:
                        self.singleCase[n] = float(temp)
                    except ValueError:
                        self.warn_heading.setText(self.warn_heading.text() + "Warning: Non-numeric data input will be"
                                                                             " ignored.\n")
                        self.singleCase[n] = None
                else: self.singleCase[n] = None

        # check to see if any single case input was given at all
        ok = False
        for x in self.singleCase:
            if x is not None: ok = True
        # ok remains False if all elements in singleCase are None
        if not ok: self.singleCase = []

        # determine which calculation to do
        if ok and self.file_path: # both inputs are valid
            try:
                answer = QMessageBox()
                answer.setWindowTitle('Multiple Inputs detected')
                answer.setText('Which input shall be processed?')
                button1 = answer.addButton("Single", QMessageBox.ButtonRole.ActionRole)
                answer.addButton("Multiple", QMessageBox.ButtonRole.RejectRole)
                answer.exec()
                if answer.clickedButton() == button1:
                    self.calc_single()
                else:
                    self.calc_file()
            except Exception as e:
                print(e)
        elif ok:  # only single case input is valid
            self.calc_single()
        elif self.file_path:  # only file input is valid
            self.calc_file()
        else: # no valid input
            self.warn_heading.setText(self.warn_heading.text() + "Error: No data provided." + "\n")

    def choose_file(self):
        # choose input file via dialog
        if self.file_path is None:
            self.file_name, ok = QFileDialog.getOpenFileName(self, "Select a File", "C:\\", "Excel File (*.xlsx)")

            if self.file_name:
                # get the path to file for easier opening
                self.file_path = Path(self.file_name)
            else:
                self.warn_heading.setText(self.warn_heading.text() + "Error: No file chosen." + '\n')
                if dev_mode: print("no file chosen :(")

            # check if the file path is valid
            if self.file_path:
                self.warn_heading.setText(self.warn_heading.text() + "File chosen successfully: " +
                                          str(self.file_path) + '\n')
                if dev_mode: print("file chosen: " + str(self.file_path))
            else:
                self.warn_heading.setText(self.warn_heading.text() + "Error: Invalid file location." + '\n')
                if dev_mode: print("file invalid")
        else:
            try:
                new_file_name, ok = QFileDialog.getOpenFileName(self, "Select a File", "C:\\", "Excel File (*.xlsx)")
                new_file_path = ""

                if new_file_name:
                    # get the path to file for easier opening
                    new_file_path = Path(self.file_name)
                else:
                    self.warn_heading.setText(self.warn_heading.text() + "Error: No file chosen." + '\n')
                    if dev_mode: print("no file chosen :(")

                # check if the file path is valid
                if new_file_path:
                    self.file_path = new_file_path
                    self.file_name = new_file_name
                    self.warn_heading.setText(self.warn_heading.text() + "File chosen successfully: " +
                                              str(self.file_path) + '\n')
                    if dev_mode: print("file chosen: " + str(self.file_path))
                else:
                    self.warn_heading.setText(self.warn_heading.text() + "Error: Invalid file location." + '\n')
                    if dev_mode: print("file invalid")
            except Exception as e:
                self.warn_heading.setText(self.warn_heading.text() + str(e) + '\n')

    def calc_single(self):
        # data preparation
        raw_data = {}
        for n in range(0, len(crisp.parameter_table["metric_vars"])):
            raw_data[crisp.parameter_table["metric_vars"][n]] = self.singleCase[n]

        result, warn_str, figs = crisp.calculate_single(raw_data)

        self.export_data.append([raw_data, result, warn_str, figs])
        if dev_mode: print(self.export_data[len(self.export_data)-1])

        # add new page with results
        new_page = QWidget(self)
        new_page.setLayout(self.generate_single_results_page_layout(figs, result, warn_str))

        self.stacked.addWidget(new_page)
        self.stacked.setCurrentIndex(self.stacked.count()-1)

        # refresh page numbers
        self.refresh_pages()

        if dev_mode: print(f"male: {result[0] * 100:.2f}%, female: {result[1] * 100:.2f}%")

    # TODO: (?) dialog to determine whether to overwrite input file
    def calc_file(self):
        if self.file_path is not None:
            # try - except in case file path has become invalid through user interference
            try:
                # read file and get prepared data
                cases_prep, cases_raw, warn_str, critical = crisp.file_cases(self.file_path)

                if not critical:
                    if cases_prep is not None and cases_raw is not None:
                        # calculate and save file
                        self.file_save_dialog(crisp.calculate_file(cases_prep, cases_raw), self.file_name, 1)

                        QMessageBox.information(self, 'Multiple Cases Input', f'File was processed successfully. '
                                                                          f'Reference {self.save_path} for results.')

                    if warn_str != "":
                        self.warn_heading.setText(self.warn_heading.text() + f"Warning: {warn_str}\nFile was "
                                                                             f"processed.\n")
                    else:
                        self.warn_heading.setText(self.warn_heading.text() + "File was processed." + "\n")
                    if dev_mode: print("calculated file")

                else:
                    self.warn_heading.setText(self.warn_heading.text() + f"{warn_str}\nFile was not processed!\n")
            except Exception as e:
                self.warn_heading.setText(self.warn_heading.text() + "Error: File has been opened, moved or "
                                                                    "deleted since selection.\n")
                if dev_mode: print(e)

    def file_save_dialog(self, contents, filename: str, path_mode: int = 0):
        if path_mode == 0:  # save as new file with generated name; 0 is default value for save_mode
            directory_name = QFileDialog.getExistingDirectory(self, "Save File in", "C:\\")

            if directory_name:
                self.save_path = Path(directory_name)
            else:
                return False

            # join the directory path and file name together to get the final path to file
            self.save_path = os.path.join(self.save_path, filename)
            if dev_mode: print(self.save_path)

        elif path_mode == 1: # overwrite input file
            self.save_path = self.file_path
            if dev_mode: print(self.save_path)

        try:
            # write contents to file
            contents.to_excel(self.save_path, index=False)
            self.warn_heading.setText(self.warn_heading.text() + "File saved successfully: " + str(self.save_path) + '\n')
            if dev_mode: print("file saved: " + str(self.save_path))
            return True
        except Exception as e:
            self.warn_heading.setText(self.warn_heading.text() + f"Error: {e}" + '\n')
            if dev_mode: print(f"Error saving result: {e}")
            return False

    # TODO: (?) add template file as source file instead of generating
    def template(self):
        # generate template from expected columns
        template = pd.DataFrame(columns=crisp.expected_columns)
        ok = self.file_save_dialog(template, 'template.xlsx')
        if ok:
            self.warn_heading.setText(self.warn_heading.text() + f"Template was saved in {self.save_path}" + '\n')
            if dev_mode: print("template generated")
        else:
            self.warn_heading.setText(self.warn_heading.text() + f"Process cancelled." + '\n')
            if dev_mode: print("template generation cancelled")

    def open_pdf_manual(self):
        if getattr(sys, 'frozen', False):  # if EXE
            base_path = sys._MEIPASS  # temporary folder of PyInstaller
        else:
            base_path = os.path.dirname(__file__)  # if console

        manual_path = os.path.join(base_path, "manual.pdf")
        if not os.path.exists(manual_path):
            self.warn_heading.setText(self.warn_heading.text() + "Manual file not found." + '\n')
            if dev_mode: print("Manual file not found.")
            return
        self.warn_heading.setText(self.warn_heading.text() + "Manual opened." + '\n')
        if dev_mode: print("Manual opened.")
        os.startfile(manual_path)
        # to include manual:
        # pyinstaller --add-data "manual.pdf;." main.py

    def about_popup(self):
        QMessageBox.information(self, 'About CRISP', 'CRISP: Cremated Remains Inference of Sex Probabilities \nV1.0 '
                                                     '(publishing date: DD/MM/YYYY)\nSoftware Licence: GPL-3.0\n\nBased'
                                                     ' on the technology described in: Waltenberger et al., submitted.'
                                                     ' CRISP: Cremated Remains Inference of Sex Probabilities – A '
                                                     'Software for Bayesian Sex Estimation in Human Cremated Remains,'
                                                     ' Journal, DOI\n\nPlease visit the softwares homepage for updates:'
                                                     ' to https://github.com/quadraBits/CRISP\n\nPlease report any bugs to '
                                                     'crisp.helpdesk@gmail.com\n\nDisclaimer: \nThis software is provided'
                                                     ' "as is", without warranty of any kind, express or implied, '
                                                     'including but not limited to the warranties of merchantability, '
                                                     'fitness for a particular purpose and noninfringement. In no '
                                                     'event shall the authors be liable for any claim, damages or '
                                                     'other liability arising from the use of this software.\nThe '
                                                     'tool is intended for academic and research purposes. It does '
                                                     'not replace professional osteological expertise. Users must '
                                                     'ensure the correct interpretation and contextualization of the '
                                                     'results.\nBy using this software, you accept the terms of this '
                                                     'disclaimer.\n\n©Sophie Beitel (CBits), Lukas Waltenberger '
                                                     '(University of Vienna)')

    def cite_popup(self):
        QMessageBox.information(self, 'Citation of this Software', 'Please cite this software as follows:\nWaltenberger'
                                                                   ' et al., submitted. CRISP: Cremated Remains '
                                                                   'Inference of Sex Probabilities – A Software for '
                                                                   'Bayesian Sex Estimation in Human Cremated Remains, '
                                                                   'Journal, DOI')

    def export(self, page_index):
        # Input legality check:
        while True:
            site = ""
            grave = ""
            individual = ""
            notes = ""
            ok = False
            try:
                dialog = InputDialog()
                ok = dialog.exec()
                if ok:
                    site, grave, individual, notes = dialog.get_inputs()
            except Exception as e:
                print(e)
            if ok:
                if site == "" or grave == "" or individual == "":
                    QMessageBox.warning(self, "Warning", "Site, Grave number, and Individual must be provided!",
                                        QMessageBox.StandardButton.Ok)
                    continue
                elif any(c in site for c in '-/ß*?') or any(c in grave for c in '-/ß*?') or any(c in individual for c in '-/ß*?'):
                    QMessageBox.warning(self, "Warning", "Site, Grave number, and Individual cannot have any of the "
                                                         "following characters: -, /, ß, *, ?",
                                        QMessageBox.StandardButton.Ok)
                    continue
                else:
                    break
            else:
                return

        if dev_mode: print(page_index)
        title_line = site + ": Grave " + grave + ", individual " + individual
        results = f"Male: {self.export_data[page_index][1][0] * 100:.2f}%, Female: {self.export_data[page_index][1][1] * 100:.2f}%"
        warnings = f"{self.export_data[page_index][2]}"
        plots = self.export_data[page_index][3]
        disclaimer = f"Made with {title} V-{version}. {self.copyright_notice}"
        used_elements = ""
        for measurement in self.export_data[page_index][0]:
            if self.export_data[page_index][0][measurement] is not None:
                used_elements = used_elements + str(measurement) + ": " + str(self.export_data[page_index][0][measurement]) + "mm\n"

        # number of rows in warning text
        i = 0
        for char in warnings:
            if char == '\n':
                i += 1

        if dev_mode: print(i)

        # Get a place to save the case in
        directory_name = QFileDialog.getExistingDirectory(self, "Save File in", "C:\\")
        dir_path = ""

        if directory_name:
            dir_path = Path(directory_name)

        # join the directory path and file name together to get the final path to file
        file_name = f"{site}_{grave}_{individual}"
        file_type = ".pdf"
        save_path = os.path.join(dir_path, file_name+file_type)
        if dev_mode: print(save_path)

        j = 0
        while True:
            j += 1
            temp_filename = file_name
            try:
                with open(save_path, 'r') as reader:  # if the file already exists
                    reader.close()
                temp_filename = temp_filename + str(j) + file_type
                save_path = os.path.join(dir_path, temp_filename)
            except Exception:
                break

        try:
            with PdfPages(save_path) as pdf:
                first_page = plt.figure(figsize=(8.27, 11.69)) # DIN A4
                first_page.clear()
                first_page.text(0.05, 0.97, disclaimer, weight='regular', size=8, ha="left")
                first_page.text(0.5, 0.92, title_line, weight='bold', size=24, ha="center")
                first_page.text(0.1, 0.89, "Results", weight='bold', size=14, ha="left", va="top")
                first_page.text(0.1, 0.87, results, weight='regular', size=14, ha="left", va="top")

                if i <= 26:
                    first_page.text(0.1, 0.82, "Warnings", weight='bold', size=12, ha="left", va="top")
                    first_page.text(0.1, 0.80, warnings, weight='regular', size=12, ha="left", va="top")
                    first_page.text(0.1, 0.38, "Used Measurements", weight='bold', size=12, ha="left", va="top")
                    first_page.text(0.1, 0.36, used_elements, weight='regular', size=12, ha="left", va="top")
                    pdf.savefig(first_page)
                    plt.close()
                    if notes != "":
                        temp_page = plt.figure(figsize=(8.27, 11.69))  # DIN A4
                        temp_page.clear()
                        temp_page.text(0.05, 0.97, disclaimer, weight='regular', size=8, ha="left")
                        temp_page.text(0.1, 0.93, "Additional Notes (optional)", weight='bold', size=12, ha="left", va="top")
                        temp_page.text(0.1, 0.91, notes, weight='regular', size=12, ha="left", va="top", wrap=True)
                        pdf.savefig(temp_page)
                        plt.close()
                else:
                    first_page.text(0.1, 0.84, "Warnings", weight='bold', size=12, ha="left", va="top")
                    first_page.text(0.1, 0.82, warnings, weight='regular', size=12, ha="left", va="top")
                    pdf.savefig(first_page)
                    plt.close()
                    temp_page = plt.figure(figsize=(8.27, 11.69))  # DIN A4
                    temp_page.clear()
                    temp_page.text(0.05, 0.97, disclaimer, weight='regular', size=8, ha="left")
                    temp_page.text(0.1, 0.93, "Used Measurements", weight='bold', size=12, ha="left", va="top")
                    temp_page.text(0.1, 0.91, used_elements, weight='regular', size=12, ha="left", va="top")
                    if notes != "":
                        temp_page.text(0.1, 0.51, "Additional notes", weight='bold', size=12, ha="left",va="top")
                        temp_page.text(0.1, 0.49, notes, weight='regular', size=12, ha="left", va="top", wrap=True)
                    pdf.savefig(temp_page)
                    plt.close()

                temp_plt = deepcopy(plots)
                #plots.set_size_inches(8.27, 11.69)
                temp_plt.text(0.05, 0.97, disclaimer, weight='regular', size=8, ha="left")
                #plots.text(0.1, 0.93, "Plots:", weight='bold', size=12, ha="left", va="top")
                pdf.savefig(temp_plt)
                plt.close()
                # todo: auto open pdf
        except Exception as e:
            print(e)


if __name__ == '__main__':
    app = QApplication([])  # only ever create one of those

    # with open(resource_path("style.qss"), "r", encoding="utf-8") as f:
    #     style = f.read()
    # fire_path = resource_path("fire.png")
    # style = style.replace("fire.png", fire_path.replace("\\", "/"))
    # app.setStyleSheet(style)

    app.setStyleSheet(Path(resource_path("style.qss")).read_text().replace("fire.png", resource_path("fire.png").replace("\\", "/")))

    app.setWindowIcon(QIcon(resource_path("icon_fire.ico")))

    window = MainWindow()

    sys.exit(app.exec())  # event loop start

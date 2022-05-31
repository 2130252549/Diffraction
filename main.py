# 2022\5\28 by GuoqingWu
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from Diffraction import MyDiffraction
from My_Ui import Ui_MainWindow


class MyApp(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.diff = MyDiffraction()
        self.fig()

    def fig(self):
        # figure 2D
        mpl2d = self.ui.mplwidget2D.canvas
        mpl2d.axes.clear()
        screen_ZZ = self.diff.get_screen()
        mpl2d.axes.plot_surface(self.diff.screen_XX, self.diff.screen_YY, screen_ZZ)
        mpl2d.draw()

    @pyqtSlot("int")
    def on_slider_lam_valueChanged(self, value):
        self.ui.SpinBox_lam.setValue(value)
        self.diff.change_lam(lam=value)
        self.fig()

    @pyqtSlot("int")
    def on_slider_z_valueChanged(self, value):
        self.ui.SpinBox_z.setValue(value)
        self.diff.change_z(z=value)
        self.fig()

    @pyqtSlot("int")
    def on_slider_scale_1_valueChanged(self, value):
        self.ui.SpinBox_scale_1.setValue(value/10)
        hole_name = self.diff.hole_name
        self.diff.change_hole(hole_name=hole_name, scale1=value)
        self.fig()

    @pyqtSlot("int")
    def on_slider_scale_2_valueChanged(self, value):
        self.ui.SpinBox_scale_2.setValue(value/10)
        hole_name = self.diff.hole_name
        self.diff.change_hole(hole_name=hole_name, scale2=value)
        self.fig()

    @pyqtSlot("int")
    def on_SpinBox_lam_valueChanged(self, value):
        self.ui.slider_lam.setValue(value)

    @pyqtSlot("double")
    def on_SpinBox_z_valueChanged(self, value):
        self.ui.slider_z.setValue(value)

    @pyqtSlot("double")
    def on_SpinBox_scale_1_valueChanged(self, value):
        self.ui.slider_scale_1.setValue(value*10)

    @pyqtSlot("double")
    def on_SpinBox_scale_2_valueChanged(self, value):
        self.ui.slider_scale_2.setValue(value*10)

    @pyqtSlot('QString')
    def on_comboBox_hole_name_currentTextChanged(self, string):
        self.diff.change_hole(hole_name=string)
        self.fig()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    my_app = MyApp()
    my_app.show()
    sys.exit(app.exec_())

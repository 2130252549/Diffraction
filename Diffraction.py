# 2022\5\27 by GuoqingWu
import numpy as np
import matplotlib.pyplot as plt


class MyDiffraction(object):
    def __init__(self, lam=532, z=15, hole_name='round', scale1=0.8, scale2=0.3):
        # Ui_MainWindow.__init__(self)
        # 定义光的基本性质
        self.lam_unit = 10 ** (-9)  # nm
        self.lam = lam * self.lam_unit
        self.k = 2 * np.pi / self.lam
        # 定义观察屏与挡光屏的间距
        self.z = z  # m

        # 定义挡光板plate
        self.plate_shape = np.array([100, 100])
        self.plate_unit = 10**(-4)   # 0.1mm
        self.plate_start = np.array([-1 * self.plate_unit, -1 * self.plate_unit])
        self.plate_end = np.array([1 * self.plate_unit, 1 * self.plate_unit])
        plate_x = np.linspace(self.plate_start[0], self.plate_end[0], self.plate_shape[0])
        plate_y = np.linspace(self.plate_start[1], self.plate_end[1], self.plate_shape[1])
        self.plate_XX, self.plate_YY = np.meshgrid(plate_x, plate_y)

        # 定义观察屏screen
        self.screen_shape = np.array([100, 100])
        self.screen_unit = 10**(-2)  # cm
        self.screen_start = np.array([-1 * self.screen_unit, -1 * self.screen_unit])
        self.screen_end = np.array([1 * self.screen_unit, 1 * self.screen_unit])
        screen_x = np.linspace(self.screen_start[0], self.screen_end[0], self.screen_shape[0])
        screen_y = np.linspace(self.screen_start[1], self.screen_end[1], self.screen_shape[1])
        self.screen_XX, self.screen_YY = np.meshgrid(screen_x, screen_y)

        # 通光孔hole
        self.hole = self.change_hole(hole_name=hole_name, scale1=scale1, scale2=scale2)
        self.hole_name = hole_name

        # 计算准备
        self.screen_R2 = np.square(self.screen_XX) + np.square(self.screen_YY)
        self.plate_R2 = np.square(self.plate_XX) + np.square(self.plate_YY)

    # 更新波长lam
    def change_lam(self, lam):
        self.lam = lam
        self.k = 2 * np.pi / self.lam

    # 更新光程矩阵
    def change_z(self, z):
        self.z = z

    # 更新光孔形状hole
    def change_hole(self, hole_name, scale1=0.8, scale2=0.3):
        hole_name = hole_name
        temp_hole = np.zeros(self.plate_shape)
        ones = np.ones(self.plate_shape)
        i = np.arange(self.plate_shape[0]) - self.plate_shape[0] / 2
        j = np.arange(self.plate_shape[1]) - self.plate_shape[1] / 2
        I, J = np.meshgrid(i, j)
        if hole_name == 'rectangle':
            length = scale2/2 * self.plate_shape[0] * ones
            width = scale1/2 * self.plate_shape[1] * ones
            temp_hole = 1 - ((np.abs(I) > length) + (np.abs(J) > width))
        elif hole_name == 'round':
            Radius = scale1 / 2 * min(self.plate_shape) * ones
            radius = scale2 / 2 * min(self.plate_shape) * ones
            R = np.sqrt(np.square(I) + np.square(J))
            temp_hole = 1 - ((R > Radius) + (R < radius))
        elif hole_name == 'triangle':
            ...
        return temp_hole

    # 使用到波长lam、光程矩阵calculate_RR、光孔hole，计算衍射图
    def get_screen(self, way='fresnel'):

        # fft2算,正确的算法计算量太大，只能将就着算错误的了
        if True:
            # times = self.screen_unit/self.plate_unit
            # shape = times*self.screen_shape
            # step = shape/self.screen_shape
            if way == 'fresnel':    # 菲涅尔近似
                # U = np.fft.fft2(self.hole * np.exp(1j*self.k*self.plate_R2/(2*self.z)), s=shape)
                # U = U[::step[0], ::step[1]]
                U = np.fft.fft2(self.hole * np.exp(1j*self.k*self.plate_R2/(2*self.z)), s=self.screen_shape) # 错误算法\doge
                had = np.exp(1j*self.k*self.z)/(1j*self.lam*self.z)*np.exp(1j*self.k*self.screen_R2/(2*self.z)) #中间变量，衍射公式的头部系数
                screen_ZZ = had * np.fft.fftshift(U)
            elif way == 'fraunhofer':  # 夫琅和费近似，远场近似
                # U = np.fft.fft2(self.hole, s=shape)
                # U = U[::step[0], ::step[1]]
                U = np.fft.fft2(self.hole, s=self.screen_shape) #错误算法\doge
                had = np.exp(1j*self.k*self.z)/(1j* self.lam*self.z)*np.exp(1j*self.k*self.screen_R2/(2*self.z)) # 中间变量，衍射公式的头部系数
                screen_ZZ = had * np.fft.fftshift(U)
            else: # 傍轴近似
                # 准备plate_delta
                plate_delta = np.prod(self.plate_end - self.plate_start) / np.prod(self.plate_shape)
                # 光程计算准备calculate_RR
                # 拍扁各个meshgrid成一维的，然后将screen与plate扩展成二维，相减得差
                plate_X = np.reshape(self.plate_XX, np.prod(self.plate_shape))
                plate_Y = np.reshape(self.plate_YY, np.prod(self.plate_shape))
                screen_X = np.reshape(self.screen_XX, np.prod(self.screen_shape))
                screen_Y = np.reshape(self.screen_YY, np.prod(self.screen_shape))
                # 将screen与plate扩展成二维
                temp_plate_XX, temp_screen_XX = np.meshgrid(plate_X, screen_X)
                temp_plate_YY, temp_screen_YY = np.meshgrid(plate_Y, screen_Y)
                # 相减得差
                calculate_XX = temp_plate_XX - temp_screen_XX
                calculate_YY = temp_plate_YY - temp_screen_YY
                calculate_ZZ = self.z * np.ones(np.shape(calculate_XX))
                # 计算相互的距离矩阵
                mutual_RR = np.sqrt(np.square(calculate_XX) + np.square(calculate_YY) + np.square(calculate_ZZ))

                phase = np.exp(1j * self.k * mutual_RR)
                U = self.hole.reshape(np.prod(self.plate_shape))
                had = -1j / (2 * self.lam * self.lam_unit * self.z)
                screen_ZZ = np.abs(had * np.dot(phase, U) * plate_delta).reshape(self.screen_shape)

        # 硬算
        if False:
            # 准备plate_delta
            plate_delta = np.prod(self.plate_end - self.plate_start) / np.prod(self.plate_shape)
            # 光程计算准备calculate_??
            plate_X = np.reshape(self.plate_XX, np.prod(self.plate_shape))
            plate_Y = np.reshape(self.plate_YY, np.prod(self.plate_shape))
            screen_X = np.reshape(self.screen_XX, np.prod(self.screen_shape))
            screen_Y = np.reshape(self.screen_YY, np.prod(self.screen_shape))
            temp_plate_XX, temp_screen_XX = np.meshgrid(plate_X, screen_X)
            temp_plate_YY, temp_screen_YY = np.meshgrid(plate_Y, screen_Y)
            calculate_XX = temp_plate_XX - temp_screen_XX
            calculate_YY = temp_plate_YY - temp_screen_YY

            if way == 'fresnel':   # 菲涅尔近似
                phase = np.exp(1j * self.k / (2 * self.z) * (np.square(calculate_XX) + np.square(calculate_YY)))
                U = self.hole.reshape(np.prod(self.plate_shape))
                had = -1j * np.exp(1j * self.z * self.k) / (self.lam * self.z)
                screen_ZZ = np.abs(had * np.dot(phase, U) * plate_delta).reshape(self.screen_shape)
            elif way == 'fraunhofer': # 夫琅和费近似
                phase = np.exp(1j * self.k / self.k * (temp_screen_YY * temp_plate_YY + temp_screen_XX * temp_plate_XX))
                U = self.hole.reshape(np.prod(self.plate_shape))
                had = -1j * np.exp(1j * self.z * self.k) / (self.lam * self.z)
                screen_ZZ = np.abs(had * np.dot(phase, U) * plate_delta).reshape(self.screen_shape)
            else:  # 傍轴近似
                calculate_ZZ = self.z * np.ones(np.shape(calculate_XX))
                mutual_RR = np.sqrt(np.square(calculate_XX) + np.square(calculate_YY) + np.square(calculate_ZZ))

                phase = np.exp(1j * self.k * mutual_RR)
                U = self.hole.reshape(np.prod(self.plate_shape))
                had = -1j / (2 * self.lam * self.lam_unit * self.z)
                screen_ZZ = np.abs(had * np.dot(phase, U) * plate_delta).reshape(self.screen_shape)

        return screen_ZZ

    def getRGB(self, maxPix=1, gamma=1):
        # dWave为波长；maxPix为最大值；gamma为调教参数
        lam = self.lam
        waveArea = [380, 440, 490, 510, 580, 645, 780]
        minusWave = [0, 440, 440, 510, 510, 645, 780]
        deltWave = [1, 60, 50, 20, 70, 65, 35]
        for p in range(len(waveArea)):
            if lam < waveArea[p]:
                break

        pVar = abs(minusWave[p] - lam) / deltWave[p]
        rgbs = [[0, 0, 0], [pVar, 0, 1], [0, pVar, 1], [0, 1, pVar],
                [pVar, 1, 0], [1, pVar, 0], [1, 0, 0], [0, 0, 0]]

        # 在光谱边缘处颜色变暗
        if (lam >= 380) & (lam < 420):
            alpha = 0.3 + 0.7 * (lam - 380) / (420 - 380)
        elif (lam >= 420) & (lam < 701):
            alpha = 1.0
        elif (lam >= 701) & (lam < 780):
            alpha = 0.3 + 0.7 * (780 - lam) / (780 - 700)
        else:
            alpha = 0  # 非可见区

        return [maxPix * (c * alpha) ** gamma for c in rgbs[p]]


if __name__ == '__main__':
    a = MyDiffraction()
    a.change_hole(hole_name='rectangle')
    fig = plt.figure()
    ax = plt.axes()
    screen_ZZ = a.get_screen()

    color = a.getRGB()
    ax.contourf(a.screen_XX, a.screen_YY, screen_ZZ)
    plt.show()

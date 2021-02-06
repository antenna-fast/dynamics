# sys
import time

# 计算
import numpy as np

# 可视化
import cv2
import matplotlib.pyplot as plt

"""
框架：

利用能量守恒，
根据弹簧弹力与重力，计算实时加速度，有了速度就会产生阻力，也会影响力  初始化一个速度即可
总之，就是要计算对合力，规定好正方向
"""

if __name__ == "__main__":
    print("物理仿真...")

    p_max = 400  # 最大位移
    img = np.zeros((p_max, 200, 3))  # h w
    img[1, 1] = [1, 1, 1]
    # print(img[1,1].shape)

    # 初始化弹簧长度与位置
    spring_length = 100.0  # m
    spring_now = 113  # m
    # delta_spring = spring_length - spring_now  # 力的方向向下为正 此时作用在物体上向上
    sprint_K = 10  # 劲度系数   N/m  不必再考虑正负

    # 模型参数 初始化
    m_obj = 12  # Kg
    m_speed_last = 0  # 上一时刻的速度
    m_speed_now = 0
    m_posi = spring_now  # 初始位置 与弹簧坐标系一致

    # 世界参数
    world_g = 10  # Nm^2

    posi_list = []
    # while True:  # 要实现完美的积分，这里还需要实时环境
    for i in range(8000):  # 要实现完美的积分，这里还需要实时环境
        # 模拟周期
        time_start = time.time()
        # time.sleep(0.001)
        time_end = time.time()
        # delta_time = time_end - time_start
        delta_time = 0.01

        # print("main while running ... ")

        # 计算当前的弹簧力
        delta_spring = spring_length - spring_now  # 伸缩程度 向上的力出来是负的
        spring_force = delta_spring * sprint_K  # 弹力 向下为正

        # 考虑摩擦力  作负功
        f_fri = -1 * m_speed_now  # 假设和速度平方成 比

        # 作用于物体上的合力
        m_force = m_obj * world_g + spring_force + f_fri  # spring_force向下为正 所以+

        # 物体的加速度  F=ma
        m_a = m_force / m_obj

        # 物体速度
        # 对加速度积分 在很短的时间内，假设是匀加速
        m_speed_now = m_speed_now + m_a * delta_time  # 周期

        # 物体位置
        # 对速度积分 
        spring_now = spring_now + m_speed_now * delta_time

        print("物体位置：{0}".format(spring_now))

        # 可视化
        # img[int(spring_now*10 - 800), 1] = [200, 0, 0]
        # cv2.imshow('cv', img)
        # cv2.waitKey(1)

        posi_list.append(spring_now)

    plt.scatter([i for i in range(len(posi_list))], posi_list)
    plt.show()

cv2.destroyAllWindows()

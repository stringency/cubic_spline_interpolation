import time
import numpy as np
import sympy
from sympy import symbols, plot_implicit, Eq
from fractions import Fraction
import matplotlib.pyplot as plt

'''
程序名称：三次样条插值算法程序
程序功能：解决三种三次样条插值问题
程序作者：Yaung
'''


# 四舍五入函数
def round_up(n, m):
    n = str(n)
    if len(n) - n.index(".") - 1 == m + 1:
        n += "01"
    n = float(n)
    return np.round(n, m)


while True:
    # 界面展示
    print("\t**********第一类固定边界(输入:1)")
    print("\t\tS'(x0)=f0'\tS'(xn)=fn'")
    print("\t**********第二类自由边界(输入:2)")
    print("\t\tS''(x0)=f0''\tS''(xn)=fn''")
    print("\t**********第三类非节点边界(输入:3)")
    print("\t\tlimSp(x0+)=limSp(xn-)\tp=0,1,2")
    print("\t**********退出程序(输入:4)")

    # 选项输入
    choice = eval(input('请输入你的选项数字:'))
    if choice == 4:
        exit()  # 退出程序
    # 输入数据的个数
    N = eval(input('请输入数据的个数:'))
    arr = input('请输入xk的所有值(每个值用空格隔开):')
    X = np.array([float(i) for i in arr.split()])
    arr = input('请输入每个xk所对应的函数值f(xk)(每个值用空格隔开):')
    Y = np.array([float(i) for i in arr.split()])
    C = np.array([0, 0])
    if choice != 3:
        arr = input('请输入两个边界条件(每个值用空格隔开):')
        C = np.array([float(i) for i in arr.split()])
    '''
    测试

    第二类
    >>
    4
    1 2 4 5
    1 3 4 2
    0 0
    3
    <<
    4.25

     '''
    # 基础公式
    # 计算h
    H = np.array([])
    for i in range(0, N - 1):
        H = np.r_[H, X[i + 1] - X[i]]
    # 计算U
    U = np.array([np.max])
    for i in range(1, N - 1):
        U = np.r_[U, round_up(H[i - 1] / (H[i] + H[i - 1]), 6)]
    # 计算R
    R = np.array([np.max])
    for i in range(1, N - 1):
        R = np.r_[R, round_up(H[i] / (H[i] + H[i - 1]), 6)]
    # 计算G
    G = np.array([3 * (Y[1] - Y[0]) / H[0] - H[0] / 2 * C[0]])  # 一开始第一个先按照第二类初始化
    for i in range(1, N - 1):
        # print(3*(U[0,i]*(Y[i+1]-Y[i])+R[0,i]*(Y[i]-Y[i-1])))
        G = np.r_[G, 3 * (U[i] * (Y[i + 1] - Y[i]) / H[i] + R[i] * (Y[i] - Y[i - 1]) / H[i - 1])]

    # 边界类型判断
    if choice == 1:
        # 第一类固定边界条件
        # 求解方程组
        A1 = np.array([[]])
        for i in range(1, N - 1):
            Ai = np.array([])
            Ai = np.r_[Ai, [0 for j in range(i - 2)]]
            if i > 1:
                Ai = np.r_[Ai, R[i]]
            Ai = np.r_[Ai, 2]
            if i != N - 2:
                Ai = np.r_[Ai, U[i]]
            Ai = np.r_[Ai, [0 for j in range(N - 2 - Ai.size)]]
            if i == 1:
                A1 = np.c_[A1, [Ai]]
            else:
                A1 = np.r_[A1, [Ai]]

        b1 = np.array([G[1] - R[1] * C[0]])
        b1 = np.r_[b1, [G[i] for i in range(2, N - 2)]]
        b1 = np.r_[b1, G[N - 2] - U[N - 2] * C[1]]
        M = np.array([C[0]])
        M = np.r_[M, np.linalg.solve(A1, b1)]
        M = np.r_[M, C[1]]
    elif choice == 2:
        # 第二类自由边界条件
        # 补充最后一个G
        G = np.r_[G, 3 * (Y[N - 1] - Y[N - 2]) / H[N - 2] + H[N - 2] / 2 * C[1]]
        # 解方程组求M
        A2 = np.array([[2, 1]])
        A2 = np.c_[A2, [[0 for i in range(N - 2)]]]
        for i in range(1, N - 1):
            Ai = np.array([])
            Ai = np.r_[Ai, [0 for j in range(i - 1)]]
            Ai = np.r_[Ai, [R[i], 2, U[i]]]
            Ai = np.r_[Ai, [0 for j in range(N - Ai.size)]]
            A2 = np.r_[A2, [Ai]]
        # A2 = np.r_[A2,[0 for i in range(N-2)]]
        A2 = np.r_[A2, [np.r_[[0 for i in range(N - 2)], [1, 2]]]]

        b2 = np.array([G[i] for i in range(N)])
        M = np.array(np.linalg.solve(A2, b2))
    elif choice == 3:
        # 第三类非节点边界条件9
        # 新增U，R，G的最后一个值
        U = np.r_[U, H[N - 2] / (H[0] + H[N - 2])]
        R = np.r_[R, H[0] / (H[0] + H[N - 2])]
        G = np.r_[G, 3 * (U[N - 1] * (Y[1] - Y[0]) / H[0] + R[N - 1] * (Y[N - 1] - Y[N - 2]) / H[N - 2])]

        # 解方程组求M
        A3 = np.array([[]])
        for i in range(1, N):
            Ai = np.array([])
            if i == N - 1:
                Ai = np.r_[Ai, U[N - 1]]
                Ai = np.r_[Ai, [0 for j in range(i - 3)]]
            else:
                Ai = np.r_[Ai, [0 for j in range(i - 2)]]
            if i > 1:
                Ai = np.r_[Ai, R[i]]
            Ai = np.r_[Ai, 2]
            if i != N - 1:
                Ai = np.r_[Ai, U[i]]
            if i == 1:
                Ai = np.r_[Ai, [0 for j in range(N - 2 - Ai.size)]]
                Ai = np.r_[Ai, R[1]]
            else:
                Ai = np.r_[Ai, [0 for j in range(N - 1 - Ai.size)]]
            if i == 1:
                A3 = np.c_[A3, [Ai]]
            else:
                A3 = np.r_[A3, [Ai]]
        b3 = np.array([G[i] for i in range(1, N)])
        M = np.array(np.linalg.solve(A3, b3))
        M = np.r_[M[N - 2], M]

    # 求出全部表达式
    x = sympy.symbols("x")  # 申明未知数"x"

    S = np.array([])
    for i in range(X.size - 1):
        S = np.r_[S, [(H[i] + 2 * (x - X[i])) / np.power(H[i], 3) * np.power(x - X[i + 1], 2) * Y[i] + (
                    H[i] - 2 * (x - X[i + 1])) / np.power(H[i], 3) * np.power(x - X[i], 2) * Y[i + 1] + (
                                  x - X[i]) * np.power(x - X[i + 1], 2) / np.power(H[i], 2) * M[i] + (
                                  x - X[i + 1]) * np.power(x - X[i], 2) / np.power(H[i], 2) * M[i + 1]]]

    while True:
        # 输入预测值
        x1 = eval(input('请输入需要预测的值:'))

        xl = 0
        xlid = 0
        xr = 0
        xrid = 0
        for i in range(X.size):
            if X[i] > x1:
                xr = X[i]
                xrid = i
                xl = X[i - 1]
                xlid = i - 1
                break
        y = (H[xlid] + 2 * (x - X[xlid])) / np.power(H[xlid], 3) * np.power(x - X[xrid], 2) * Y[xlid] + (
                    H[xlid] - 2 * (x - X[xrid])) / np.power(H[xlid], 3) * np.power(x - X[xlid], 2) * Y[xrid] + (
                        x - X[xlid]) * np.power(x - X[xrid], 2) / np.power(H[xlid], 2) * M[xlid] + (
                        x - X[xrid]) * np.power(x - X[xlid], 2) / np.power(H[xlid], 2) * M[xrid]
        y1 = y.evalf(subs={x: x1})

        # 打印数据
        print("方程组的解为：")
        print(M)
        print("三次样条插值的表达式为：")
        print(S)

        # 打印预测值
        print("预测值为：")
        print(y1)

        # 画图
        picture = plt.figure()
        # plt.ion()
        plt.scatter(X, Y, marker='.', c='b')
        # plt.pause(0.01)

        # 画出预测值
        plt.scatter(x1, y1, marker='.', c='r')
        # plt.pause(0.01)

        # 画函数曲线
        for i in range(S.size):
            XX = np.arange(X[i], X[i + 1], 0.01)
            XX = np.array(XX)
            YY = np.array([])
            for j in range(XX.size):
                Z = S[i]
                K = Z.evalf(subs={x: XX[j]})
                YY = np.r_[YY, K]
            plt.plot(XX, YY, color='k')
            # plt.pause(0.01)

        # plt.pause(0.01)
        # plt.ioff()  # 关闭interactive mode
        plt.show(block=True)
        tmpFlag = eval(input('输入\'1\'继续预测，输入\'2\'重新执行程序。'))
        if tmpFlag != 1:
            plt.close()
            break
        plt.close()
    tmpFlag = eval(input('输入\'1\'继续程序，输入\'2\'退出程序。'))
    if tmpFlag != 1:
        break


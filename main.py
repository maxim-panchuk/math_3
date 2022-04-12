from math import sqrt, log, exp
import numpy as np
import matplotlib.pyplot as plt

FILE_IN = "iofiles/input.txt"


def define_minor(matrix, i, j):
    n = len(matrix)
    return [[matrix[row][col] for col in range(n) if col != j] for row in range(n) if row != i]


def define_determ(matrix):
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    det = 0
    sgn = 1
    for j in range(n):
        det += sgn * matrix[0][j] * define_determ(define_minor(matrix, 0, j))
        sgn *= -1
    return det


def define_mean(dots, f):
    return sqrt(define_s(dots, f) / len(dots))


def define_s(dots, func):
    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]
    return sum([(func(x[i]) - y[i]) ** 2 for i in range(n)])


def define_pirson(dots, func):
    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    av_x = sum(x) / n
    av_y = sum(y) / n

    chislit = sum([(x[i] - av_x) * (func(y[i]) - av_y) for i in range(n)])
    znam = sqrt(sum([(x[i] - av_x) ** 2 for i in range(n)]) * sum([(func(y[i]) - av_y) ** 2 for i in range(n)]))

    return chislit / znam


def lin_appr(dots):
    data = {}

    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    sx = sum(x)
    sxx = sum(xi ** 2 for xi in x)
    sy = sum(y)
    sxy = sum(x[i] * y[i] for i in range(n))

    d = define_determ([[sxx, sx], [sx, n]])
    d1 = define_determ([[sxy, sx], [sy, n]])
    d2 = define_determ([[sxx, sxy], [sx, sy]])

    try:
        a = d1 / d
        b = d2 / d
    except ZeroDivisionError:
        return None
    data['a'] = a
    data['b'] = b

    func = lambda z: a * z + b

    data['func'] = func
    data['str_func'] = "f = ax * b"
    data['s'] = define_s(dots, func)
    data['mean'] = define_mean(dots, func)
    data['pirson'] = define_pirson(dots, func)
    return data


def pol_2_appr(dots):
    data = {}

    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    sx = sum(x)
    sx2 = sum([xi ** 2 for xi in x])
    sx3 = sum([xi ** 3 for xi in x])
    sx4 = sum([xi ** 4 for xi in x])
    sy = sum(y)
    sxy = sum([x[i] * y[i] for i in range(n)])
    sx2y = sum([(x[i] ** 2) * y[i] for i in range(n)])

    d = define_determ([[n, sx, sx2],
                       [sx, sx2, sx3],
                       [sx2, sx3, sx4]])
    d1 = define_determ([[sy, sx, sx2],
                        [sxy, sx2, sx3],
                        [sx2y, sx3, sx4]])
    d2 = define_determ([[n, sy, sx2],
                        [sx, sxy, sx3],
                        [sx2, sx2y, sx4]])
    d3 = define_determ([[n, sx, sy],
                        [sx, sx2, sxy],
                        [sx2, sx3, sx2y]])

    try:
        a = d3 / d
        b = d2 / d
        c = d1 / d
    except ZeroDivisionError:
        return None
    data['a'] = a
    data['b'] = b
    data['c'] = c

    func = lambda z: a * (z ** 2) + b * z + c

    data['func'] = func
    data['str_func'] = "f = ax**2 + bx + c"
    data['s'] = define_s(dots, func)
    data['mean'] = define_mean(dots, func)

    return data


def pol_3_appr(dots):
    data = {}

    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    sx = sum(x)
    sx2 = sum([xi ** 2 for xi in x])
    sx3 = sum([xi ** 3 for xi in x])
    sx4 = sum([xi ** 4 for xi in x])
    sx5 = sum([xi ** 5 for xi in x])
    sx6 = sum([xi ** 6 for xi in x])
    sy = sum(y)
    sxy = sum([x[i] * y[i] for i in range(n)])
    sx2y = sum([(x[i] ** 2) * y[i] for i in range(n)])
    sx3y = sum([(x[i] ** 3) * y[i] for i in range(n)])

    d = define_determ([[n, sx, sx2, sx3],
                       [sx, sx2, sx3, sx4],
                       [sx2, sx3, sx4, sx5],
                       [sx3, sx4, sx5, sx6]])
    d1 = define_determ([[sy, sx, sx2, sx3],
                        [sxy, sx2, sx3, sx4],
                        [sx2y, sx3, sx4, sx5],
                        [sx3y, sx4, sx5, sx6]])
    d2 = define_determ([[n, sy, sx2, sx3],
                        [sx, sxy, sx3, sx4],
                        [sx2, sx2y, sx4, sx5],
                        [sx3, sx3y, sx5, sx6]])
    d3 = define_determ([[n, sx, sy, sx3],
                        [sx, sx2, sxy, sx4],
                        [sx2, sx3, sx2y, sx5],
                        [sx3, sx4, sx3y, sx6]])
    d4 = define_determ([[n, sx, sx2, sy],
                        [sx, sx2, sx3, sxy],
                        [sx2, sx3, sx4, sx2y],
                        [sx3, sx4, sx5, sx3y]])

    try:
        a = d4 / d
        b = d3 / d
        c = d2 / d
        q = d1 / d
    except ZeroDivisionError:
        return None

    data['a'] = a
    data['b'] = b
    data['c'] = c
    data['q'] = q

    func = lambda z: a * (z ** 3) + b * (z ** 2) + c * z + q

    data['func'] = func
    data['str_func'] = "f = ax**3 + bx**2 + c*x + q"
    data['s'] = define_s(dots, func)
    data['mean'] = define_mean(dots, func)

    return data


def exp_appr(dots):
    data = {}
    n = len(dots)
    x = [dot[0] for dot in dots]
    y = []
    for dot in dots:
        if dot[1] <= 0:
            return None
        y.append(dot[1])
    linear_y = [log(y[i]) for i in range(n)]
    linear_result = lin_appr([(x[i], linear_y[i]) for i in range(n)])

    a = exp(linear_result['b'])
    b = linear_result['a']
    data['a'], data['b'] = a, b

    func = lambda z: a * exp(b * z)

    data['func'] = func
    data['str_func'] = "f = a*e^(b*x)"
    data['s'] = define_s(dots, func)
    data['mean'] = define_mean(dots, func)

    return data


def log_appr(dots):
    data = {}

    n = len(dots)
    x = []
    for dot in dots:
        if dot[0] <= 0:
            return None
        x.append(dot[0])
    y = [dot[1] for dot in dots]

    lin_x = [log(x[i]) for i in range(n)]
    lin_result = lin_appr([(lin_x[i], y[i]) for i in range(n)])

    a = lin_result['a']
    b = lin_result['b']
    data['a'] = a
    data['b'] = b

    func = lambda z: a * log(z) + b

    data['func'] = func
    data['str_func'] = "fi = a*ln(x) + b"
    data['s'] = define_s(dots, func)
    data['mean'] = define_mean(dots, func)

    return data


def pow_appr(dots):
    data = {}

    n = len(dots)
    x = []
    for dot in dots:
        if dot[0] <= 0:
            return None
        x.append(dot[0])
    y = []
    for dot in dots:
        if dot[1] <= 0:
            return None
        y.append(dot[1])

    lin_x = [log(x[i]) for i in range(n)]
    lin_y = [log(y[i]) for i in range(n)]
    lin_result = lin_appr([(lin_x[i], lin_y[i]) for i in range(n)])

    a = exp(lin_result['b'])
    b = lin_result['a']
    data['a'] = a
    data['b'] = b

    func = lambda z: a * (z ** b)
    data['func'] = func

    data['str_func'] = "fi = a*x^b"

    data['s'] = define_s(dots, func)

    data['mean'] = define_mean(dots, func)

    return data


def get_data_console():
    data = {'dots': []}

    print("\nВведите координаты через пробел, каждая точка - с новой строки")
    print("Для обозначения конца ввода - введите 'END'")
    while True:
        try:
            line = input().strip()
            if line == 'END':
                if len(data['dots']) < 2:
                    raise AttributeError
                break
            dot = tuple(map(float, line.split()))
            if len(dot) != 2:
                raise ValueError
            data['dots'].append(dot)
        except ValueError:
            print("Введите точку снова в формате x y")
        except AttributeError:
            print("Должно быть больше одной точки")
    return data


def get_data_file():
    data = {'dots': []}

    with open(FILE_IN, 'rt', encoding='UTF-8') as fin:
        try:
            for line in fin:
                dot = tuple(map(float, line.strip().split()))
                if len(dot) != 2:
                    raise ValueError
                data['dots'].append(dot)
            if len(data['dots']) < 2:
                raise AttributeError
        except (ValueError, AttributeError):
            return None
    return data


def get_data(input_way):
    if input_way == 'f':
        data = get_data_file()
        if data is None:
            print("\nДанные некорректны")
            exit(1)
    else:
        data = get_data_console()
    return data


def plot(x, y, plot_x, plot_ys, labels):
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.plot(1, 0, marker=">", ms=5, color='k',
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, marker="^", ms=5, color='k',
            transform=ax.get_xaxis_transform(), clip_on=False)

    plt.plot(x, y, 'o')
    for i in range(len(plot_ys)):
        plt.plot(plot_x, plot_ys[i], label=labels[i])

    plt.legend()
    plt.show()


def main():
    print("\tЛабораторная работа №14")
    print("\tВариант №9")
    print("\tАппроксимация функций")

    print("\nДанные из файла - (f). Данные из консоли - (c)")
    input_way = input("Режим ввода: ")
    while input_way != 'f' and input_way != 'c':
        print("Введите (f) - для ввода из файла или (c) - для ввода с консоли")
        input_way = input("Режим ввода: ")

    data = get_data(input_way)

    answers = []
    dots = data['dots']

    temp_answers = [lin_appr(dots),
                    pol_2_appr(dots),
                    pol_3_appr(dots),
                    exp_appr(dots),
                    log_appr(dots),
                    pow_appr(dots)]
    for answer in temp_answers:
        if answer is not None:
            answers.append(answer)
    print("\n\n%20s%20s" % ("Вид функции", "Ср. отклонение"))
    print("-" * 40)
    for answer in answers:
        print("%20s%20.4f" % (answer['str_func'], answer['mean']))
    x = np.array([dot[0] for dot in data['dots']])
    y = np.array([dot[1] for dot in data['dots']])
    plot_x = np.linspace(np.min(x), np.max(x), 100)
    plot_y = []
    labels = []
    for answer in answers:
        plot_y.append([answer['func'](x) for x in plot_x])
        labels.append(answer['str_func'])
    plot(x, y, plot_x, plot_y, labels)

    best_answer = min(answers, key=lambda z: z['mean'])
    print("\nНаилучшая аппроксимирующая функция.")
    print(f" {best_answer['str_func']}, где")
    print(f"  a = {round(best_answer['a'], 4)}")
    print(f"  b = {round(best_answer['b'], 4)}")
    print(f"  c = {round(best_answer['c'], 4) if 'c' in best_answer else '-'}")
    print(f"  d = {round(best_answer['q'], 4) if 'q' in best_answer else '-'}")

    r = temp_answers[0]['pirson']
    print("\nКоэффициент Пирсона:")
    print(temp_answers[0]['pirson'])

    if r < 0.3:
        print("Связь слабая")
    elif 0.3 <= r < 0.5:
        print("Связь умеренная")
    elif 0.5 <= r < 0.7:
        print("Связь заметная")
    elif 0.7 <= r < 0.9:
        print("Связь высокая")
    elif 0.9 <= r < 0.99:
        print("Связь правомерна")

main()

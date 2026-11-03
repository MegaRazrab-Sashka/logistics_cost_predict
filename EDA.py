from pprint import pprint
import matplotlib.pyplot as plt


with open('n_research.txt') as f:
    lines = f.readlines()[1:]
    a = []
    delta = []
    for line in lines:
        line = line.strip().split('\t')
        a.append([line[0], line[-1]])
    for i in range(1, len(lines)):
        line1 = [float(x) for x in lines[i - 1].strip().split('\t')]
        line2 = [float(x) for x in lines[i].strip().split('\t')]
        delta.append(line1[-1] - line2[-1])
    mi1 = min(a, key=lambda x: x[1])
    ma1 = max(a, key=lambda x: x[1])
    pprint(mi1)
    pprint(ma1)

    pprint(delta)
    x1 = [x for x in range(20, 510, 10)]
    fig, ax1 = plt.subplots()
    ax1.plot(x1, delta, label = 'Падение MAE')
    ax1.set_xlabel("Количество деревьев")
    ax1.legend()

    # ax2 = ax1.twinx()
    # print('\n' * 10)
    # print([x[-1] for x in a[1:]])
    # ax2.plot(x1, [x[-1] for x in a[1:]], color='red', label='MAE')
    # ax2.set_yticks([2.6, 2.7, 2.8, 2.9])
    x2 = [x for x in range(10, 510, 10)]
    y2 = [float(x[-1]) for x in a]
    print(y2)
    fig2, ax2 = plt.subplots()
    ax2.scatter(x2, y2, label='MAE')
    ax2.set_xlabel("Количество деревьев")
    plt.show()



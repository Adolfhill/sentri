import matplotlib.pyplot as plt


def drawAndSave(data : list, fileName = "./test.png"):
    fig=plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(data)
    fig.savefig(fileName)
    plt.show()

if __name__ == '__main__':
    drawAndSave([1,3,9,8,4,5])
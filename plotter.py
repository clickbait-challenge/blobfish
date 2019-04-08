import os
import matplotlib
import matplotlib.pyplot as plt


class Plotter:

    @staticmethod
    def plotError(epochs_plot, errors, ts_error, de, m, lrate, statesize, path):
        plt.plot(epochs_plot, errors, color="blue", label="error TR")
        plt.plot(epochs_plot, ts_error, color="red", label="error VL", linestyle="-.")
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.legend(loc='upper right', frameon=False)

        path = path+"plots/"
        if not os.path.exists(path):
            os.makedirs(path)
        file = path + "dropout"+str(de)+"_momentum"+str(m)+"_lrate"+str(lrate)+"_gru"+str(statesize)+"__LOSS.png"
        plt.savefig(file)
        plt.clf()

    @staticmethod
    def plotAccuracy(epochs_plot, accuracy, ts_accuracy, de, m, lrate, statesize, path):
        plt.plot(epochs_plot, accuracy, color="blue", label="accuracy TR")
        plt.plot(epochs_plot, ts_accuracy, color="red", label="accuracy VL", linestyle="-.")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='lower right', frameon=False)

        path = path+"plots/"
        if not os.path.exists(path):
            os.makedirs(path)
        file = path + "dropout"+str(de)+"_momentum"+str(m)+"_lrate"+str(lrate)+"_gru"+str(statesize)+"__ACCURACY.png"
        plt.savefig(file)
        plt.clf()

    @staticmethod
    def plotPrecision(epoch_count, precision, precisionVal, dropout_embedding, m, lrate, g_u, path):
        plt.plot(epoch_count, precision, color="blue", label="Precision TR")
        plt.plot(epoch_count, precisionVal, color="red", label="Precision VL", linestyle="-.")
        plt.xlabel("epochs")
        plt.ylabel("precision")
        plt.legend(loc='lower right', frameon=False)

        path = path+"plots/"
        if not os.path.exists(path):
            os.makedirs(path)
        file = path + "dropout"+str(dropout_embedding)+"_momentum"+str(m)+"_lrate"+str(lrate)+"_gru"+str(g_u)+"__PRECISION.png"
        plt.savefig(file)
        plt.clf()

    @staticmethod
    def plotRecall(epoch_count, recall, recallVal, dropout_embedding, m, lrate, g_u, path):
        plt.plot(epoch_count, recall, color="blue", label="Recall TR")
        plt.plot(epoch_count, recallVal, color="red", label="Recall VL", linestyle="-.")
        plt.xlabel("epochs")
        plt.ylabel("recall")
        plt.legend(loc='lower right', frameon=False)

        path = path + "plots/"
        if not os.path.exists(path):
            os.makedirs(path)
        file = path + "dropout" + str(dropout_embedding) + "_momentum" + str(m) + "_lrate" + str(lrate) + "_gru" + str(
            g_u) + "_RECALL.png"
        plt.savefig(file)
        plt.clf()

    @staticmethod
    def plotFOne(epoch_count, f1, f1Val, dropout_embedding, m, lrate, g_u, path):
        plt.plot(epoch_count, f1, color="blue", label="F1 TR")
        plt.plot(epoch_count, f1Val, color="red", label="F1 VL", linestyle="-.")
        plt.xlabel("epochs")
        plt.ylabel("f1")
        plt.legend(loc='lower right', frameon=False)

        path = path + "plots/"
        if not os.path.exists(path):
            os.makedirs(path)
        file = path + "dropout" + str(dropout_embedding) + "_momentum" + str(m) + "_lrate" + str(lrate) + "_gru" + str(
            g_u) + "_F1.png"
        plt.savefig(file)
        plt.clf()



    @staticmethod
    def plotError_noval(epochs_plot, errors, de, m, lrate, statesize, path):
        plt.plot(epochs_plot, errors, color="blue", label="error TR")
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.legend(loc='upper right', frameon=False)

        path = path+"plots/"
        if not os.path.exists(path):
            os.makedirs(path)
        file = path + "dropout"+str(de)+"_momentum"+str(m)+"_lrate"+str(lrate)+"_gru"+str(statesize)+"__LOSS.png"
        plt.savefig(file)
        plt.clf()

    @staticmethod
    def plotAccuracy_noval(epochs_plot, accuracy, de, m, lrate, statesize, path):
        plt.plot(epochs_plot, accuracy, color="blue", label="accuracy TR")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='lower right', frameon=False)

        path = path+"plots/"
        if not os.path.exists(path):
            os.makedirs(path)
        file = path + "dropout"+str(de)+"_momentum"+str(m)+"_lrate"+str(lrate)+"_gru"+str(statesize)+"__ACCURACY.png"
        plt.savefig(file)
        plt.clf()

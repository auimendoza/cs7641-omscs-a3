from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_scree(label, method, ver, n_components=None):
    print("plot scree...")
    n = n_components
    plt.plot(range(ver.shape[0]), ver, '.-')
    plt.plot(range(ver.shape[0]), np.cumsum(ver), '.-')
    plt.suptitle("%s %s Scree Plot" % (label, method))
    plt.title("n_components = %s" % ("None" if n_components is None else str(n_components)))
    plt.ylabel("Proportion of Variance Explained")
    plt.xlabel("Principal Component")
    plt.legend(["Variance", "Cumulative Variance"], loc="best")
    plt.gcf()
    filename = '%s-%s-%d-scree.png' % (label.lower(), method, n)
    plt.savefig(filename)
    plt.close()

def plot_re(label, method, mse, nc):
    print("plot re...")
    plt.plot(nc, mse, '.-')
    plt.ylabel('Reconstruction Error')
    plt.xlabel('n components')
    plt.title('%s %s Reconstruction Error' % (label, method))
    plt.gcf()
    plt.savefig(('%s %s reconstructionerror.png' % (label, method)).replace(" ", "-"), bbox_inches='tight')
    plt.close()

# reference:
# https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
def biplot(label, method, score, coeff, labels):
    print("biplot...")
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("%s %s Biplot" % (label, method))
    plt.grid()
    plt.gcf()
    plt.savefig("%s-%s-biplot.png" % (label, method))
    plt.close()

def plot_first_images(firstimages, nc, method, label):
    print("plot %s reconstructed images..." % (method))
    fig=plt.figure(figsize=(8,8))
    columns = 4
    rows = 4
    for i in range(1, len(firstimages)+1):
        img = np.array(firstimages[i-1]).reshape((28,28))
        ax = fig.add_subplot(rows, columns, i)
        ax.title.set_text("Original" if i-1 == 0 else "nc=%d" % (nc[i-2]))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap='gray')
    plt.suptitle("Original vs Reconstructed Images (%s)" % (method))    
    plt.gcf()
    plt.savefig(('%s %s reconstucted.png' % (label, method)).replace(" ", "-"))
    plt.close()

def plot_pdiff(label, method, pdiffms, pdiffstds, n_components):
    print("plot pdiff...")
    plt.plot(n_components, pdiffms, '.-')
    plt.fill_between(n_components, pdiffms - pdiffstds,
                     pdiffms + pdiffstds, alpha=0.1)
    plt.xlabel("n_components")
    plt.ylabel("% difference pairwise distances")
    plt.title("%s %s Pairwise Distance Differences" % (label, method))
    plt.gcf()
    plt.savefig("%s-%s-pdiff.png" % (label, method))
    plt.close()

from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
"""
========================================
VISUALIZE TENNESSEE EASTMAN VARIABLES
========================================
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import matplotlib.cm as cm
from sammon import sammon
import ipdb
from sklearn.preprocessing import LabelEncoder


class TE():
    """ Tennessee Eastman Simulator Data Reading and Manipulation
    Parameters
    ----------

    Attributes
    ----------
    Each training data file contains 480 rows and 52 columns and
    each testing data file contains 960 rows and 52 columns.
    An observation vector at a particular time instant is given by

    x = [XMEAS(1), XMEAS(2), ..., XMEAS(41), XMV(1), ..., XMV(11)]^T
    where XMEAS(n) is the n-th measured variable and
    XMV(n) is the n-th manipulated variable.
    """
    
    XMEAS = ['Input Feed - A feed (stream 1)'	,       	#	1
        'Input Feed - D feed (stream 2)'	,       	#	2
        'Input Feed - E feed (stream 3)'	,       	#	3
        'Input Feed - A and C feed (stream 4)'	,       	#	4
        'Miscellaneous - Recycle flow (stream 8)'	,	#	5
        'Reactor feed rate (stream 6)'	,                 	#	6
        'Reactor pressure'	,                           	#	7
        'Reactor level'	,                                	#	8
        'Reactor temperature'	,                           	#	9
        'Miscellaneous - Purge rate (stream 9)'	,       	#	10
        'Separator - Product separator temperature'	,	#	11
        'Separator - Product separator level'	,       	#	12
        'Separator - Product separator pressure'	,	#	13
        'Separator - Product separator underflow (stream 10)'	,	#	14
        'Stripper level'	,                           	#	15
        'Stripper pressure'	,                           	#	16
        'Stripper underflow (stream 11)'             	,	#	17
        'Stripper temperature'	,                           	#	18
        'Stripper steam flow'	,                           	#	19
        'Miscellaneous - Compressor work'	,       	#	20
        'Miscellaneous - Reactor cooling water outlet temperature'	,	#	21
        'Miscellaneous - Separator cooling water outlet temperature'	,	#	22
        'Reactor Feed Analysis - Component A'	,	#	23
        'Reactor Feed Analysis - Component B'	,	#	24
        'Reactor Feed Analysis - Component C'	,	#	25
        'Reactor Feed Analysis - Component D'	,	#	26
        'Reactor Feed Analysis - Component E'	,	#	27
        'Reactor Feed Analysis - Component F'	,	#	28
        'Purge gas analysis - Component A'	,	#	29
        'Purge gas analysis - Component B'	,	#	30
        'Purge gas analysis - Component C'	,	#	31
        'Purge gas analysis - Component D'	,	#	32
        'Purge gas analysis - Component E'	,	#	33
        'Purge gas analysis - Component F'	,	#	34
        'Purge gas analysis - Component G'	,	#	35
        'Purge gas analysis - Component H'	,	#	36
        'Product analysis -  Component D'	,	#	37
        'Product analysis - Component E'	,	#	38
        'Product analysis - Component F'	,	#	39
        'Product analysis - Component G'	,	#	40
        'Product analysis - Component H']		#	41
			
    XMV = ['D feed flow (stream 2)'	,                 	#	1
        'E feed flow (stream 3)'	,                 	#	2
        'A feed flow (stream 1)'	,                 	#	3
        'A and C feed flow (stream 4)'	,                 	#	4
        'Compressor recycle valve'	,                 	#	5
        'Purge valve (stream 9)'	,                 	#	6
        'Separator pot liquid flow (stream 10)'	,       	#	7
        'Stripper liquid product flow (stream 11)'	,	#	8
        'Stripper steam valve'	,                           	#	9
        'Reactor cooling water flow'	,                 	#	10
        'Condenser cooling water flow']	#,                 	#	11
        #'Agitator speed']             # constant 50%			12

    def var_category_str(self, featnr):
        '''Returning string with the original category 'XMEAS #' or 'XMV #'
        '''
        if featnr < 41:
            name = 'XMEAS (' + str(featnr+1) + '): '
        else:
            name = 'XMV (' + str(featnr+1-41) + '): '
        return name


    def __init__(self):
        #print('Executing __init__() ....')

        self.Xtrain = None
        self.Xtest = None
        self.featname = self.XMEAS + self.XMV
        self.extendedfeatname = list(self.featname)
        self.numfeat = len(self.featname)
        for i in range(self.numfeat):
            self.extendedfeatname[i] = self.var_category_str(i) + self.featname[i]
        #print('TE.extendedfeatname=', self.extendedfeatname);
        #print('TE.featname=', self.featname); quit()

    def standardize(self):
        print('Data standardization to zero mean and unit variance...')
        X = self.Xtrain
        #print('\nTraining dataset before standardization=\n', X)
        #print('\nTest dataset before standardization=\n', self.Xtest)
        self.meanX = np.mean(X, axis=0)
        # ddof=1 ==> divide by (n-1) --- ddof=0 ==> divide by n
        ddof_std = 0    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.std.html#numpy.std
        self.stdX = X.std(axis=0, ddof=ddof_std)

        #print('Dataset statistic:\n Mean=', meanX, '\nStandard deviation=\n', stdX)
        #minX = X.min(axis=0)
        #maxX = X.max(axis=0)
        #print('Dataset statistic:\nMin=', minX, '\nMax=', maxX )

        self.Xcentered_train = X - self.meanX
        #print('Dataset X=\n', X, '\nDataset centralized Xcentered_train=\n', self.Xcentered_train)
        self.Xstandardized_train = self.Xcentered_train / self.stdX
        #print('Dataset standadized Xstandardized_train=\n', self.Xstandardized_train)

        self.Xcentered_test = self.Xtest - self.meanX
        self.Xstandardized_test = self.Xcentered_test / self.stdX 


    def labelledcsvread(self, filename, delimiter = '\t', fmode='r'):

        print('Reading CSV from file ', filename )
        f = open(filename, fmode)
        reader = csv.reader(f, delimiter=delimiter)
        ncol = len(next(reader)) # Read first line and count columns
        nfeat = ncol-1
        f.seek(0)              # go back to beginning of file
        #print('ncol=', ncol); quit()
        
        x = np.zeros(nfeat)
        X = np.empty((0, nfeat))
        y = []
        for row in reader:
            #print(row)
            for j in range(nfeat):
                x[j] = float(row[j])
                #print('j=', j, ':', x[j])
            X = np.append(X, [x], axis=0)
            label = row[nfeat]
            y.append(label)
            #print('label=', label)
            #quit()
        #print('X.shape=\n', X.shape)#, '\nX=\n', X)
        #print('y=\n', y)
        
        
        # Resubsitution for all methods
        from sklearn.preprocessing import LabelEncoder
        from LabelBinarizer2 import LabelBinarizer2
        lb = LabelBinarizer2()
        Y = lb.fit_transform(y)
        classname = lb.classes_
        #print('lb.classes_=', lb.classes_, '\nY=\n',Y)

        le = LabelEncoder()
        ynum = le.fit_transform(y)
        #print(ynum)
        
        return X, Y, y, ynum, classname


    def read_train_test_pair(self, datadir, fault_num='01', standardize=True):
        ''' Read a training and test pair from the predefined TE datasets and put them
        into the respective data structures
        '''
        ftrain = datadir+'d'+fault_num+'.dat'
        ftest = datadir+'d'+fault_num+'_te.dat'
        self.Xtrain = self.datacsvreadTE(ftrain)
        self.Xtest = self.datacsvreadTE(ftest)
        if standardize:
            self.standardize()

    def plot_condition(self, X, y, classlabel, classname, featname, plot_time_axis=True,
            dropfigfile=None, title=None):
        '''Given a set of patters with class label, plot in 2D.
        If the time axis option is true, plot the postion in time, following
        the order in the data matrix X (first pattern X[0] at t=0
        '''
        if plot_time_axis:
            print ('Generating 2-D plot with time evolution ...')
        else:
            print ('Generating 2-D plot ...')
        #print('X=\n', X.shape, '\nclassname=', classname)
        numclasses = len(classname)
        xlab = featname[0]
        ylab = featname[1]
        fig, ax = plt.subplots(); # Create a figure and a set of subplots
        #colors = 'bry'
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, numclasses)]

        #plot_time_axis=False   # DEBUG
        if plot_time_axis:
            from mpl_toolkits.mplot3d import Axes3D
            title = title + ' 2-D with time evolution'
            zlab = 't'
            ax = Axes3D(fig, azim=-45, elev=30)
            ax.set_title(title)
            ax.set_xlabel(xlab)
            #ax.w_xaxis.set_ticklabels([])
            ax.set_ylabel(ylab)
            #ax.w_yaxis.set_ticklabels([])
            ax.set_zlabel(zlab)
            #ax.w_zaxis.set_ticklabels([])
            toffset = 0
            for i in range(numclasses):
                idx = np.where(y == i)
                numpts = len(idx[0])
                t = np.linspace(0, toffset+numpts-1, numpts)
                #toffset += numpts
                ax.scatter(X[idx, 0], X[idx, 1], t, c=colors[i], label=classname[i])
            #ax.set_zlim(bottom=0, top=toffset+numpts)
        else:
            for i, color in zip(classlabel, colors):
                idx = np.where(y == i)
                plt.scatter(X[idx, 0], X[idx, 1], c=color, label=classname[i],
                        cmap=plt.cm.Paired, edgecolor='black', s=20)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.title(title)
        plt.legend()
        plt.axis('tight')
        if not dropfigfile is None:
            print('Saving figure in ', dropfigfile)
            plt.savefig(dropfigfile, dpi=1200)
        plt.show()


    def plotscatter(self, datafile, feat1, feat2, standardize=True, dropfigfile=None,
            title='Tennessee Eastman: Classes in Feature Space'):
        delimiter = '\t'
        X, Y, y, ynum, classname = self.labelledcsvread(filename=datafile, delimiter=delimiter)
        #print('X=\n',X,'shape=',X.shape,'\nY=\n',Y,'shape=',Y.shape,'y=\n',y,'ynum=\n',ynum,'shape=',ynum.shape,'classname=',classname); quit()

        labels = ynum
        classes = classname
        classlabel = np.unique(ynum)

        X2feat = X[:, [feat1,feat2]] # only two features can be visualized directly

        X = X2feat
        y = ynum

        if standardize:
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X = (X - mean) / std
        #featname = [self.featname[feat1], self.featname[feat2]]
        featname = [self.extendedfeatname[feat1], self.extendedfeatname[feat2]]
        self.plot_condition(X, y, classlabel, classname, featname, plot_time_axis=True,
                dropfigfile=None, title=title)


    def plot_train_test_pair(self, datadir, fault_num='01', feat1=1, feat2=2,
                             standardize=True, plot_time_axis=True, dropfigfile=None, title=None):
        ''' Plot the training and test pair
        '''
        self.read_train_test_pair(datadir=datadir, fault_num=fault_num, standardize=standardize)
        
        if standardize:
            Xtrain = self.Xstandardized_train
            Xtest = self.Xstandardized_test
        else:
            Xtrain = self.Xtrain
            Xtest = self.Xtest
        
        Xtrain = Xtrain[:, [feat1,feat2]]
        Xtest = Xtest[:, [feat1,feat2]]
        ntrain = Xtrain.shape[0]
        ntest = Xtest.shape[0]
        X = np.concatenate((Xtrain, Xtest), axis=0)
        y = np.concatenate((np.zeros(ntrain),np.ones(ntest)))
        classname = ['Normal', 'Fault'+' '+fault_num]
        classlabel = np.array([0, 1])
        featname = [self.extendedfeatname[feat1], self.extendedfeatname[feat2]]

        #print('ntrain=', ntrain, 'ntest=', ntest, 'n=', 'X=\n', X, '\nshape=', X.shape, 'y=\n',y,'shape=',y.shape,) #; quit()
        
        self.plot_condition(X, y, classlabel, classname, featname, plot_time_axis=plot_time_axis,
                dropfigfile=dropfigfile, title=title)
        
        

    def datacsvreadTE(self, filename, delimiter = ' '):

        print('===> Reading TE data from file ', filename, '...')
        f = open(filename, 'rt')
        reader = csv.reader(f, delimiter=delimiter)
        row1 = next(reader)
        ncol = len(row1) # Read first line and count columns
        # count number of non-empty strings in first row
        nfeat = 0
        for j in range(ncol):
            cell = row1[j]
            if cell != '':
                nfeat = nfeat + 1
                #print('%.2e' % float(cell))

        f.seek(0)              # go back to beginning of file
        #print('ncol=', ncol, 'nfeat=', nfeat)
        
        x = np.zeros(nfeat)
        X = np.empty((0, nfeat))
        r = 0
        for row in reader:
            #print(row)
            c = 0
            ncol = len(row)
            for j in range(ncol):
                cell = row[j]
                if cell != '':
                    x[c] = float(cell)
                    #print('r=%4d' % r, 'j=%4d' % j, 'c=%4d' % c, 'x=%.4e' % x[c])
                    c = c + 1
            r = r + 1
            X = np.append(X, [x], axis=0)
            #if r > 0: # DBG
            #    break
        #print('X.shape=\n', X.shape)#, '\nX=\n', X)
        return X

    def filter_vars(self, X, mask):
        return X[:,np.array(mask,dtype=int)], list(np.array(self.extendedfeatname)[mask])

    def signal_plot(self, infile=None, X=None, divide_by_mean=True, dropfigfile=None, title=None, mask=None):

        if not infile is None:
            if X is None:
                print('===> Reading TE data from file ', infile, '...')
                X = self.datacsvreadTE(infile)
            else:
                print('Data X exist. Ignoring infile...')

        featname = self.extendedfeatname
        #print('featname=',featname,'mask=',mask)
        if not mask is None:
            '''
            mask = np.array(mask,dtype=int)
            X = X[:,mask]
            featname = list(np.array(extendedfeatname)[mask])
            '''
            X, featname = self.filter_vars(X, mask)

        n, d = X.shape
        #print(X)
        tsfig = plt.figure(2, figsize=(12,6)) # figsize in inches
        for j in range(d):
            ts = X.T[j,:]
            if divide_by_mean:
                ts = ts / np.mean(ts)
                ts = ts + j
            #print('Feat#', j+1, '=', ts)
            plt.plot(ts, linewidth=0.5)

        if not title is None:
            plt.title(title)
        # Legend ouside plot:
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        #plt.legend(featname, fontsize=7, loc='best')
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', mode='expand')
        #plt.tight_layout(pad=70)
        #plt.legend(featname, fontsize=7, loc='best', bbox_to_anchor=(0.5, 0., 1.0, 0.5))
        plt.legend(featname, fontsize=7, loc='center left', bbox_to_anchor=(0.85, 0.60),
                fancybox=True, shadow=True, ncol=1)
        if not dropfigfile is None:
            print('Saving figure in ', dropfigfile)
            plt.savefig(dropfigfile, dpi=1200)
        plt.show()

    def read_file_by_pandas(self, basefile, num_failure):
        
        file = basefile + ("%02d" % (num_failure,)) + '_te.dat'       

        df = pd.read_csv(file, sep='   ', names=self.extendedfeatname)
        ipdb.set_trace()
        X = df.values
        ynum = np.repeat(num_failure, X.shape[0])
        #Y = np.array(df.values[:,df.values.shape[1]:df.values.shape[1]])

        #le = LabelEncoder()
        #ynum = le.fit_transform(Y)
        
        return X, ynum, ynum, df

    def read_all_files(self, basefile, failure_list):

        X_all = []
        Y_all = []
        ynum_all = []        
        for i in failure_list:
            X, Y, ynum, df = self.read_file_by_pandas(basefile, i)
            X_all.append(X)
            Y_all.append(Y)
            ynum_all.append(ynum)
            if X_all.size == 0:
                X_all = X
                Y_all = Y
                ynum_all = ynum            
            else:
                X_all = np.concatenate((X_all, X), axis=0)
                Y_all = np.concatenate((Y_all, Y), axis=0)
                ynum_all = np.concatenate((ynum_all, ynum), axis=0)
                

        return X_all, Y_all, ynum_all    

    def plot3d(self,X,Y):
        #time = np.array(range(0,X.shape[0]*180,180)).reshape(len(range(0,X.shape[0],180)),1)
        #ipdb.set_trace()
        #for j in range(X.shape[1]):
        #    X[:, j] = (X[:, j] / np.mean(X[:, j])) + j
            #ts = ts + j
            #print('Feat#', j+1, '=', ts)
            #plt.plot(ts, linewidth=0.5)
        
        matrix = np.append(X,Y,axis=1)
        #matrix = np.append(matrix,time,axis=1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        classes = np.unique(Y)
        colors = cm.rainbow(np.linspace(0, 1, len(classes)))
        
        for clas,c in zip(classes,colors):
            elementFromClass = matrix[matrix[:,X.shape[1]] == clas]
            ax.scatter(elementFromClass[:, 0], elementFromClass[:, 1], list(range(0, elementFromClass.shape[0])), color=c, label=clas)
            # print(clas)
        
        plt.legend()
        #ax.title('TSNE projection - 2 components of simulator TENNESSEE data')
        #plt.show()
        

def test1():
    import csv
    
    with open('all.csv') as csvfile:
        
        all_dados = list(csv.reader(csvfile, delimiter="\t"))
        all_dados = np.array(all_dados[1:], dtype=np.str)
        
        X = all_dados[:,0:all_dados.shape[1]-1]
        Y = all_dados[:,all_dados.shape[1]]
    return X, Y


def main(argv):
    print('Executing main() ....')
    
    if len(argv) > 0:
        file=argv[0]
    else:
        file="out/all.csv"

    #X, Y = test1()
    #return        
    
    # datadir = 'out/'
    # fault_num='01'

    # ftrain = datadir+'d'+fault_num+'.dat'
    # ftest = datadir+'d'+fault_num+'_te.dat'

    # te = TE()
    # X = te.datacsvreadTE(ftest)
    # te.signal_plot(infile=None, X=X, divide_by_mean=True, dropfigfile='/tmp/outfig.svg', title='Todas as variaveis'+' \n '+ftest)
    
  
    # feat1 = 17 # First feature
    # feat2 = 43 # Second feature
    # featname = '{'+te.extendedfeatname[feat1] + ',' + te.extendedfeatname[feat2]+'}'

    # te.signal_plot(infile=ftrain, divide_by_mean=False, dropfigfile='/tmp/outfig1.svg',
    #         title='Subconjunto de variaveis: '+featname+' \n '+ftrain, mask=[feat1,feat2])

    # te.signal_plot(infile=ftest, divide_by_mean=False, dropfigfile='/tmp/outfig2.svg',
    #         title='Subconjunto de variaveis: '+featname+' \n '+ftest, mask=[feat1,feat2])

    # te.plot_train_test_pair(datadir, fault_num='01', feat1=feat1, feat2=feat2,
    #         standardize=False, plot_time_axis=True, dropfigfile='/tmp/outfig3.svg', title='Training and test pair')

    # quit()
    

    # csvdatafile = 'out/all.csv'
    # te.plotscatter(csvdatafile, feat1, feat2, standardize=True) #; quit() 
    te = TE()
    #X,Y,ynum,df = te.read_file_by_pandas(file)
    X,Y,ynum = te.read_all_files('data/d', [1,2,4,6])
    ipdb.set_trace()
    
    # ipdb.set_trace()
    #te.signal_plot(infile=None, X=X, divide_by_mean=True, dropfigfile='/tmp/outfig.svg', title='Todas as variaveis'+' \n ')

    #FULL
    #rad_viz = pd.plotting.radviz(df,'Class')
    #plt.title('Radviz projection - all components of simulator TENNESSEE data')
    #plt.show()

    #TSNE
    X_embedded_tsne = TSNE(n_components=2).fit_transform(X)
    #df_test = pd.DataFrame(np.append(X_embedded_tsne,Y,axis=1),columns=['A','B','Class']) 
    #rad_viz = pd.plotting.radviz(df_test,'Class')
    t = range(0, 239940, 180)
    #ipdb.set_trace()

    te.plot3d(X_embedded_tsne,Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X_embedded_tsne[0:333, 0], X_embedded_tsne[0:333, 1], list(range(0, 333, 1))  , marker='^', label="class 1")
    ax.scatter(X_embedded_tsne[333:666, 0], X_embedded_tsne[333:666, 1], list(range(0, 333, 1)), marker='o', label="class 2")
    ax.scatter(X_embedded_tsne[666:999, 0], X_embedded_tsne[666:999, 1], list(range(0, 333, 1)), marker='x', label="class 4")
    ax.scatter(X_embedded_tsne[999:, 0], X_embedded_tsne[999:, 1], list(range(0, 334, 1)), marker='s', label="class 6")
    plt.legend()
    #ax.title('TSNE projection - 2 components of simulator TENNESSEE data')
    #plt.show()    

    #PCA
    # X_embedded_pca = PCA(n_components=2).fit_transform(X)
    #df_test = pd.DataFrame(np.append(X_embedded_pca,Y,axis=1),columns=['A','B','Class'])
    #rad_viz = pd.plotting.radviz(df_test,'Class')
    #plt.title('PCA projection - 2 components of simulator TENNESSEE data')
    #plt.show()

    # t = range(0, 38520, 180)
    #ipdb.set_trace()
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_embedded_tsne[0:44, 0], X_embedded_pca[0:44, 1], t[0:44], marker='^', label="class 1")
    # ax.scatter(X_embedded_tsne[44:88, 0], X_embedded_pca[44:88, 1], t[44:88], marker='o', label="class 2")
    # ax.scatter(X_embedded_tsne[88:133, 0], X_embedded_pca[88:133, 1], t[88:133], marker='x', label="class 4")
    # ax.scatter(X_embedded_tsne[133:, 0], X_embedded_pca[133:, 1], t[133:], marker='s', label="class 6")
    # plt.legend()
    #ax.title('TSNE projection - 2 components of simulator TENNESSEE data')
    # plt.show()

    # Run the Sammon projection
    # [ySammon,E] = sammon(X)

    # Plot
    # plt.scatter(ySammon[target ==0, 0], ySammon[target ==0, 1], s=20, c='r', marker='o',label=np.unique(Y)[0])
    # plt.scatter(ySammon[target ==1, 0], ySammon[target ==1, 1], s=20, c='b', marker='D',label=np.unique(Y)[1])
    # plt.title('Sammon projection of simulator TENNESSEE data')
    # plt.legend(loc=2)
    # plt.show()

    #ipdb.set_trace()

    te.plot_condition(X_embedded_tsne, ynum, np.unique(ynum), np.unique(Y), ['tsne1','tsne2'], plot_time_axis=True, dropfigfile=None, title='Trabalho finalizado')
    
    


if __name__ == "__main__":
    main(sys.argv[1:])

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import sklearn.preprocessing as skpp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sammon import sammon
import datetime



class TE():

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

    def __init__(self):

        self.featname = self.XMEAS + self.XMV
        self.extendedfeatname = list(self.featname)
    
    def var_category_str(self, featnr):
        '''Returning string with the original category 'XMEAS #' or 'XMV #'
        '''
        if featnr < 41:
            name = 'XMEAS (' + str(featnr+1) + '): '
        else:
            name = 'XMV (' + str(featnr+1-41) + '): '
        return name


    def read_fault_train_test_file(self, datadir, fault_num, standardize, features=[]):

        fmt_fault_num = ("%02d" % (fault_num,))
        ext = ".dat"

        ftrain = os.path.join(datadir, 'd' + fmt_fault_num + ext)
        ftest = os.path.join(datadir, 'd' + fmt_fault_num + "_te" + ext)

        df_train = pd.read_csv(ftrain, delim_whitespace=True, names=self.extendedfeatname)
        df_test = pd.read_csv(ftest, delim_whitespace=True, names=self.extendedfeatname)

        ss = skpp.StandardScaler()
        
        if standardize:
            X_train = ss.fit_transform(df_train)
            X_test = ss.transform(df_test)
        else:
            X_train = df_train.values
            X_test = df_test.values
            
        if len(features) < 3:
            return X_train, X_test
        else:
            return X_train[:,features],X_test[:,features]

    def read_concat_multiple_faults(self, datadir, faults, standardize, features=[]):

        X_train_all = []
        X_test_all = []
        Y_train_all = []
        Y_test_all = []
        for fault_num in faults:
            X_train, X_test = self.read_fault_train_test_file(datadir, fault_num, standardize, features)
            Y_train = np.repeat(fault_num, X_train.shape[0]).reshape(-1,1)
            Y_test = np.repeat(fault_num, X_test.shape[0]).reshape(-1,1)

            X_train_all.append(X_train)
            X_test_all.append(X_test)
            Y_train_all.append(Y_train)
            Y_test_all.append(Y_test)
        
        return np.concatenate(X_train_all, axis=0), np.concatenate(Y_train_all, axis=0), np.concatenate(X_test_all, axis=0), np.concatenate(Y_test_all, axis=0)
    

    def plot3d(self, X, Y, title="Example", features_name= ["Feature 1"," Feature 2"] ):
        
        X_aug = np.append(X, Y, axis=1)
        fig = plt.figure()
        
        ax = fig.add_subplot(111, projection='3d')
        classes = np.unique(Y)
        colors = cm.gnuplot(np.linspace(0, 1, len(classes)))
        
        for clazz, c in zip(classes, colors):
            elementFromClass = X_aug[X_aug[:,-1] == clazz]
            ax.scatter(elementFromClass[:, 0], elementFromClass[:, 1], list(range(0, elementFromClass.shape[0])), color=c, label=clazz)
        
        plt.legend()

        ax.set_zlabel("Time")
        ax.set_xlabel(features_name[0])
        ax.set_ylabel(features_name[1])
        #plt.xlabel(features_name[0])
        #plt.ylabel(features_name[1])
        plt.title(title)
        plt.show()

    def plot3d_save_images(self, X, Y, title="Example", features_name= ["Feature 1"," Feature 2"] ):
        X_aug = np.append(X, Y, axis=1)
        fig = plt.figure()
        
        ax = fig.add_subplot(111, projection='3d')
        classes = np.unique(Y)
        colors = cm.gnuplot(np.linspace(0, 1, len(classes)))
        
        color_classes = {}
        elements_class = {}
        for clazz, c in zip(classes, colors):
            elements_class[clazz] = X_aug[X_aug[:,-1] == clazz]
            color_classes[clazz] = c

        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        if not os.path.isdir("out_img/"):
            os.mkdir("out_img")

        os.mkdir('out_img/'+time)

        
        for i in range(0,int(X.shape[0]/len(classes))):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')     
            for k,v in elements_class.items():
                ax.scatter(v[i][0], v[i][1], i, color=color_classes[k],label=k)
            
            # PLOTA O PONTO ANTERIOR
            # if i > 0:
            #     for k,v in elements_class.items():
            #         ax.scatter(v[i-1][0], v[i-1][1], i-1, color=color_classes[k],label=k)

            # PLOTA TODOS OS PONTOS ANTERIORES AO PONTO ATUAL
            for k,v in elements_class.items():
                ax.scatter(v[:i,0], v[:i,1], range(0,i), color=color_classes[k])

            ax.set_zlabel("Time")
            ax.set_xlabel(features_name[0])
            ax.set_ylabel(features_name[1])
            ax.set_xlim(np.amin(X[:,0]*1.2),np.amax(X[:,0]*1.2))
            ax.set_ylim(np.amin(X[:,1]*1.2),np.amax(X[:,1]*1.2))
            ax.set_zlim([0,X.shape[0]/len(elements_class)])
            plt.legend()
            plt.title(title)
            plt.savefig("out_img/"+time+"/"+str(i)+".png")
            plt.close(fig)

    def view_tsne(self, X, Y, save_img=False):

        X_ = TSNE(n_components=2).fit_transform(X)
        self.plot3d(X_, Y, title="Simultaneous 2-D with time evolution - TSNE",features_name=["tSNE 1","tSNE 2"])
        if save_img:
            self.plot3d_save_images( X_, Y,title="Simultaneous 2-D with time evolution - TSNE",features_name=["tSNE 1","tSNE 2"])


    def view_pca(self, X, Y, save_img=False):
        X_ = PCA(n_components=2).fit_transform(X)
        te.plot3d(X_ ,Y, title="Simultaneous 2-D with time evolution - PCA",features_name=["PCA 1","PCA 2"])
        if save_img:
            self.plot3d_save_images( X_, Y,title="Simultaneous 2-D with time evolution - PCA",features_name=["PCA 1","PCA 2"])

    def view_radviz(self, X, Y, feats):

        names = np.array(self.extendedfeatname)
        
        X_df = pd.DataFrame(X[:, feats] , columns=names[feats])
        Y_df = pd.DataFrame(Y, columns=["Fault"])

        X_ = pd.concat([X_df, Y_df], axis=1)

        pd.plotting.radviz(X_, 'Fault')
        plt.title('Radviz')
        plt.show()

    def view_sammon(self, X, Y):
        
        target = Y.flatten()
        [y,E] = sammon(X, maxiter=20)

        names = np.unique(target)
        colors = cm.gnuplot(np.linspace(0, 1, len(names)))
        for clazz, c in zip(names, colors):
        
            # Plot
            plt.scatter(y[target == clazz, 0], y[target == clazz, 1], s=20, c=c, label=clazz)
       
        
        plt.title('Sammon Plot')
        plt.legend(loc=2)
        plt.show()
        
if __name__ == "__main__":

    faults = [1,2,4]
    te = TE()
    X_train, Y_train, X_test, Y_test = te.read_concat_multiple_faults("data", faults, False, [])
    
    te.view_tsne(X_test, Y_test, True)
    # te.view_pca(X_test, Y_test)
    # te.view_radviz(X_test, Y_test, [0, 20, 41, 31, 6])
    # te.view_sammon(X_test, Y_test)
    
    
    

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
'''
LIKERT SCORE:
                         Strongly Approve Undecided Disapprove Strongly
                          Approve                             Disapprove
Percent checking            13%     43%     21%         13%       10%
Corresponding 1-5 value      1       2       3           4         5
Corresponding sigma value  -1.63   -0.43   +0.43       +0.99     +1.76
'''


class DataSampler(object):
    def __init__(self,chunk, percent=0.7,
                filename = 'dataset/61featuresData.csv'):
        self.chunk_size=chunk
        self.percent = percent
        self.filename=filename
        self.read_dataset()
        self.get_dictionary()
        self.make_training_pairs(percent)
        self.count_batch=0




    def get_train_test(self):
        names = ['xtrain_positive_chunk', 'xtrain_negative_chunk', 'xtest_positive_chunk', 'xtest_negative_chunk']
        s = lambda x: np.shape(x)

        index=[0,0,0,0]
        x_train,x_test=[],[]
        for i in range(4):
            if i<2:

                for B,E in zip(self.behave_dict[names[i]],self.env_dict[names[i]]):
                    x_train.append(np.hstack((B,E)))
                index[i]=len(x_train)
            else:
                for B,E in zip(self.behave_dict[names[i]],self.env_dict[names[i]]):
                    x_test.append(np.hstack((B,E)))
                index[i]=len(x_test)

        x_train,x_test=np.array(x_train),np.array(x_test)
        y_train,y_test=np.ones(x_train.shape[0]),np.ones(x_test.shape[0])
        y_train[index[0]:]=0
        y_test[index[2]:]=0

        x_train=np.concatenate((x_train,x_test))
        y_train=np.concatenate((y_train,y_test))

        x_train, x_test,y_train, y_test=train_test_split(x_train,y_train,test_size=1-self.percent,random_state=42)
        # print(s(x_train),s(y_train))


        return (x_train,y_train),(x_test,y_test)



    def normalize_dataset(self):
        self.behavior_scale=MinMaxScaler(feature_range=(-1.,1.))
        self.environment_scale=MinMaxScaler(feature_range=(-1.,1.))
        self.behavior_scale.fit(self.behavior)
        self.environment_scale.fit(self.environment)
        self.behavior=self.behavior_scale.transform(self.behavior)
        self.environment=self.environment_scale.transform(self.environment)




    def read_dataset(self):
        '''
        Partially reading the rawdata file which is a csv format
        :return: classify the rawdata into 3 groups
        '''
        self.behavior = np.array(pd.read_csv(self.filename, header=0, usecols=np.arange(17, 24)))
        self.environment = np.array(pd.read_csv(self.filename, header=0, usecols=np.arange(24, 61)))
        self.likert_score = np.array(pd.read_csv(self.filename, header=0, usecols=np.arange(16, 17)))
        self.normalize_dataset()





    def get_dictionary(self):
        '''
        classify the dataset based on LIKERT SCORE into two examples : positive and negative
        :return: populate global dictionary namely driving behavior and environment condition
        '''
        positive_index = np.where(self.likert_score <=-0.43)
        negative_index = np.where(self.likert_score >=0.99)
        undecided_index= np.where((self.likert_score>-0.43) & (self.likert_score<0.99))

        self.num_positive=int(len(positive_index[0])/self.chunk_size)*self.chunk_size
        self.num_negative=int(len(negative_index[0])/self.chunk_size)*self.chunk_size
        self.num_undecide = int(len(undecided_index[0]) / self.chunk_size) * self.chunk_size
        positive_index=positive_index[0][:self.num_positive]
        negative_index=negative_index[0][:self.num_negative]
        undecided_index = undecided_index[0][:self.num_undecide]

        self.behave_dict = {'positive': self.behavior[positive_index], 'negative': self.behavior[negative_index]}
        self.env_dict = {'positive': self.environment[positive_index], 'negative': self.environment[negative_index]}
        self.score={'positive':self.likert_score[positive_index],'negative':self.likert_score[negative_index]}
        self.undecided_dict={'behave':self.behavior[undecided_index],'env':self.environment[undecided_index]}


    def get_undecide_data(self):
        # print(self.num_undecide)
        chunk_indices = lambda x, n: list(zip(np.arange(n - 1) * x, np.arange(1, n) * x))
        index_pair=chunk_indices(x=self.chunk_size,n=self.num_undecide//self.chunk_size)
        B,E=self.undecided_dict['behave'],self.undecided_dict['env']

        # print(index_pair)

        data=[]
        for iz in index_pair:
            data.append(np.hstack((B[iz[0]:iz[1]], E[iz[0]:iz[1]])))
        # print(np.shape(data))
        return np.array(data)






    def get_chunk_index(self,percent):
        '''
        :param percentage: [domain:= positive || negativ]e chunk_dict: [domain:= x_train || x_test]
        :return: dictionnary {x_train_positive, x_train_negative, x_test_positive,x_test_negative}
        '''
        chunk = lambda x, n: list(zip(np.arange(n - 1) * x, np.arange(1, n) * x))
        num=[self.num_positive,self.num_negative]
        self.percentage=[int(res*p)for res in num for p in [percent, 1-percent]]
        chunk_dict={i: chunk(self.chunk_size, percent//self.chunk_size)for i, percent in enumerate(self.percentage)}
        return chunk_dict




    def make_training_pairs(self, percent):
        '''
        Generate two dictionaries with 4 parameters based on the percent value
        :param percent: The amount of total data which would be considered as of Training; (1-p) is then Testing
        :return: populate existing dictionaries with new names
        '''
        chunk_dict=self.get_chunk_index(percent)
        chunk_data=lambda data,indices:[data[iz[0]:iz[1]]for i,iz in enumerate(indices)if len(data[iz[0]:iz[1]])>0]

        self.names=['xtrain_positive_chunk','xtrain_negative_chunk','xtest_positive_chunk','xtest_negative_chunk']
        clas_name=['positive','negative']
        order=[ (clas_name[i%2],y) for i,y in enumerate(self.names)  ]

        for i,ord in enumerate(order):
            data=chunk_data(self.behave_dict[ord[0]][:self.percentage[i]],chunk_dict[i])
            self.behave_dict[ord[1]]=data
            data=chunk_data(self.env_dict[ord[0]][:self.percentage[i]],chunk_dict[i])
            self.env_dict[ord[1]]=data
            data=chunk_data(self.score[ord[0]][:self.percentage[i]],chunk_dict[i])
            self.score[ord[1]]=data


        # for i, nam in enumerate(self.names):
        #     print("{} Examples:= Behavior {} Environment {} Score {}".format\
        #           (clas_name[i%2], np.shape(self.behave_dict[nam]), np.shape(self.env_dict[nam]), np.shape(self.score[nam]) ))
        #
        # print(self.behave_dict.keys())

    def get_positive_examples(self):
        '''
        Get only the positive examples from the dictionaries. Behavior dictionary will be conditioned by Environment dict
        :return: x_train,x_test,c_train,c_test
        '''
        return (self.behave_dict[self.names[0]],self.behave_dict[self.names[2]]),(self.env_dict[self.names[0]],self.env_dict[self.names[2]]),\
               (self.score[self.names[0]],self.score[self.names[2]])


    def get_negative_examples(self):
        '''
        Get only the negative examples from the dictionaries. Behavior dictionary will be conditioned by Environment dict
        :return: x_train,x_test,c_train,c_test
        '''
        return (self.behave_dict[self.names[1]], self.behave_dict[self.names[3]]), (
        self.env_dict[self.names[1]], self.env_dict[self.names[3]]),(self.score[self.names[1]], self.score[self.names[3]])

    def get_score_label(self,A):
        '''
        :param A: compute mean Anomaly score
        :return:  Find the nearest likert score index w.r.t. mean
        '''
        mu = np.mean(A)
        mu_v = np.ones(self.liker_sigma.shape) * mu
        dist = abs((self.liker_sigma - mu_v) ** 2)
        indx = np.argmin(dist)
        label_v = indx
        return label_v.astype(int)


    def get_next_batch(self,batch):
        x_train,y_train=self.dataset
        start=self.count_batch
        end=start+batch
        num_x_train=len(x_train)
        batch_x_train=x_train[start:end]
        batch_y_train=y_train[start:end]
        self.count_batch=(end+1)if end<=(num_x_train-batch) else 0
        # print('start: {}-->end {}'.format(start,end))
        return np.array(batch_x_train),np.array(batch_y_train)

if __name__ == '__main__':
    xs=DataSampler(44)
    xs.get_train_test()



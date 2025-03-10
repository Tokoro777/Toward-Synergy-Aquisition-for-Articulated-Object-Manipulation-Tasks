from sklearn.decomposition import PCA
import numpy as np

class SynergyManager:
    def __init__(self, reward_lambda, num_axis, init_poslist, maxn_pos):
        self.poslist = init_poslist
        self.reward_lambda = reward_lambda
        self.num_axis = num_axis
        self.maxn_pos = maxn_pos

        self.axis = np.array([[0.5] * 20] * self.num_axis)  # num_axis=5
        print(self.axis)  # 0.5を20個持つ、(5, 20)が出力

        self.pca = PCA(self.num_axis)

    def set_lambda(self, lambda_c=0.0):
        self.reward_lambda = lambda_c

    def get_lambda(self):
        return self.reward_lambda

    def add_pos(self, pos):  # 要素をリストに追加
        self.poslist.append(pos)

    def add_list(self, list):  # リストをリストに追加. new
        self.poslist.extend(list)

    def set_poslist(self, poslist):
        self.poslist = poslist

    def get_poslist(self):
        if len(self.poslist) > self.maxn_pos:
            self._pop_pos()
        return self.poslist

    def _pop_pos(self):
        return self.poslist.pop(0)

    def calc_pca(self):
        self.pca.fit(self.poslist)

    def calc_transform(self, pos):
        t_pos = self.pca.transform(pos)
        return t_pos

    def calc_inverse(self, pos):
        i_pos = self.pca.inverse_transform(pos)
        return i_pos

    def get_variance_ratio(self):
        return self.pca.explained_variance_ratio_

from typing import Iterator

import torch
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from model.GCN import *
import torch.nn as nn
import random
import copy
import argparse




seed = 15
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser(description='GA_main')

parser.add_argument('--dataset',type=str,default='cora',choices=['cora','polblogs','citeseer'],help='The dataset used for the adversarial attack')
parser.add_argument('--ptb_rate',type=float,default=0.05,choices=[0.05,0.10,0.15,0.20],help='perturbation rate')

args = parser.parse_args()

seed = 15
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = args.dataset

data = Dataset(root='/tmp/', name=dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels

labels = torch.LongTensor(labels)
labels = labels.to(device)
n_labels = int(labels.max().item() + 1)

features = sparse_mx_to_torch_sparse_tensor(features)
features = features.to_dense()
features = features.to(device)

adj = sparse_mx_to_torch_sparse_tensor(adj)
adj = adj.to_dense()

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
print("train_size",len(idx_train))
print("test_size",len(idx_test))
print("val_size",len(idx_val))
# print(idx_val)
# print(idx_test)



class Genetic_Algorithm(nn.Module):

    def __init__(self,adj,idx_train,idx_test,dataset,ptb_rate = args.ptb_rate,device='cuda:0'):
        super(Genetic_Algorithm, self).__init__()
        self.clean_adj =  adj
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.dataset = dataset
        self.ptb_rate = ptb_rate
        self.device = device
        self.zero_idx = []
        self.get_zero_idx()
        self.adj_list = []
        self.fitness_list =[]
        self.accuracy_list = []
        self.parameters_set()

        self.edge_num =None
        self.canditate_edge_num = None
        self.population = None
        self.get_edge_num()


    def parameters_set(self):
        if self.dataset == 'cora':
            self.mut_rate = 0.01
        elif self.dataset =='citeseer':
            self.mut_rate = 0.02
        else:
            self.mut_rate = 0.04

        if self.dataset == 'cora':
            if self.ptb_rate == 0.05:
                self.init_individual_len = 250
                self.select_size = 80
                self.population_size= 800
                self.mut_rate2 = 0.998

            if self.ptb_rate == 0.10:
                self.init_individual_len = 500
                self.select_size = 40
                self.population_size = 400
                self.mut_rate2 = 0.997

            if self.ptb_rate == 0.15:
                self.init_individual_len = 750
                self.select_size = 27
                self.population_size = 270
                self.mut_rate2 = 0.996

            if self.ptb_rate == 0.20:
                self.init_individual_len = 1000
                self.select_size = 20
                self.population_size = 200
                self.mut_rate2 = 0.995

        if self.dataset == 'citeseer':
            if self.ptb_rate == 0.05:
                self.init_individual_len = 180
                self.select_size = 80
                self.population_size = 800
                self.mut_rate2 = 0.9985

            if self.ptb_rate == 0.10:
                self.init_individual_len = 360
                self.select_size = 40
                self.population_size = 400
                self.mut_rate2 = 0.997

            if self.ptb_rate == 0.15:
                self.init_individual_len = 540
                self.select_size = 27
                self.population_size = 270
                self.mut_rate2 = 0.996

            if self.ptb_rate == 0.20:
                self.init_individual_len = 720
                self.select_size = 20
                self.population_size = 200
                self.mut_rate2 = 0.995

        if self.dataset == 'polblogs':
            if self.ptb_rate == 0.05:
                self.init_individual_len = 800
                self.select_size = 10
                self.population_size = 50
                self.mut_rate2 = 0.97

            if self.ptb_rate == 0.10:
                self.init_individual_len = 1650
                self.select_size = 10
                self.population_size = 50
                self.mut_rate = 0.02
                self.mut_rate2 = 0.94

            if self.ptb_rate == 0.15:
                self.init_individual_len = 2460
                self.select_size = 10
                self.population_size = 50
                self.mut_rate = 0.02
                self.mut_rate2 = 0.9

            if self.ptb_rate == 0.20:
                self.init_individual_len = 3300
                self.select_size = 10
                self.population_size = 50
                self.mut_rate = 0.02
                self.mut_rate2 = 0.885



    def forward(self,epoch=40):
        print('正在初始化种群......')
        self.initial_population()

        print(f'种群初始化完毕，共有{self.population_size}个个体')
        print('开始进行遗传算法')

        for e in range(epoch):
            print(f'epoch{e}:')
            self.generate_adj()
            self.compute_fitness()
            selected_population = self.selection()
            new_population = None
            count_limit = 0

            while True:
                 # print(len(new_population))
                parent1,parent2 = self.select_parents(selected_population)
                child1,child2 = self.crossover(parent1,parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                count_limit += 1
                # print(child1.sum())
                population_size = 0

                if child1.sum() <= int(self.edge_num*self.ptb_rate):
                    if new_population is None:
                        new_population = child1.unsqueeze(0)
                    else:
                        new_population = torch.cat((new_population,child1.unsqueeze(0)),dim=0)

                if child2.sum() <= int(self.edge_num*self.ptb_rate):
                    if new_population is None:
                        new_population = child2.unsqueeze(0)
                    else:
                        new_population = torch.cat((new_population, child2.unsqueeze(0)), dim=0)

                if new_population is not None:
                    population_size = new_population.size()[0]
                if count_limit > self.population_size*2:
                    self.population_size = int(self.population_size/2)
                    count_limit = 0
                if population_size >= self.population_size:
                    break


            self.population = new_population

        self.generate_adj()
        self.compute_fitness()
        final_selected = self.selection()
        print(final_selected)

    def select_parents(self,selected_population):
        indices = torch.randperm(selected_population.size(0))

        # 从随机排列的索引中选出前2个，用它们来索引原张量，得到随机选择的2行
        selected_rows = selected_population[indices[:2]]

        return selected_rows[0],selected_population[1]


    def get_zero_idx(self):
        degree = self.clean_adj.sum(dim=1)
        canditate_train_idx = self.idx_train[degree[self.idx_train] < (int(degree.mean()) + 1)]
        candidate_test_idx = self.idx_test[degree[self.idx_test] < (int(degree.mean()) + 1)]
        print("candidate_test_size",len(candidate_test_idx))
        print("candidate_train_size",len(canditate_train_idx))

        for i in range(len(canditate_train_idx)):
            for j in range(len(canditate_train_idx)):
                if j > i and self.clean_adj[self.idx_train[i]][self.idx_train[j]] == 0 \
                        and labels[self.idx_train[i]]!=labels[self.idx_train[j]]:
                    # print((i,j))
                    self.zero_idx.append((self.idx_train[i],self.idx_train[j]))
        for i in canditate_train_idx:
            for j in candidate_test_idx:
                if self.clean_adj[i][j] == 0 and labels[i]!=labels[j]:
                    self.zero_idx.append((i, j))

        print('zero_size:',len(self.zero_idx))




    def initial_population(self):
        # 初始化种群
        n = self.canditate_edge_num
        population= torch.zeros(self.population_size,n,dtype=torch.bool)
        for p in population:
            indices_to_set_to_one = torch.randperm(n)[:self.init_individual_len]
            p[indices_to_set_to_one] = 1
        # print(population.sum(dim=1))
        self.population = population

    def generate_adj(self):
        assert len(self.population) != 0
        print(self.clean_adj.sum())
        self.adj_list.clear()

        for p in self.population:
            adj = torch.clone(self.clean_adj)
            # print('p_size',p.size())


            for idx, value in enumerate(p):
                if value == 1:
                    num1, num2 = self.zero_idx[idx]
                    adj[num1][num2] = 1
                    adj[num2][num1] = 1
            # print(adj.sum()-self.clean_adj.sum())

            self.adj_list.append(adj)




    def compute_fitness(self):

        print("开始计算个体适应度：")
        self.fitness_list.clear()
        self.accuracy_list.clear()

        for adj in self.adj_list:
            adj = adj.to(self.device)
            model = GCN(nfeat=features.shape[1], nhid=16, nclass=n_labels, device=device)
            model = model.to(self.device)
            model.fit(features,adj,labels,self.idx_train,idx_val,train_iters=200,verbose=False)
            model.eval()
            loss, accuracy= model.test(self.idx_test)
            self.fitness_list.append(loss)
            self.accuracy_list.append(accuracy)

        fitness_list = copy.deepcopy(self.fitness_list)
        accuracy_list = copy.deepcopy(self.accuracy_list)

        fitness_list.sort(reverse=True)
        accuracy_list.sort()
        print('fitness_list:')
        print(fitness_list)
        print('accuracy_list:')
        print(accuracy_list)

    def selection(self):

        assert len(self.fitness_list) != 0
        fitness = torch.tensor(self.fitness_list)
        if self.select_size > self.population.size()[0]:
            self.select_size = int(self.population.size()[0]/2)

        values, indices = torch.topk(fitness,self.select_size,largest=True)
        print("Values:\n", values)
        print("Indices:\n", indices)
        select_population = self.population[indices]
        print('select_population_size:',len(select_population))

        return select_population




    def crossover(self,parent1, parent2):
        # 确保父代长度相同且都包含1000个1
        assert len(parent1) == len(parent2) == self.canditate_edge_num


        # 找出两个父代在哪些位置上是可以交换的（即一个为1，一个为0）
        swap_positions = [(i, parent1[i], parent2[i]) for i in range(len(parent1)) if parent1[i] != parent2[i]]

        # 在可以交换的位置上，随机选择一半进行交换
        random.shuffle(swap_positions)
        swap_positions = swap_positions[:len(swap_positions) // 2]  # 只选择一半进行交换

        # 创建后代，初始时与父代相同
        offspring1, offspring2 = torch.clone(parent1), torch.clone(parent2)

        # 在选定的位置上交换基因
        for pos, gene1, gene2 in swap_positions:
            offspring1[pos], offspring2[pos] = gene2, gene1

        return offspring1, offspring2

    # 变异操作
    def mutation(self,individual):

        mutated_individual = individual[:]
        for i in range(self.canditate_edge_num):
            if random.random() < self.mut_rate:
                if mutated_individual[i]==0 and random.random() > self.mut_rate2:
                    mutated_individual[i] = 1
                else:
                    mutated_individual[i] = 0



        return mutated_individual




    def get_edge_num(self):
        self.edge_num = self.clean_adj.sum()/2
        self.canditate_edge_num = len(self.zero_idx)
        print("canditate_edge_num:",self.canditate_edge_num)
        print("edge_num:",self.edge_num)
        print("1_size:",int(self.edge_num*self.ptb_rate))




GA = Genetic_Algorithm(adj,idx_train,idx_test,dataset)
GA()

# adj_indice = get_zero_indice(adj,idx_train,idx_test)
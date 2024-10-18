import copy
import openpyxl
import numpy as np
import os
import copy as cp
from CEC2017MTSO import *
from evo_operator import *
import random
import time
import pandas as pd


class MFLLSO:
    def __init__(self, level,popsize, max_d,taskf1, taskf2, rmp, maxfes):
        self.rmp = rmp
        self.MAXFES = maxfes
        self.fes = 0
        self.pop_size = popsize
        self.max_d = max_d
        self.taskf1 = taskf1
        self.taskf2 = taskf2
        self.task1_num = 0
        self.task2_num = 0
        self.level = level
        self.level_num = self.pop_size//self.level
        self.fine =0.4

    def pop_init(self):
        # 初始化种群
        self.taks1_pop = []
        self.taks2_pop = []
        self.task1_totalfit = 0
        self.task2_totalfit = 0
        for i in range(self.pop_size):
            new_pop = {}
            new_pop['skill_f'] = 1
            new_pop['x'] = np.random.random(self.max_d)
            new_pop['velocity'] = np.zeros(self.max_d)
            new_pop['fit'] = self.taskf1.function(new_pop['x'])
            self.fes += 1
            self.taks1_pop.append(new_pop)
        for i in range(self.pop_size):
            new_pop = {}
            new_pop['skill_f'] = 2
            new_pop['x'] = np.random.random(self.max_d)
            new_pop['velocity'] = np.zeros(self.max_d)
            new_pop['fit'] = self.taskf2.function(new_pop['x'])
            self.fes += 1
            self.taks2_pop.append(new_pop)

        self.taks1_pop.sort(key = lambda x:x['fit'])
        self.taks2_pop.sort(key = lambda x:x['fit'])
    def llso_offspring_gen(self,pop1,pop2):
        offspring = []
        for i in range(self.level_num,self.pop_size):
            better_level = (i//self.level_num)
            new_pop = {}
            k1= np.random.randint(0,better_level*self.level_num)
            k2 = np.random.randint(1, better_level * self.level_num)
            if k1>k2:
                temp = k2
                k2 = k1
                k1 = temp
            elif k1 == k2:
                k1 = np.random.randint(0,k2)
            if np.random.rand()<self.rmp:
                xk1,xk2 = pop2[k1]['x'],pop2[k2]['x']
            else:
                xk1,xk2 = pop1[k1]['x'],pop1[k2]['x']
            r1,r2,r3 = np.random.random((3,self.max_d))
            new_pop['velocity'] = r1*pop1[i]['velocity'] + r2*(xk1-pop1[i]['x']) + r3*self.fine*(xk2-pop1[i]['x'])
            new_pop['velocity']  = np.clip(new_pop['velocity'],-0.2,0.2)
            new_pop['x'] = copy.copy(pop1[i]['x'] + new_pop['velocity'])
            new_pop['x'] = np.clip(new_pop['x'],0,1)
            new_pop['skill_f'] = copy.copy(pop1[i]['skill_f'])
            offspring.append(new_pop)
        if pop1[i]['skill_f']==1:
            for i in range(len(offspring)):
                offspring[i]['fit'] = self.taskf1.function(offspring[i]['x'])
                self.fes+=1
        else:
            for i in range(len(offspring)):
                offspring[i]['fit'] = self.taskf2.function(offspring[i]['x'])
                self.fes += 1
        return offspring
    def llso_optimize(self):
        while self.fes<self.MAXFES:
            offspring1 = self.llso_offspring_gen(self.taks1_pop,self.taks2_pop)
            offspring2 = self.llso_offspring_gen(self.taks2_pop,self.taks1_pop)

            self.taks1_pop = self.taks1_pop + offspring1
            self.taks2_pop = self.taks2_pop + offspring2

            self.taks1_pop.sort(key=lambda x: x['fit'])
            self.taks2_pop.sort(key=lambda x: x['fit'])

            self.taks1_pop = copy.deepcopy(self.taks1_pop[:self.pop_size])
            self.taks2_pop = copy.deepcopy(self.taks2_pop[:self.pop_size])
        best_1 = self.taks1_pop[0]['fit']
        best_2 = self.taks2_pop[0]['fit']
        return best_1,best_2



task_fc = ['CIHS','CIMS','CILS','PIHS','PIMS','PILS','NIHS','NIMS','NILS']
task_bound = {'CIHS':{'task1':[-100,100],'task2':[-50,50]},'CIMS':{'task1':[-50,50],'task2':[-50,50]},\
              'CILS':{'task1':[-50,50],'task2':[-500,500]},'PIHS':{'task1':[-50,50],'task2':[-100,100]},\
              'PIMS':{'task1':[-50,50],'task2':[-50,50]},'PILS':{'task1':[-50,50],'task2':[-0.5,0.5]},\
              'NIHS':{'task1':[-50,50],'task2':[-50,50]},'NIMS':{'task1':[-100,100],'task2':[-0.5,0.5]},\
              'NILS':{'task1':[-50,50],'task2':[-500,500]},}
test_task = ['MTSO2017']
if __name__ == '__main__':
    filename_with_suffix = __file__
    filename_without_suffix = os.path.basename(filename_with_suffix)
    filename = os.path.splitext(filename_without_suffix)[0]+'_'+test_task[1]
    ori_path = './independent_data/' + filename
    xls_savepath = './record_data/' + filename + '.xlsx'
    if not os.path.exists(ori_path):
        os.makedirs(ori_path, exist_ok=True)
    task_num = len(task_fc)
    mfea_record = openpyxl.Workbook()
    sheet = mfea_record.active
    sheet['A1'] = 'test_task'
    sheet['B1'] = 'task1'
    sheet['C1'] = 'task2'
    best_avg_array = np.zeros((task_num, 2))
    record_time_matrix = np.zeros((30, len(task_fc)))
    for i in range(task_num):
        path = ori_path+'/' + task_fc[i] + '.txt'
        task1 = Tasks(task_fc[i], 1)
        task2 = Tasks(task_fc[i], 2)
        for j in range(30):

            mfllso = MFLLSO(4,100,50,task1,task2,0.1,1e+05)
            start_time = time.time()
            mfllso.pop_init()
            best_t1,best_t2 = mfllso.llso_optimize()
            end_time = time.time()
            record_time_matrix[j][i] = end_time - start_time
            best_avg_array[i][0] += best_t1
            best_avg_array[i][1] += best_t2
            f_file = open(path, 'a+')
            f_file.write(f'{best_t1} {best_t2}\n')
            f_file.close()
    record_time_matrix = pd.DataFrame(record_time_matrix)
    pd_time_path = ori_path + '/' + 'MTSO2017_time.xlsx'
    record_time_matrix.to_excel(pd_time_path, index=False)
    best_avg_array /= 30
    for i in range(task_num):
        sheet.append([task_fc[i], best_avg_array[i][0], best_avg_array[i][1]])
    mfea_record.save(xls_savepath)
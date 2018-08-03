# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import csv

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    # print '%s正确率：%.3f%%' % (tip, acc_rate)
    return acc_rate
def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

#读取训练数据


data_train = xgb.DMatrix('cleandata1.csv')

param = {'max_depth': 3, 'eta': 0.2, 'silent': 0,'objective':'reg:logistic'}
bst = xgb.train(param, data_train, num_boost_round=100)#, evals=watch_list)
#data_test = xgb.DMatrix(x_test)
data_test = xgb.DMatrix('tap_fun_test1.csv')
y_hat = bst.predict(data_test)
y_train_hat=bst.predict(data_train)


#读取训练数据
traindata = pd.read_csv('cleandata.csv')

                                                                                                                                                                                                                                                                                                                                                                                                                                              #x = traindata[['wood_add_value', 'wood_reduce_value', 'stone_add_value', 'stone_reduce_value', 'ivory_add_value','ivory_reduce_value','meat_add_value','meat_reduce_value','magic_add_value','magic_reduce_value','infantry_add_value','infantry_reduce_value','cavalry_add_value','cavalry_reduce_value','shaman_add_value','shaman_reduce_value','wound_infantry_add_value','wound_infantry_reduce_value','wound_cavalry_add_value','wound_cavalry_reduce_value','wound_shaman_add_value','wound_shaman_reduce_value','general_acceleration_add_value','general_acceleration_reduce_value','building_acceleration_add_value','building_acceleration_reduce_value','reaserch_acceleration_add_value','reaserch_acceleration_reduce_value','training_acceleration_add_value','training_acceleration_reduce_value','pvp_battle_count', 'pvp_lanch_count', 'pvp_win_count', 'pve_battle_count' ,'pve_lanch_count', 'pve_win_count', 'avg_online_minutes', 'pay_price', 'pay_count']]
x = traindata[[
               'stone_add_value',
               'stone_reduce_value',
               'ivory_add_value',
               'ivory_reduce_value',
               'infantry_add_value',
               'infantry_reduce_value',
               'cavalry_add_value',
               'cavalry_reduce_value',
               'shaman_add_value',
               'shaman_reduce_value',
               'wound_infantry_add_value',
               'wound_infantry_reduce_value',
               'wound_cavalry_add_value',
               'wound_cavalry_reduce_value',
               'wound_shaman_add_value',
               'wound_shaman_reduce_value',
               'general_acceleration_add_value',
               'general_acceleration_reduce_value',
               'building_acceleration_add_value',
               'building_acceleration_reduce_value',
               'reaserch_acceleration_add_value',
               'reaserch_acceleration_reduce_value',
               'training_acceleration_add_value',
               'training_acceleration_reduce_value',
               'bd_training_hut_level',
               'bd_healing_lodge_level',
               'bd_stronghold_level',
               'bd_outpost_portal_level',
               'bd_barrack_level',
               'bd_healing_spring_level',
               'bd_dolmen_level',
               'bd_guest_cavern_level',
               'bd_warehouse_level',
               'bd_watchtower_level',
               'bd_magic_coin_tree_level',
               'bd_hall_of_war_level',
               'bd_market_level',
               'bd_hero_gacha_level',
               'bd_hero_strengthen_level',
               'bd_hero_pve_level',
               'sr_scout_level',
               'sr_training_speed_level',
               'sr_infantry_tier_2_level',
               'sr_cavalry_tier_2_level',
               'sr_shaman_tier_2_level',
               'sr_infantry_atk_level',
               'sr_cavalry_atk_level',
               'sr_shaman_atk_level',
               'sr_infantry_tier_3_level',
               'sr_cavalry_tier_3_level',
               'sr_shaman_tier_3_level',
               'pvp_battle_count',
               'pvp_lanch_count',
               'pvp_win_count',
               'pve_battle_count' ,
               'pve_lanch_count',
               'pve_win_count',
               'avg_online_minutes',
               'pay_price',
               'pay_count']]
y = traindata['continue_pay']
z = traindata['user_id']

x_train = np.array(x)
y_train = np.array(y)
userid_train = np.array(z)

#读取测试数据
testdata = pd.read_csv('tap_fun_test.csv')

x = testdata[[
              'pvp_battle_count',
              'pvp_lanch_count',
              'pvp_win_count',
              'pve_battle_count' ,
              'pve_lanch_count',
              'pve_win_count',
              'avg_online_minutes',
              'pay_price',
              'pay_count']]
#x = testdata[['wood_add_value', 'wood_reduce_value', 'stone_add_value', 'stone_reduce_value', 'ivory_add_value','ivory_reduce_value','meat_add_value','meat_reduce_value','magic_add_value','magic_reduce_value','infantry_add_value','infantry_reduce_value','cavalry_add_value','cavalry_reduce_value','shaman_add_value','shaman_reduce_value','wound_infantry_add_value','wound_infantry_reduce_value','wound_cavalry_add_value','wound_cavalry_reduce_value','wound_shaman_add_value','wound_shaman_reduce_value','general_acceleration_add_value','general_acceleration_reduce_value','building_acceleration_add_value','building_acceleration_reduce_value','reaserch_acceleration_add_value','reaserch_acceleration_reduce_value','training_acceleration_add_value','training_acceleration_reduce_value','pvp_battle_count', 'pvp_lanch_count', 'pvp_win_count', 'pve_battle_count' ,'pve_lanch_count', 'pve_win_count', 'avg_online_minutes', 'pay_price', 'pay_count']]

x_test = np.array(x)

#获取训练数据
trainpredictiondata = pd.read_csv('cleandata.csv')
x = trainpredictiondata[[
                         'pvp_battle_count',
                         'pvp_lanch_count',
                         'pvp_win_count',
                         'pve_battle_count' ,
                         'pve_lanch_count',
                         'pve_win_count',
                         'avg_online_minutes',
                         'pay_price',
                         'pay_count',
                         'continue_pay_price',
                         'continue_pay']]
x = np.array(x)

x = x[x[:, len(x[0])-1] == 1, ]
x=np.delete(x,len(x[0])-1,axis=1)

yy_train = x[:, len(x[0])-1]
xx_train = np.delete(x,len(x[0])-1,axis=1)


#XGboost
data_train = xgb.DMatrix(xx_train, label=yy_train)
data_test = xgb.DMatrix(x_test)
#watch_list = [(data_test, 'eval'), (data_train, 'train')]
param = {'max_depth': 6, 'eta': 0.2, 'silent': 0,'objective': 'reg:gamma'}
bst = xgb.train(param, data_train, num_boost_round=100)#, evals=watch_list)
yy_hat = bst.predict(data_test)


finaldata = pd.read_csv('tap_fun_test.csv')
x = finaldata[[
            'user_id',
            'pay_price',
            ]]
x=np.array(x)
x=np.insert(x,2,values=y_hat,axis=1)
x=np.insert(x,3,values=yy_hat,axis=1)

for i in range(len(x)):
    if(x[i][2]>0.18):
        x[i][1]=x[i][1]+x[i][3]

x=np.delete(x,3,axis=1)
x=np.delete(x,2,axis=1)
np.savetxt('finaldata01.csv', x, delimiter = ',')
print()

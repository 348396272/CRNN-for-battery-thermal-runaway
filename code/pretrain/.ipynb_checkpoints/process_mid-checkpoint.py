'''
Descripttion:
version:
Author: YinFeiyu
Date: 2022-11-02 15:24:48
LastEditors: YinFeiyu
LastEditTime: 2022-12-11 17:34:35
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy
import scipy.interpolate

# 只有 在你需要批量修改的时候，你才会想改这种东西。
# 模块化 啊 模块化 do you know what is 模块化

# start =20000 mid=31694 end=31700 steplength1=300 steplength2=20

def process(num,filepath,start,front_end,back_start,back_end):
    df = pd.read_csv(filepath,' ')
    df['n(mol)'] = df['n(mol)']-df.iloc[start]['n(mol)']
    df['time(s)'] = df['time(s)']-df.iloc[start]['time(s)']

    df_new=df.iloc[start:back_end].copy()
    df_new["n(mol)"][df_new["n(mol)"]<0.] = 0.
    a = list(df_new["n(mol)"])
    a.sort()
    df_new.loc[:,"n(mol)"] = np.array(a)
    a=list(df_new["T(K)"])
    a.sort()
    df_new.loc[:,"T(K)"]=np.array(a)

    df_back=df_new.iloc[front_end-start:]

    no_detail=np.arange(front_end,back_start+1)-front_end
    detail=np.linspace(back_start,back_end-1,20)-front_end
    all_inputx=np.concatenate([no_detail,detail[1:]])
    all_inputx=np.rint(all_inputx).astype(np.int32)

    df_back=df_back.iloc[all_inputx]
    last_one=df_back.iloc[-1].copy()
    for i in range(1,20):
        #time(s) T(K) P(bar) n(mol)
        last_one['time(s)']+=0.8
        df_back.loc[len(df_back)]=(last_one)
    df_back.to_csv(f"exp_data_2/back_exp_no{num}.txt", sep='\t', index=False, header=None)

    print("finish")
    plt.ticklabel_format(useOffset=False)
    plt.plot(df_back['time(s)'], df_back['n(mol)'],'o')
    plt.savefig(f"{num}_backnmol.png")
    plt.cla()
    plt.ticklabel_format(useOffset=False)
    plt.plot(df_back['time(s)'], df_back['T(K)'],'o')
    plt.savefig(f"{num}_backT.png")
    plt.cla()
    # plt.plot(df_back[0], df_back[2], 'o')
    # plt.savefig(f"{num}_back.png")
    # plt.cla()
    # plt.plot(df_new[0], df_new[2], 'o')
    # plt.savefig(f"{num}_end.png")
    # plt.cla()
    plt.close()

# t, T, P, mol
# 20000 396.594008 0.44957121349 0.041994560760221435
# 31690 437.65154099999995 0.78730402757 0.06664302585022772
# 31694 439.131583 0.79878141711 0.06738666692579136
# 31699.26 567.3798914 3.532578838 0.230652741
process(1,"exp_data_2/exp1.txt",20000,31690,31694,32221) ##/ 注 我这里 给end + 1了 因为显然这个是左闭右开

# 31000 399.057555 0.50838098374 0.04719483327187309
# 37660 416.62901899999997 0.79653445465 0.07082649437081072
# 37673 418.066367 0.81742480962 0.07243413542403458
# 37679 546.1065579 4.133429747 0.280397324
process(2,"exp_data_2/exp2.txt",31000,37660,37671,38274)


# 31000.0 359.673996 0.17257593 0.01777509
# 82400.0 443.584229 0.53068263 0.04431997
# 82411.0 445.25 0.55297813 0.0460092
# 82415.99 686.480005 3.96629943 0.21404166
process(3,"exp_data_2/test5.txt",31000,82400,82411,82911)

# 40000.0 363.752495 0.17997249 0.018329087
# 87710.0 470.298416 0.58671117 0.046215913
# 87723.0 484.13 0.58336077 0.04463915
# 87726.47 600.11079 2.9623382 0.1828706
process(4,"exp_data_2/test6.txt",40000,87710,87723,88071)

# 46000.0	371.582518	0.20122276	0.02006146
# 76666.0	420.70838	0.54385355	0.04788963
# 76672.41	692.818725	3.66145926	0.19578314

process(5,"exp_data_2/test7.txt",46000,76655,76666,77308)


# 40000.0	364.115981	0.17000696	0.01729688
# 89094.0	440.446051	0.50469365	0.04244982
# 89099.74	723.145749	4.76962736	0.24434267
process(6,"exp_data_2/test8.txt",40000,89085,89095,89669) #



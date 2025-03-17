'''
Descripttion:
version:
Author: YinFeiyu
Date: 2022-11-02 15:32:55
LastEditors: YinFeiyu
LastEditTime: 2022-11-16 14:42:59
'''
import numpy as np
import torch
import yaml
import argparse
import os
import shutil

# 我们 julia 真的是太快啦！ = =
class Settings:

    def parse(self):

        self.gas_m=28 ##/ 气体分子量 质量=分子量x摩尔数？
        self.init_m=5 ##/ 假设初始反应物体的质量为5g
        args=self.read_config("config.yaml")

        self.is_restart=args['is_restart']
        self.ns=args['ns']
        self.nr=args['nr']
        self.lb=args['lb']
        self.n_epoch=args['n_epoch']
        self.n_plot=args['n_plot']
        self.grad_max=args['grad_max']
        self.maxiters=args['maxiters']

        self.expr_name=args["expr_name"]
        self.fig_path=f'./results{self.ns}{self.nr}/{self.expr_name}/figs'
        self.ckpt_path=f'./results{self.ns}{self.nr}/{self.expr_name}/checkpoint'
        self.config_path=f'./results{self.ns}{self.nr}/{self.expr_name}/config.yaml'

        # 目测这一堆参数 后面一个用不上
        self.lr_max=args['lr_max']
        self.lr_min=args['lr_min']
        self.lr_decay=args['lr_decay']
        self.lr_decay_step=args['lr_decay_step']
        self.w_decay=args['w_decay']

        self.llb=self.lb # 我看不懂 我大受震撼
        self.p_cutoff=-1.0

        self.sample_size=4 # 一共四个数据 后面添加再改



        if not os.path.exists(f"./results{self.ns}{self.nr}"):
            os.mkdir(f"./results{self.ns}{self.nr}")

        if not os.path.exists(f"./results{self.ns}{self.nr}/{self.expr_name}"):
            os.mkdir(f"./results{self.ns}{self.nr}/{self.expr_name}")


        if not os.path.exists(self.fig_path):
            os.mkdir(self.fig_path)
            os.mkdir(os.path.join(self.fig_path, "conditions"))


        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)

        shutil.copyfile("config.yaml",self.config_path)

    def read_config(self,config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            result = yaml.load(f.read(), Loader=yaml.FullLoader)
        return result

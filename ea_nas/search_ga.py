import torch
import random
import copy
from thop import profile

from ea_nas.res_search import resnet_nas


# 约束条件计算（macs、parmas）
def constraint_cal(net, img_size=128):
    input = torch.randn(1, 3, img_size, img_size)
    macs, params = profile(net, inputs=(input,))
    return (macs, params)


# 是否满足约束条件
def is_satisfy_consts(consts, choose, all_choose, rate=(0.6, 1.05)):
    arch = resnet_nas(choose)
    if choose not in all_choose:
        arch_macs, arch_params = constraint_cal(arch)
        if consts[0] * rate[0] < arch_macs < consts[0] * rate[1] \
                and consts[1] * rate[0] < arch_params < consts[1] * rate[1]:
            return True
        else:
            return False
    else:
        return False


# 交叉
def cross(father, mother, cross_rate=0.5):
    child_fa = copy.deepcopy(father)
    child_mo = copy.deepcopy(mother)
    for i_stage in range(len(father)):
        for i_dna in range(min(father[i_stage][0], mother[i_stage][0])):
            if random.random() >= cross_rate:
                child_fa[i_stage][1][i_dna] = copy.deepcopy(mother[i_stage][1][i_dna])
                child_mo[i_stage][1][i_dna] = copy.deepcopy(father[i_stage][1][i_dna])
    return child_fa, child_mo


# 变异
def mutate(child, stages, channel_scales, kernels, mutate_rate=0.9):
    for i_stage in range(len(child)):
        del_num = 0
        for i_dna in range(child[i_stage][0]):
            if random.random() >= mutate_rate:
                mutate_type = random.choice(list(range(4)))
                if mutate_type <= 1:
                    tmp = [random.choice(channel_scales), random.choice(kernels)]
                    child[i_stage][1][i_dna - del_num] = tmp
                elif mutate_type == 2 and child[i_stage][0] < stages[i_stage][0] - 1:
                    child[i_stage][0] += 1
                    tmp = [random.choice(channel_scales), random.choice(kernels)]
                    child[i_stage][1].append(tmp)
                elif mutate_type == 3 and child[i_stage][0] > 2:
                    idx_del = random.choice(list(range(child[i_stage][0])))
                    child[i_stage][0] -= 1
                    del child[i_stage][1][idx_del]
                    del_num += 1
    return child


# 种子
def init_popu(consts_choose, stages, channel_scales, kernels, seed_len=40, rate=(0.6, 1.05)):
    arch = resnet_nas(consts_choose)
    consts = constraint_cal(arch)

    ga_seed = []
    ga_seed.append(consts_choose)
    while True:
        if len(ga_seed) >= seed_len:
            break

        new_idx = random.choice(list(range(len(ga_seed))))
        new_choose = copy.deepcopy(ga_seed[new_idx])
        new_choose = mutate(new_choose, stages, channel_scales, kernels, mutate_rate=0.8)
        if is_satisfy_consts(consts, new_choose, ga_seed, rate):
            ga_seed.append(new_choose)
    return ga_seed


# 交叉变异
def cross_mutation(consts_choose, ga_seed, stages, channel_scales, kernels, all_population,
                   rate=(0.6, 1.05), children_len=40, max_iter=1000):
    arch = resnet_nas(consts_choose)
    consts = constraint_cal(arch)

    children = []
    iter = 0
    while True:
        if len(children) >= children_len or iter >= max_iter:
            break
        idx = list(range(len(ga_seed)))
        random.shuffle(idx)
        idx_father, idx_mother = idx[:2]
        father = ga_seed[idx_father]
        mother = ga_seed[idx_mother]

        child_fa, child_mo = cross(father, mother)
        child_fa = mutate(child_fa, stages, channel_scales, kernels, mutate_rate=0.9)
        child_mo = mutate(child_mo, stages, channel_scales, kernels, mutate_rate=0.9)

        # 是否满足约束条件
        if is_satisfy_consts(consts, child_fa, all_population, rate):
            children.append(child_fa)
        if is_satisfy_consts(consts, child_mo, all_population, rate):
            children.append(child_mo)

        iter += 1
    return children


# 更新种群
def update_popu(population, population_acc, k=40):
    population_acc = torch.tensor(population_acc)
    v, idx = torch.topk(population_acc, k, dim=-1)
    population_new = []
    population_new_acc = []
    for i in idx:
        population_new.append(population[i])
        population_new_acc.append(population_acc[i])
    return population_new, population_new_acc

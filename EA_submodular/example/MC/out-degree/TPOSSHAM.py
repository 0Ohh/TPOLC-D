import numpy as np

import coverage


n = 595
k = 10  # IOH set
delta = 5
epsilon = 3  # delta * (2e + 1) ~= 2k;  d, e = (4,2) or (7,1)
n_sl = 2*epsilon + 1

def Tposs():

    for i in range(7, 15):
        print(i, slot_index(i))

    popu_slots = np.zeros([n_sl, delta, n], 'int')
    popu_index_tuples = []
    f_slots = np.zeros([n_sl, delta], 'float')

    all_sum = np.zeros(n, 'int')
    t = 0
    T = 600_0000
    # todo initial
    init = np.zeros(n, 'int')
    poses = np.random.choice(range(n), size=k, replace=False)
    init[poses] = 1
    init_i = slot_index(init.sum())
    init_f = F(init)
    popu_slots[init_i, 0] = init
    f_slots[init_i, 0] = init_f
    popu_index_tuples.append((init_i, 0))
    while t < T:
        t += 1
        if t % 10000 == 0:
            # print(t, f_slots)
            best_f = -1
            best_at = None
            for i, j in popu_index_tuples:
                if popu_slots[i, j].sum() > k:
                    continue
                if f_slots[i, j] > best_f:
                    best_f = f_slots[i, j]
                    best_at = (i, j)
            print(t, best_f)
        if t % 10_0000 == 0:
            print(f_slots)
        if t % 20_0000 == 0:
            print(popu_slots)
        rand_ind = np.random.randint(0, len(popu_index_tuples))
        x_tuple = popu_index_tuples[rand_ind]
        x_old = popu_slots[x_tuple]
        x_cp = np.copy(x_old)
        x_new = mutate_to_card_tar(x_cp, k, k-epsilon, k+epsilon, 1/n)
        fx = F(x_new)
        card = x_new.sum()
        x_sl_index = slot_index(card)

        # print(x_sl_index)

        # todo selection
        if card > k+epsilon or card < k-epsilon:
            continue
        slot = popu_slots[x_sl_index]
        f_sl = f_slots[x_sl_index]
        add_info = (x_new, fx, slot, f_sl)
        f_min = np.inf
        f_min_at = []
        if (f_sl == 0).all():  # todo empty
            add_x(add_info, 0)
            popu_index_tuples.append((x_sl_index, 0))
            all_sum += x_new
            continue
        has = False
        for pi in range(delta):    # todo already has x_new
            if (slot[pi] == x_new).all():
                has = True
                break
        if has: continue
        added = False
        for pi in range(delta):
            if f_sl[pi] == 0:
                add_x(add_info, pi)
                popu_index_tuples.append((x_sl_index, pi))
                all_sum += x_new
                added = True
            if f_sl[pi] < f_min:
                f_min = f_sl[pi]
                f_min_at = [pi]
            elif f_sl[pi] == f_min:
                f_min_at.append(pi)
        if added: continue
        if fx < f_min:  #todo   1111111111111111111111111111     1233333333333333333333333333333333333333333333333333333333
            continue
        else:
            if len(f_min_at) == 1 and fx > f_min:
                add_x(add_info, f_min_at[0])
                all_sum = all_sum + x_new - popu_slots[x_sl_index, f_min_at[0]]
                print('fuck')
                continue

            temp_global = all_sum + x_new
            ham_min = np.inf
            worst_at = None
            temp_popN = len(popu_index_tuples) + 1
            for pi in f_min_at:
                ham_i = (np.multiply(temp_popN - temp_global, popu_slots[pi]) +
                         np.multiply(temp_global, 1 - popu_slots[pi])
                        ).sum()
                if ham_i < ham_min:
                    ham_min = ham_i
                    worst_at = pi
            ham_x = (np.multiply(temp_popN - temp_global, x_new) +
                         np.multiply(temp_global, 1 - x_new)
                        ).sum()
            if ham_x > ham_min:
                add_x(add_info, worst_at)
                all_sum = all_sum + x_new - popu_slots[x_sl_index, worst_at]


def add_x(add_info, pi):
    x, fx, sl, f_sl = add_info
    sl[pi] = x
    f_sl[pi] = fx


def F(x):
    return coverage.F(x)


def mutate_to_card_tar(x, k, l_b, r_b, pm=-1):
    x_ori = np.copy(x)
    cardx = x.sum()
    k_x = k - cardx
    n = len(x)
    if pm == -1:
        pm = 1/n
    p0 = (np.abs(k_x) + k_x + pm * n) / (2*(n - cardx))
    p1 = (np.abs(k_x) - k_x + pm * n) / (2*cardx)
    while 1:
        x = np.copy(x_ori)
        change1_to_0 = np.random.binomial(1, p1, n)
        change1_to_0 = np.multiply(x, change1_to_0)
        change1_to_0 = 1 - change1_to_0
        x = np.multiply(x, change1_to_0)

        change0_to_1 = np.random.binomial(1, p0, n)
        change0_to_1 = np.multiply(1-x_ori, change0_to_1)

        x += change0_to_1

        # if x.sum() > k:
        #     print('----')

        if l_b <= x.sum() <= r_b and (x != x_ori).any():
        # if (x != x_ori).any():
            # if 1:
            return x
def slot_index(card):
    i = card - (k - epsilon)
    # if 0 <= i < n_sl:
    return i


if __name__ == "__main__":
    Tposs()


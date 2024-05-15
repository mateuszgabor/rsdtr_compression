import torch
import math


def reverse_vector(vector):
    l = len(vector)
    rev_vector = torch.empty(l)
    for i in range(len(vector)):
        rev_vector[i] = vector[l - 1 - i]

    return rev_vector


def truncation_index(vector, bound):
    rnew = len(vector)
    rev_vector = reverse_vector(vector)
    norm_vector = torch.cumsum(rev_vector[:] ** 2, dim=0)
    l = len(norm_vector)
    for i in range(l - 1, -1, -1):
        if norm_vector[i] < bound**2:
            rnew = l - i - 1
            break

    return rnew


def get_divisors(n):
    for i in range(1, int(n / 2) + 1):
        if n % i == 0:
            yield i
    yield n


def fullTR(tr):
    a = tr[0]
    d = len(tr)
    r0, n, _ = a.shape
    ns = []
    ns.append(n)

    for k in range(1, d):
        rold, n, rnew = a.shape
        b = tr[k]
        rnew, n, rnext = b.shape
        ns.append(n)
        b = torch.reshape(b, (rnew, int(torch.numel(b) / rnew)))
        tmp = a @ b
        a = torch.reshape(tmp, (rold, int(torch.numel(tmp) / (rold * rnext)), rnext))

    a = torch.reshape(a, (r0, int(torch.numel(tmp) / (r0**2))))
    res = torch.zeros(a.shape)

    for k in range(r0):
        res = res + a[k, :]

    res = torch.reshape(res, (ns))
    return res


def storage_size(tr):
    total_size = 0
    for core in tr:
        total_size += math.prod(core.shape)

    return total_size


def shift_list(lst, curr_index):
    index = len(lst) - curr_index
    return lst[index:] + lst[:index]


def tr(A, prec, r0):
    s = A.shape
    d = len(s)
    norm_error = prec / math.sqrt(d - 1)
    cores = []
    C = A
    rold = 1
    n = s[0]
    C = torch.reshape(C, (rold * n, int(torch.numel(C) / (rold * n))))
    [U, S, Vt] = torch.linalg.svd(C, full_matrices=False)
    rnew = truncation_index(S, norm_error * torch.norm(S))
    U = U[:, 0:rnew]
    S = S[0:rnew]
    Vt = Vt[0:rnew, :]
    Unew = torch.zeros(rold * r0, n, int(rnew / r0))
    for index in range(r0):
        start = (index) * int(rnew / r0)
        end = (index + 1) * int(rnew / r0)
        Unew[index, :, :] = U[:, start:end]
    cores.append(Unew)
    C = torch.diag(S) @ Vt
    Cnew = torch.zeros(int(rnew / r0), int(torch.numel(C) / rnew), r0)
    for index in range(r0):
        start = (index) * int(rnew / r0)
        end = (index + 1) * int(rnew / r0)
        Cnew[:, :, index] = C[start:end, :]
    C = Cnew
    C = torch.reshape(C, (int(rnew / r0), *s[1:], r0))
    C = torch.reshape(C, (int(rnew / r0), *s[1 : len(s) - 1], s[-1] * r0))
    C = torch.reshape(C, (int(rnew / r0), int(torch.numel(C) / (rnew / r0))))
    rold = int(rnew / r0)
    for k in range(1, d - 1):
        n = s[k]
        C = torch.reshape(C, (rold * n, int(torch.numel(C) / (rold * n))))
        [U, S, Vt] = torch.linalg.svd(C, full_matrices=False)
        rnew = truncation_index(S, norm_error * torch.norm(S))
        U = U[:, 0:rnew]
        S = S[0:rnew]
        Vt = Vt[0:rnew, :]
        cores.append(torch.reshape(U, (rold, n, rnew)))
        C = torch.diag(S) @ Vt
        rold = rnew

    C = torch.reshape(C, (rold, s[-1], r0))
    cores.append(C)

    return cores


def rsdtr(A, prec):
    s = A.shape
    d = len(s)
    norm_error = prec / math.sqrt(d - 1)
    curr_small = math.inf
    curr_ind = -1
    B = A
    tr_res = None

    for start_ind in range(d):
        if start_ind != 0:
            B = torch.moveaxis(B, 0, -1)

        s = B.shape
        unf = torch.reshape(B, (s[0], math.prod(s[1:])))
        [_, S, _] = torch.linalg.svd(unf, full_matrices=False)
        rnew = truncation_index(S, norm_error * torch.norm(S))
        factors = list(get_divisors(rnew))
        l = len(factors)
        for i in range(l):
            r0 = factors[i]
            trtmp = tr(B, prec, r0)
            a = storage_size(trtmp)
            if a < curr_small:
                tr_res = trtmp
                curr_ind = start_ind
                curr_small = a

    res = shift_list(tr_res, curr_ind)
    return res

import torch
import torch.nn as nn



class h2():
    def __init__(self):
        super().__init__()

    def constrain_0(self,x):
        return torch.sign(x)*torch.softmax(x,dim=1)                                      #both long short and |wt|=1


    def constrain_1(self,x):
        return torch.softmax(x, dim=1)                                                   #long only+|wt|=1

    def constrain_2(self,x,y_dim,u):
        a_value=  (1.0 - u) / (y_dim * u - 1.0)                                          #maximum+|wt|=1
        phi_s = a_value + torch.sigmoid(torch.abs(x))
        sum_phi_s = torch.sum(phi_s, dim=-1, keepdim=True)
        normalized_magnitudes = phi_s / (sum_phi_s + 1e-12)
        w = torch.sign(x) * normalized_magnitudes
        return w


    def constrain_3(self,x,y_dim,K):
        device = x.device
        dtype = x.dtype
        s_j = x.unsqueeze(2)
        s_m = x.unsqueeze(1)
        abs_diff = torch.abs(s_j - s_m)
        term2 = torch.sum(abs_diff, dim=2)
        i_vec = torch.arange(1, y_dim + 1, device=device, dtype=dtype)
        coeff_i = (y_dim + 1) - (2 * i_vec)
        s_broadcasted = x.unsqueeze(1)
        coeff_i_broadcasted = coeff_i.view(1, y_dim, 1)
        term1 = coeff_i_broadcasted * s_broadcasted
        term2_broadcasted = term2.unsqueeze(1)

        #get ranking matrix
        A= term1 - term2_broadcasted
        A=torch.softmax(A, dim=2)
        s_expanded = x.unsqueeze(-1)
        weighted_scores_expanded = torch.matmul(A, s_expanded)
        sorted_s = weighted_scores_expanded.squeeze(-1)     #get the sorted score



        n = (K // 2) + 1
        N = y_dim

        n_idx_upper = n-1
        n_idx_lower = (N - n)

        d_u = sorted_s[:, n_idx_upper]
        d_l = sorted_s[:, n_idx_lower]



        d_u = d_u.unsqueeze(-1)
        d_l = d_l.unsqueeze(-1)

        exp_abs_s = torch.exp(torch.abs(x))
        mask_u = (x > d_u).float()  # .float() 将 bool 转为 1.0/0.0


        num_u = mask_u * exp_abs_s
        den_u = torch.sum(num_u, dim=-1, keepdim=True) + 1e-12


        term_u = num_u / den_u
        mask_l = (x < d_l).float()

        num_l = mask_l * exp_abs_s

        den_l = torch.sum(num_l, dim=-1, keepdim=True) + 1e-12
        term_l = num_l / den_l

        w = 0.5 * (term_u - term_l)                 #long the top n instruments and short the bottom n instruments


        return w


    def constrain_4(self,x,L):
        return L*torch.sign(x)*torch.softmax(x,dim=1)

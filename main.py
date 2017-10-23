from ClusteringCascadeLinUCB import *
from CascadeLinUCB import *
from MultipleLinUCB import *
import numpy as np
import data_factory_yelp as yelp_df
import data_factory_movielens as movie_df
from useful_fns import *

para_dict={}
# size of data
d = 20
para_dict['d'] = d
K=4
para_dict['K'] = K
item = 1000
para_dict['item'] = item
hist_user=100
para_dict['hist_user'] = hist_user

criterion_user=900
para_dict['criterion_user'] = criterion_user
group_number = 1  # only for synthetic experiments, works with selected number of users
para_dict['group_number'] = group_number
group_size = criterion_user / group_number
# specific paras for clustering algorithm
edge_mode = 'compelte'
para_dict['edge_mode'] = edge_mode
cut_mode = 'p2p'
para_dict['cut_mode'] = cut_mode


# almost fixed paras
#seeds=np.random.randint(0, 100000000)
seeds = 372165
para_dict['seeds'] = seeds

# other important paras
# to ensure that the picture looks beautiful, there should be less than 10 ticks in x dimension, so take x_step carefully
time = 5000000
para_dict['time'] = time
synthetic = False
para_dict['synthetic'] = synthetic


# experiment starts here
dict_list=[]
np.random.seed(seeds)

sub_X, sub_bool = movie_df.get_data(item, hist_user, criterion_user, d)
# if synthetic:
#     sub_X = st.feature_vector(item, d)

para_dict['dataset'] = 'movie'

group_theta = st.orthogonal_theta(d, group_number)
opt_list, w_list = group_stimulated_optimal(sub_X, group_theta, K)

d1={}
list1 = ClusteringCascadeLinUCB(time, K, sub_X, sub_bool, synthetic, edge_mode, cut_mode, group_size, group_theta, opt_list, w_list)
d1['data']=list1
d1['color']='red'
d1['linewidth']=1
d1['linestyle']='-'
d1['marker']='o'
d1['label']='CLUB-cascade'
dict_list.append(d1)

d2={}
list2 = CascadeLinUCB(time, K, sub_X, sub_bool, synthetic, group_size, group_theta, opt_list, w_list)
d2['data']=list2
d2['color']='green'
d2['linewidth']=1
d2['linestyle']='--'
d2['marker']='^'
d2['label']= "C$^3$-UCB"+'/CascadeLinUCB'
dict_list.append(d2)

d3={}
list3 = MultipleLinUCB(time, K, sub_X, sub_bool, synthetic, edge_mode, cut_mode, group_size, group_theta, opt_list, w_list)
d3['data']=list3
d3['color']='blue'
d3['linewidth']=1
d3['linestyle']=':'
d3['marker']='x'
d3['label']='MultipleLinUCB'
dict_list.append(d3)

save_data(para_dict, dict_list, "whole")
# remember that the unit of x_step and y_step is K, not 1
plot_curves(dict_list)

print seeds

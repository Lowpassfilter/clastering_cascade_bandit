import json
from manifest import *
import numpy as np
import scipy.sparse as spp
import scipy.io as sio

def read_business_as_dict(filename):
    f=open(filename)
    d={}
    for line in f:
        pair=line.strip('\n').strip('\r')
        d[pair]=0
    return d

def load_full_review_matrix(reviewfilename, business_dict):
    data_list=[]
    row_indics_list=[]
    col_indics_list=[]

    business_rowindex_dict={}
    user_colindex_dict={}
    review_index_dict={}

    new_review_index=0
    new_row_index=0
    new_col_index=0

    rf=open(reviewfilename)
    for line in rf:
        json_object=json.loads(line)

        if business_dict.has_key(json_object['business_id']):
            review_index_dict[json_object['review_id']]=new_review_index
            
            row_index=0
            col_index=0

            if business_rowindex_dict.has_key(json_object['business_id']):

                row_index=business_rowindex_dict[json_object['business_id']]
            else:
                row_index=new_row_index
                business_rowindex_dict[json_object['business_id']]=new_row_index
                new_row_index+=1
            if user_colindex_dict.has_key(json_object['user_id']):

                col_index=user_colindex_dict[json_object['user_id']]
            else:
                col_index=new_col_index
                user_colindex_dict[json_object['user_id']]=new_col_index
                new_col_index+=1


            data_list.append(json_object['stars'])
           
            row_indics_list.append(row_index)
            col_indics_list.append(col_index)
        new_review_index += 1
    
    data=np.array(data_list)
    rows=np.array(row_indics_list)
    cols=np.array(col_indics_list)
    s_m=spp.csr_matrix((data, (rows, cols)))

    return s_m, business_rowindex_dict, user_colindex_dict, review_index_dict

def save_sparse_matrix(s_m, s_file_name, business_rowindex_dict, business_rowindex_dict_file_name, user_colindex_dict, user_colindex_dict_file_name):
    sio.mmwrite(s_file_name, s_m)
    np.save(business_rowindex_dict_file_name, business_rowindex_dict)
    np.save(user_colindex_dict_file_name, user_colindex_dict)

def save_all_restaurant(reviewfilename):
    business_dict=read_business_as_dict(RESTAURANT_LIST_FILE_NAME)
    s_m, business_rowindex_dict, user_colindex_dict, review_index_dict=load_full_review_matrix(reviewfilename, business_dict)
    save_sparse_matrix(s_m, FULL_MATRIX_FILE_NAME, business_rowindex_dict, FULL_BUSINESS_ROWINDEX_DICT_FILE_NAME, user_colindex_dict, FULL_USER_COLINDEX_DICT_FILE_NAME)
    

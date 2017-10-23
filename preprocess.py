from manifest import *
import create_restaurant_file as rd
import create_small_review_file as sr
import create_full_data_file as fd

#restaurant id

# rd.creat_restaurants_id(BUSINESS_FILE_NAME,RESTAURANT_LIST_FILE_NAME)

#small reveiw
# sr.small_review_file(REVIEW_COUNT, REVIEW_FILE_NAME, SMALL_REVIEW_FILE_NAME)

#full
fd.save_all_restaurant(REVIEW_FILE_NAME)
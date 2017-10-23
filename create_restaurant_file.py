import json

def find_all_resturants(filename):
    business=open(filename)
    resturant_id=[]
    for line in business:
        busi_json=json.loads(line)
        if "Restaurants" in busi_json['categories']:
            resturant_id.append(busi_json['business_id'])
    return resturant_id

def write_list(l, filename):
    f=open(filename, "wb")
    for a in l:
        f.write(a+'\r\n')



def creat_restaurants_id(readfilename, writefilename):
    restaurants=find_all_resturants(readfilename)
    write_list(restaurants, writefilename)





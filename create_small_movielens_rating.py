
rf = open('ml-20m/ratings.csv')
wf = open('ml-20m/small_ratings.csv', "wb")
count = 0

for line in rf:
    wf.write(line)
    count +=1

    if count> 1000000:
        break

rf.close()
wf.close()

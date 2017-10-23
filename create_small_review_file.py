

def small_review_file(count, readfile, writefile):
    rf=open(readfile)
    wf=open(writefile,"wb")
    i=0
    for line in rf:
        wf.write(line)
        i+=1
        if i>count:
            break


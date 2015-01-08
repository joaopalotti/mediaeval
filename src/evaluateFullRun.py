from evaluation import evaluate, deleteFile

def evaluteFullFile(filename):

    tempname = "tmp_" + filename
    temp = open(tempname, "w")
    
    for row in open(filename, "r").readlines():
        if int(row.split()[3]) > 49:
            continue
        temp.write(row)
    temp.flush()
    evaluate(tempname, "dev2014", "__eval.csv", []) 
    deleteFile(tempname)


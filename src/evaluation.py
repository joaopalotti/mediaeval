import subprocess
import csv

def deleteFile(filename):
    p = subprocess.Popen("rm " + filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def evaluate(filename, collection, evalcsv, tocsv=[]):

    if collection == "dev2013":
        cmd = "java -jar div_eval.jar -r " + filename + " -rgt ../archive2013/devset/keywords/gt/rGT/ -dgt ../archive2013/devset/keywords/gt/dGT/ -t ../archive2013/devset/keywords/devsetkeywords_topics.xml -o ."
    elif collection == "test2013":
        cmd = "java -jar div_eval.jar -r " + filename + " -rgt ../archive2013/testset/keywords/gt/rGT/ -dgt ../archive2013/testset/keywords/gt/dGT/ -t ../archive2013/testset/keywords/testsetkeywords_topics.xml -o ."
    elif collection == "dev2014":
        cmd = "java -jar div_eval.jar -r " + filename + " -rgt ../devset/gt/rGT/ -dgt ../devset/gt/dGT/ -t ../devset/devset_topics.xml -o ."

    #p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #,,.7515,.6924,.6742,.6606,.6282,.5967,.278,.4334,.6329,.7475,.8257,.8808,.3826,.5062,.6192,.6668,.6751,.6679
    #print p.communicate()[0]
    
    import time
    time.sleep(3)
    
    fin = open(filename + "_metrics.csv", "r")
    resultLine = fin.readlines()[-1]
    rFloats = map(float,resultLine.split(",,")[1].split(","))
    with open(evalcsv, 'ab') as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        p10, p20, cr10, cr20, f10, f20 = rFloats[1], rFloats[2], rFloats[7], rFloats[8], rFloats[13], rFloats[14]
        print "%.3f %.3f %.3f %.3f %.3f %.3f" % (p10, cr10, f10, p20, cr20, f20)
        writer.writerow(tocsv + [p10,cr10,f10,p20,cr20,f20])



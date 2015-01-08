#!/bin/python
from __future__ import division
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import networkx as nx
import glob
import scipy
import metis
import sys
import re
from itertools import permutations
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics 

from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering

from scipy.spatial.distance import pdist, cdist

from filterDataset import filterDT
from collections import Counter
from getParamters import getParamters
from evaluation import evaluate, deleteFile

###############################################################################
#####----------------------INPUT PART BEGINS-----------------------------######
###############################################################################
args = getParamters()

#The name says by itself
OUTFILE = args.outfile
deleteFile(OUTFILE)

#Options: "flickr", "oracle", "metis", "kmeans"
# "simkmeans", "min-max", "multigraph",
# "spectral", "agglomerative", "affinity"
# "textsim"
METHOD = args.method

#Auxiliar method is an option only if the clustering method is multigraph
AUXILIAR_METHOD = args.method_aux

#Should be "dev2013", "test2013", "dev2014" or "test2014"
WORKING_DATASET = args.working_dataset

#CSV where the main values will be saved
EVAL_CSV= args.evalcsv

#If another initial ranking is provided, this is the paramter for that
APPLY_NEW_RANK_FILE=args.apply_new_ranking

#Default set to allow self loopings
ALLOW_SELF_LOOP=not args.disableSelfLooping

#If you want to use text similarity
USE_TEXT_SIM = args.textsim

#Options: "HOG", "CM3x3", "CM", "CN", "CN3x3", "GLRLM", "GLRLM3x3", "LBP", "LBP3x3"
# CM and CM3x3 have negative values. How to deal with it? I am not treating it now.
VISUAL_DESCRIPTORS = args.visual

#["license", "latitude", "longitude", "nbComments", "tags", "title", "username" (see uid, buid), "dist", "views", "partOfTheDay", "uid", "buid"]
OTHER_DESCRIPTORS = args.others

#Options: cosine (0..1), chebyshev (0..1), braycurtis (0..1)
# euclidian (int), 'canberra' (int), 'minkowski' (int)
SIM_METRIC = args.metric

#Whether we should use pre-filter or not and which datasets should be used for it.
USE_FILTER = args.use_filter
SUPERVISED = args.supervised_filter
#Options: ["dev2013", "test2013", "dev2014"]
SUPERVISED_DATASETS = args.supervised_dataset 

USE_CREDIBILITY = args.use_credibility

#Options for Multigraph method only
MULTIGRAPH_MEAN_THRESHOLD = args.pmean
MULTIGRAPH_MIN_THRESHOLD = args.pmin
MULTIGRAPH_MAX_THRESHOLD = args.pmax
MULTIGRAPH_MEAN_INCREMENT = args.pimean
MULTIGRAPH_MIN_INCREMENT = args.pimin
MULTIGRAPH_MAX_INCREMENT = args.pimax

#Options for Min_max method only
MIN_MAX_RANK_THRESHOLD = args.minmax_rank

#Number of elements to be saved to the .csv file
MAX_ELEMENTS = 50

#Some variables not set as parameters:
SIM_KEEP_ABOVE=.0 #.95
MAX_USERS_PER_POI=20
DEBUG=False
ONLY_RELEVANT = False
GRAPH_RULE = "default"


#Please, change the following lines according to your configuration:
TEST_2014 = {'NUM_CLUSTERS':20, 'TOPICS_PATH':"../testset/testset_topics.xml", 'XML_PATH':"../testset/xml/",
            'IMAGE_DESC':"../testset/descvis/img/", 'CRED_PATH':"../testset/desccred/", 'IMAGE_PATH':"../testset/img/",
             'RGT_PATH':None, 'DGT_PATH':None, 'SOURCE':"test2014", 'TEXT_SIM':"../testset/simmatrix_2014test.csv"}

DEV_2014 = {'NUM_CLUSTERS':20, 'TOPICS_PATH':"../devset/devset_topics.xml", 'XML_PATH':"../devset/xml/",
            'RGT_PATH':"../devset/gt/rGT/",'DGT_PATH':"../devset/gt/dGT/", 'IMAGE_DESC':"../devset/descvis/img/",
            'CRED_PATH':"../devset/desccred/", 'IMAGE_PATH':"../devset/img/", "SOURCE":"dev2014",
            'TEXT_SIM':"../devset/simmatrix_2014dev.csv"}

DEV_2013 = { 'NUM_CLUSTERS':10, 'TOPICS_PATH':"../archive2013/devset/keywords/devsetkeywords_topics.xml",
            'XML_PATH':"../archive2013/devset/keywords/xml/", 'RGT_PATH':"../archive2013/devset/keywords/gt/rGT/",
            'DGT_PATH':"../archive2013/devset/keywords/gt/dGT/", 'IMAGE_DESC':"../archive2013/devset/keywords/descvis/img/",
            "IMAGE_PATH":"../archive2013/devset/keywords/img/", "SOURCE":"dev2013"}

TEST_2013 = { 'NUM_CLUSTERS':10, 'TOPICS_PATH':"../archive2013/testset/keywords/testsetkeywords_topics.xml",
             'IMAGE_DESC':"../archive2013/testset/keywords/descvis/img/", "SOURCE":"test2013",
               'XML_PATH':"../archive2013/testset/keywords/xml/", 'RGT_PATH':"../archive2013/testset/keywords/gt/rGT/",
             'DGT_PATH':"../archive2013/testset/keywords/gt/dGT/", "IMAGE_PATH":"../archive2013/testset/keywords/img/"}

###############################################################################
#####-----------------------INPUT PART ENDS------------------------------######
###############################################################################

###############################################################################
#####-----------------LOAD DATA METHODS BEGINS---------------------------######
###############################################################################

def get_map(name):
    if name == 'dev2013':
        return DEV_2013
    elif name == 'test2013':
        return TEST_2013
    elif name == 'dev2014':
        return DEV_2014
    else:
        print "ERROR - MAP NOT DEFINED"
        sys.exit(0)
################################################################################

def get_topic_map(filename="../devset/devset_topics.xml"):
    #devset_topics.xml
    topic_map = dict()
    tree = ET.parse(filename)
    root = tree.getroot()
    for topic in root:
        num, name = int(topic.find("number").text), topic.find("title").text
        topic_map[name] = num
    return topic_map
################################################################################

def xmlFile_to_dataframe(filename, use_credibility, working_dataset):
    """ 
    gets any xml file from the folder "xml" as input and generates a pandas DataFrame
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    poi_name = filename.split("/")[-1].split(".xml")[0]
    photos = []

    for photo in root:
        photos.append(photo.attrib)

    xml_atts = ['username', 'nbComments', 'description', 'license', 'tags', 'views', 'longitude', 'id', 'rank',\
                        'latitude', 'title', 'url_b', 'date_taken']
    
    if use_credibility and "2014" in working_dataset:
        xml_atts += ["userid"]

    photo_dict = {}
    for att in xml_atts:
        photo_dict[att] = [photo[att] for photo in photos]

    df = pd.DataFrame(data=photo_dict)
    df["id"] = df["id"].astype(np.int64)
    df["nbComments"] = df["nbComments"].astype(np.int32)
    df["rank"] = df["rank"].astype(np.int32)
    df["poiName"] = poi_name
    df["latitude"] = df["latitude"].astype(np.float64)
    df["longitude"] = df["longitude"].astype(np.float64)
    df["views"] = df["views"].astype(np.int32)
    df["license"] = df["license"].astype(np.int32)
    return df
################################################################################

def credibility_XMLFile_to_dict(filename):
    content = open(filename, "r").readlines()
    important_content = content[0:12] + [content[-1]]
    tree = ET.fromstring("".join(important_content))
    credscore = {}
    credscore["userid"] = [filename.split("/")[-1].split(".xml")[0]]
    for f in tree[0]:
        credscore[f.tag] = [float(f.text)]
    return credscore
################################################################################

def get_relevance(rgt_path):
    """
    Reads the relevance ground truth
    """
    gt_type_file = "rGT.txt"
    gt_files = glob.glob(rgt_path + "*" + gt_type_file)
    gtcsv = pd.read_csv(gt_files[0], names=["id","rel","poiName"])
    gtcsv["poiName"].fillna(gt_files[0].split(rgt_path)[1].split(gt_type_file)[0].strip(), inplace=True)
    for gt_file in gt_files[1:]:
        gtcsv = pd.concat([gtcsv, pd.read_csv(gt_file, names=["id","rel","poiName"])] )
        gtcsv["poiName"].fillna(gt_file.split(rgt_path)[1].split(gt_type_file)[0].strip(), inplace=True)
    return gtcsv
################################################################################

def get_group_relevance(dgt_path):
    """
    Reads the group relevance
    """
    dgt_type_file = "dGT.txt"
    gtd_files = glob.glob(dgt_path + "*" + dgt_type_file)
    gtdcsv = pd.read_csv(gtd_files[0], names=["id","grp","poiName"])
    gtdcsv["poiName"].fillna(gtd_files[0].split(dgt_path)[1].split(dgt_type_file)[0].strip(), inplace=True)

    for gtd_file in gtd_files[1:]:
        gtdcsv = pd.concat([gtdcsv, pd.read_csv(gtd_file, names=["id","grp","poiName"])] )
        gtdcsv["poiName"].fillna(gtd_file.split(dgt_path)[1].split(dgt_type_file)[0].strip(), inplace=True)
    gtdcsv["grp"] = gtdcsv["grp"].astype(np.int32)
    return gtdcsv
################################################################################

"""
Reads one visual feature (HOG, CN, CM...)
"""
def read_visual_feature(datapath, feature, prefix):
    #feature_files = glob.glob("../devset/descvis/img/hearst_castle " + metric + ".csv")
    feature_files = glob.glob(datapath + "*" + feature + ".csv")
    feature_csv = pd.read_csv(feature_files[0], prefix=prefix, header=None)
    feature_csv["poiName"] = feature_files[0].split(datapath)[1].split(feature)[0].strip()
    
    for feature_file in feature_files[1:]:
        feature_csv = pd.concat([feature_csv, pd.read_csv(feature_file, prefix=prefix, header=None)] )
        feature_csv["poiName"].fillna(feature_file.split(datapath)[1].split(feature)[0].strip(), inplace=True)

    feature_csv["id"] = feature_csv[prefix + "0"]
    del feature_csv[prefix + "0"]
    return feature_csv
################################################################################

"""
Generates the similarities based on pre-defined metrics (HOG, CM3x3...)
"""
def create_feature_header(list_of_features):
    features = []
    for (prefix, size) in list_of_features:
        for n in xrange(1,size):
            features.append(prefix + str(n))
    return features
################################################################################
def get_gps_from_topics(filename="../devset/devset_topics.xml"):
    """
    <topic>
        <number>5</number>
        <title>pont_alexandre_iii</title>
        <latitude>48.863611</latitude>
        <longitude>2.313611</longitude>
        <wiki>http://en.wikipedia.org/wiki/Pont_Alexandre_III</wiki>
    </topic>
    """
    gps = {}
    tree = ET.parse(filename)
    root = tree.getroot()
    for child in root:
                                        # (lat, long) pairs
        gps[child.find("title").text] = (float(child.find("latitude").text),\
                                         float(child.find("longitude").text)) 
    return gps
################################################################################

def distance_from_poi(row, gps):
    poilat, poilon = gps[row["poiName"]]
    lat, lon = float(row["latitude"]), float(row["longitude"])
    
    if lat == 0 or lon == 0:
        return -1.0
    return np.sqrt( (lat-poilat) * (lat-poilat) + (lon-poilon) * (lon-poilon) )
################################################################################

def create_dataset(dtmap, visual_descriptors, other_descriptors, working_dataset, use_credibility=False):

    topics_path = dtmap["TOPICS_PATH"]
    xml_path = dtmap["XML_PATH"]
    rgt_path = dtmap["RGT_PATH"]
    dgt_path = dtmap["DGT_PATH"]
    image_description_path = dtmap["IMAGE_DESC"]

    topic_map = get_topic_map(filename=topics_path)
    xmlfiles = glob.glob(xml_path + "*.xml")
    dataset = xmlFile_to_dataframe(xmlfiles[0], use_credibility, working_dataset)

    for xmlfile in xmlfiles[1:]:
        dataset = pd.concat([dataset, xmlFile_to_dataframe(xmlfile, use_credibility, working_dataset)] )

    if rgt_path and dgt_path:
        gtcsv = get_relevance(rgt_path)
        gtdcsv = get_group_relevance(dgt_path)
        
        # Merge everything!
        dataset = pd.merge(gtcsv, dataset, on=["id","poiName"])
        #fills the NA added because not every id was listed in the dGT filename
        dataset = pd.merge(gtdcsv, dataset, on=["id","poiName"], how='right').fillna(0) 

    #Add visual descriptions
    desccsv = {}
    for descriptor in visual_descriptors:
        desccsv[descriptor] = read_visual_feature(image_description_path, descriptor, descriptor)
        dataset = pd.merge(dataset, desccsv[descriptor], on=["id","poiName"], how='left').fillna(0)

    #Create header for visual descriptors
    list_of_descriptors = []
    for descriptor in visual_descriptors:
        list_of_descriptors.append( [descriptor, desccsv[descriptor].shape[1] - 1] )
    header = create_feature_header(list_of_descriptors)
    
    for descriptor in other_descriptors:
        header.append(descriptor)

    # Add distance information based on GPS data
    gps = get_gps_from_topics(topics_path)
    dataset["dist"] = dataset.apply(distance_from_poi, args=(gps,), axis=1)

    dataset["descLength"] = dataset["description"].apply((lambda x: len(x)))
    dataset["descWLength"] = dataset["description"].apply((lambda x: len(x.split())))
    def timeOfDay(d):
        #d[0:4] correstponds to the year while d[5:7] is the number of months
        if int(d[0:4]) < 1960 or int(d[5:7]) == 0:
            return -1
        dd = datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
        if dd.hour <= 12:
            return 0
        if dd.hour <= 18:
            return 1
        return 2
    dataset["partOfTheDay"] = dataset["date_taken"].apply(timeOfDay)

    if use_credibility and "2014" in working_dataset:
        print "Reading credibility...."
        credfiles = glob.glob(dtmap["CRED_PATH"] + "*.xml")
        creddict = credibility_XMLFile_to_dict(credfiles[0])
        for credfile in credfiles[1:]:
            credother = credibility_XMLFile_to_dict(credfile)
            for k,v in credother.iteritems():
                creddict[k].append(v[0])
        creddt = pd.DataFrame(creddict)
        dataset = pd.merge(dataset, creddt, on=["userid"])
        print "Done reading credibility"

    dataset["poiId"] = dataset["poiName"].apply(lambda x: topic_map[x])
    dataset["source"] = dtmap["SOURCE"]
    
    return dataset, topic_map, header
################################################################################

###############################################################################
#####-------------------LOAD DATA METHODS ENDS---------------------------######
###############################################################################
#
###############################################################################
#####-----------------CLUSTERING METHODS BEGINS--------------------------######
###############################################################################

def get_number_of_clusterss(dataset, num_clusters):
    if dataset.shape[0] < num_clusters:
        return dataset.shape[0] 
    return num_clusters
################################################################################

def flickr(dataset):
    return map(int, dataset[["id","rank"]].sort(columns=["rank"], axis=0)["id"].values)
################################################################################

def oracle(dataset):
    ranked = []
    groups = dataset[dataset.rel == 1].groupby("grp").groups
    done = False
    while not done:
        done = True
        for k in groups.keys():
            if len(groups[k]) > 0:
                done = False
                row_id = groups[k].pop()
                ranked.append( int(dataset.ix[row_id]["id"]) )
    #All the rest
    ranked.extend( map(int, dataset[dataset.rel == 0]["id"].values) )
    return ranked
################################################################################

def spectral(num_clusters, similarity):
    #return SpectralClustering(n_clusters=num_clusters, random_state=29, assign_labels="discretize")\
    #        .fit_predict(similarity)
    return SpectralClustering(n_clusters=num_clusters, random_state=29, n_init=100, assign_labels="kmeans")\
            .fit_predict(similarity)
################################################################################

def agglomerative(num_clusters, similarity, dataset, header, text_sim=False):
    if text_sim:
        connectivity = kneighbors_graph(similarity, 5)
    else:
        values = dataset[header]
        connectivity = kneighbors_graph(values, 5)
    
    """
    #Based on images of each users?
    from scipy.sparse import csr_matrix
    users = set(dataset_now["uid"])
    connectivity = np.zeros([ dataset_now.shape[0], dataset_now.shape[0] ])
    for i, user1 in enumerate(dataset_now["uid"]):
        for j, user2 in enumerate(dataset_now["uid"]):
            if user1 == user2:
                connectivity[i][j] = 1
    connectivity = csr_matrix(connectivity)
    """
    return AgglomerativeClustering(n_clusters=num_clusters, connectivity=connectivity, compute_full_tree=True)\
            .fit_predict(similarity)
##########################################################################################################

def createGraph(similarity, sim_keep_above, graph_rule):
    G = nx.Graph()

    #Default (old version): may cause problems
    if graph_rule == "default":
        similarity = (1.0 - similarity)
    elif graph_rule == "minmax":
        mm = MinMaxScaler()
        similarity = mm.fit_transform(1.0 - similarity)
    elif graph_rule == "normalization":
        norm = Normalizer()
        similarity = norm.fit_transform(1.0 - similarity)
    elif graph_rule == "inversed":
        similarity = (similarity)
    
    # Remove similarity small than S
    similarity[similarity < (sim_keep_above)] = 0.0
    similarity = (similarity * 10000).astype(int)

    if similarity.shape[0] == 1:
        G.add_node(0)
        return G
    
    max_size = similarity.shape[0]
    vertice = 0
    for i in xrange(0, max_size):
        for j in xrange(i, max_size):
            if not ALLOW_SELF_LOOP:
                if i == j:
                    continue
            
            v = similarity[i][j]
            #print i,j,v
            if v > 1:
                vertice+=1
                G.add_edge(i, j, weight=v)
    return G
################################################################################


def take_rank_of(dataset, min_max_rank_threshold):
    if dataset.shape[0] <= min_max_rank_threshold:
        return dataset.ix[dataset["rank"].rank().argmax()]["id"].astype(int)
    return dataset.ix[dataset["rank"].rank() == min_max_rank_threshold]["id"].astype(int).values[0]
################################################################################

def minmax(dt, header, num_clusters, sim_metric, min_max_rank_threshold):
    
    final_ranking = []
    dataset = dt.copy()
    dataset["selected"] = pd.Series(0, index=dataset.index)

    dataset = dataset[dataset["rank"] < take_rank_of(dataset, min_max_rank_threshold)]
    
    #The top ranked is the first to be selected
    #dataset[dataset["rank"] == 1]["selected"] = 1 
    min_ranking = dataset["rank"].min()
    dataset.loc[dataset[dataset["rank"] == min_ranking].index, "selected"] = 1
    final_ranking.append(dataset[dataset["rank"] == min_ranking].id.astype(int))
    
    selected = dataset[dataset["selected"] == 1]
    others = dataset[dataset["selected"] == 0]
    
    while not dataset.selected.all():
        # Some options are avaiable here, as taking the mean, max ou sum of similarity. In the paper, they used max (min in this case)
        # however, my personal analysis says that using mean is not that different than using min
        #next_selected_index = cdist(others[header], selected[header], 'cosine').mean(axis=1).argmax()
        next_selected_index = cdist(others[header], selected[header], 'cosine').min(axis=1).argmax()
        id = others.iloc[next_selected_index]["id"].astype(int)
        final_ranking.append(id)
        dataset.loc[dataset[dataset["id"] == id].index, "selected"] = 1
        
        selected = dataset[dataset["selected"] == 1]
        others = dataset[dataset["selected"] == 0]

    return final_ranking
################################################################################

def kmeans_on_similarity(similarity, num_clusters):
    return KMeans(n_clusters=num_clusters, n_init=100, n_jobs=-1, random_state=29).fit(similarity).predict(similarity)
################################################################################

def get_text_similarity(textdt, ids):
    sim = {}
    for i1,i2,v in textdt[["img1","img2","sim"]].values:
        i1, i2 = int(i1), int(i2)
        if i1 not in sim:
            sim[i1] = {}
        if i2 not in sim[i1]:
            sim[i1][i2] = 0.0
        sim[i1][i2] = float(v)
    
    similarity = []
    for id1 in ids:
        simRow = []
        for id2 in ids:
            if id1 not in sim:
                simRow.append(0.0)
            elif id2 not in sim[id1]:
                simRow.append(0.0)
            else:
                simRow.append( sim[id1][id2])
        similarity.append(simRow)
    return np.array(similarity)
################################################################################

def create_similarity_matrix(dataset, header, sim_metric):
    
    values = dataset[header].values
    vs = pdist(values, sim_metric)
    
    if (vs>1.0).any():
        print "CHECK IT: there is at least one value in the similarity table greater than 1.0"
        print "Max %.3f -- Header[0] = %s" % (vs.max(), header[0])
    
    if np.isnan(np.sum(vs)):
        print "WARNING: similarity has NAN values..."   

    #vs[vs>1.0] = 1.0
    similarity = scipy.spatial.distance.squareform(vs)
    return similarity
################################################################################

def metis_clusters(G, num_clusters, additionalClusters=2):
    G.graph['edge_weight_attr'] = 'weight'
    #G = metis.networkx_to_metis(G)
    if len(G.nodes()) == 0:
        print "Error! Graph has 0 nodes"
        sys.exit()

    elif len(G.nodes()) == 1:
        return [0]

    elif len(G.nodes()) < num_clusters:
        return range(len(G.nodes()))

    (edgecuts, parts) = metis.part_graph(G, nparts=num_clusters + additionalClusters, recursive=False, niter=1000, ufactor=700)
    return parts
################################################################################

###############################################################################
#####-----------------CLUSTERING METHODS ENDS----------------------------######
###############################################################################
#
###############################################################################
#####-----------------DEBUGING METHODS BEGINS----------------------------######
###############################################################################

def get_similarity_element_group(element, group, sim_metric):
    if len(group.shape) <= 1:
        r = cdist([element], [group], sim_metric)
    else:
        r = cdist([element], group, sim_metric)
    return (r.min(), r.max(), r.mean(), np.percentile(r, 25), np.median(r), np.percentile(r, 75), np.percentile(r,90))
################################################################################

def show_original_rank_group(rank, id_rank,id_group, header, dataset):        
    #element = dataset[dataset["id"] == rank[0]][header].astype(float).values[0]
    #group = element
    print "1 it was", id_rank[rank[0]], id_group[rank[0]], dataset[dataset["id"] == rank[0]]["uid"]

    for i, r in enumerate(rank[1:]):
        #element = dataset[dataset["id"] == r][header].astype(float).values[0]
        #sim = get_similarity_element_group(element, group, SIM_METRIC[0])
        #group = np.vstack((group,element))
        #print i + 2, " it was ", id_rank[r], id_group[r], "sim:", sim
        print i + 2, " it was ", id_rank[r], id_group[r], dataset[dataset["id"] == r]["uid"]
################################################################################

###############################################################################
#####-----------------DEBUGING METHODS ENDS------------------------------######
###############################################################################
#
###############################################################################
#####-----------------GENERAL METHODS BEGINS-----------------------------######
###############################################################################

def get_id_rank(dataset):
    id_rank = {}
    for id, rank in dataset[["id","rank"]].values:
        id_rank[int(id)] = float(rank)
    return id_rank
################################################################################

def get_id_group(dataset):
    id_group = {}
    for id, group in dataset[["id","grp"]].values:
        id_group[int(id)] = int(group)
    return id_group
################################################################################

def sort_bt_rank(id_rank, inparts, ids):
    final_rank = list()
    
    if type(inparts) == list:
        parts = inparts[:]
    else:
        parts = inparts.copy()

    clusters = Counter(parts).keys()
    while sum(parts) != (-1 * len(parts)):

        #This part below can be divided into 2 steps:
        # First step: For each cluster, it will search the best ranked for that cluster, building a final list of best ranked "first_step"
        # Second step: Re-rank the best ranked documents.
        first_step = []
        for cluster_num in clusters:

            cluster_now = [n for n,p in enumerate(parts) if p == cluster_num]
            if len(cluster_now) == 0:
                #cluster already done
                continue
            
            r_min = 100000
            id_selected = -1
            for n in cluster_now:
                r = id_rank[ids[n]]
                if r < r_min:
                    r_min = r
                    id_selected = n
            
            parts[id_selected] = -1
            first_step.append( (ids[id_selected], r_min) )
        #Second Step starts here:
        second_step = sorted(first_step, key=lambda x:x[1])
        for (id, rank) in second_step:
            final_rank.append(id)
        
    return final_rank
################################################################################

def write_output(filename, mode, rankedlist, topic_map, poi, max_elements):
    with open(filename, mode) as f:
        for i, id in enumerate(rankedlist[:max_elements]):
            f.write("%d Q0 %d %d %f bla\n" % (topic_map[poi], id, i + 1, 1.0/(i+1)))
################################################################################

def load_old_run(filename, dataset):
    oids = set(dataset["id"])
    dataset = dataset.set_index(["id","poiId"])
    ids = []
    ranks = []
    with open(filename,"r") as f:
        for row in f:
            fields = row.split()
            tid = int(fields[0])
            nid = int(fields[2])
            newr = int(fields[3]) 

            if nid in oids:
                ids.append((nid,tid))
                ranks.append(newr)
    #26 Q0 3338743092 0 1.000000 bla
    
    dataset["rank"] = np.nan
    dataset.ix[ids,"rank"] = ranks
    dataset = dataset[~dataset["rank"].isnull()]

    return dataset.reset_index()
################################################################################


###############################################################################
#####-----------------GENERAL METHODS ENDS-------------------------------######
###############################################################################

###############################################################################
#####-----------------------LOAD DATA BEGINS-----------------------------######
###############################################################################

if WORKING_DATASET == 'dev2013':
    DT_MAP = DEV_2013
elif WORKING_DATASET == 'test2013':
    DT_MAP = TEST_2013
elif WORKING_DATASET == 'dev2014':
    DT_MAP = DEV_2014
elif WORKING_DATASET == 'test2014':
    DT_MAP = TEST_2014

if not args.useGT:
    print "Not using Ground Truth data"
    DT_MAP["RGT_PATH"] = None
    DT_MAP["DGT_PATH"] = None
else:
    print "Using Ground Truth data"

"""
#Code to eliminate duplicates
for id in gtdcsv.id.values:                                                                                                                                        
    if gtdcsv[gtdcsv.id == int(id)].shape[0] > 1:
        if gtdcsv[gtdcsv.id == int(id)].groupby("poiName").last().shape[0] == 1:                                                                                           
            print id
"""
dataset, topic_map, header = create_dataset(DT_MAP, VISUAL_DESCRIPTORS, OTHER_DESCRIPTORS, WORKING_DATASET, USE_CREDIBILITY)

names = set(dataset["username"])
name_uid = dict([(n,i) for i,n in enumerate(names)])
dataset["uid"] = dataset["username"].apply(lambda x:name_uid[x])

#if I want to load a run file created by Navid
if APPLY_NEW_RANK_FILE:
    dataset = load_old_run(APPLY_NEW_RANK_FILE, dataset)

if SUPERVISED:
    dev_dataset, _, _ = create_dataset(get_map(SUPERVISED_DATASETS[0]), VISUAL_DESCRIPTORS, OTHER_DESCRIPTORS, SUPERVISED_DATASETS[0], USE_CREDIBILITY)

    for trainset in SUPERVISED_DATASETS[1:]:
        other_dev_dataset, _, _ = create_dataset(get_map(trainset), VISUAL_DESCRIPTORS, OTHER_DESCRIPTORS, trainset, USE_CREDIBILITY)
        dev_dataset = pd.concat([dev_dataset, other_dev_dataset])

if USE_FILTER:
    if SUPERVISED:
        dataset, model = filterDT(dataset, topics_path=DT_MAP['TOPICS_PATH'], supervised=True, rerank=False, 
                           train_dataset=dev_dataset)
    else:
        dataset, _ = filterDT(dataset, topics_path=DT_MAP['TOPICS_PATH'], supervised=False, rerank=False)

if ONLY_RELEVANT:
    dataset = dataset[dataset.rel == True]
       
if "buid" in OTHER_DESCRIPTORS:
    header.remove("buid")
    for i in xrange(MAX_USERS_PER_POI):
        header += ["buid" + str(i)]
        dataset["buid" + str(i)] = 0

if USE_TEXT_SIM:
    print "Loading text similarity"
    #TODO: add code to merge TEST set as well
    textSimDT = pd.read_csv(DT_MAP["TEXT_SIM"], names=["tid","img1","img2","sim"])


###############################################################################
#####-----------------------LOAD DATA ENDS-------------------------------######
###############################################################################


###############################################################################
#####----------------------REAL CLUSTERING BEGINS------------------------######
###############################################################################
amis, arss, homo, comp, vmetric = [], [], [], [], []
pois = set(dataset["poiName"])
for ipoi, poi in enumerate([poi for poi, _ in sorted(topic_map.iteritems(), key=lambda x:x[1])]):
    if DEBUG:
        if WORKING_DATASET == "dev2014" and poi != "angkor_wat": 
            continue
        #if WORKING_DATASET == "test2013" and poi != "Angel Falls Venezuela": 
        if WORKING_DATASET == "test2013" and poi != "Wainwright Building Missouri": 
            continue
    dataset_now = dataset[dataset.poiName == poi]
      
    if "buid" in OTHER_DESCRIPTORS:
        uids = set(dataset_now["uid"])
        if len(uids) > MAX_USERS_PER_POI:
            print "ERROR! Increase the number of users per POI"

        for i, uid in enumerate(uids):
            dataset_now["buid" + str(i)] = dataset_now["uid"].apply(lambda x:x==uid)
    
    if USE_CREDIBILITY and "2014" in WORKING_DATASET:
        #Here I should use the credibility to ML and remove the not relevant documents.
        if WORKING_DATASET == "dev2014":
            dataset_other = dataset[dataset.poiName != poi]
        elif WORKING_DATASET == "test2014":
            dataset_other = dev_dataset[dev_dataset.source == "dev2014"]

        #credFieldsForML = ['bulkProportion', 'faceProportion', 'locationSimilarity', 'photoCount', 'tagSpecificity', 'uniqueTags', 'uploadFrequency', 'visualScore']
        credFieldsForML = ['faceProportion',"locationSimilarity","uploadFrequency","bulkProportion"]
        
        for visualDesc in ["CN"]:
            h = [h for h in header if re.match(visualDesc + "\d+$", h)]
            credFieldsForML += h

        X = dataset_other[credFieldsForML].astype(float).values
        if WORKING_DATASET == "dev2014" and not args.useGT:
            print "PLEASE IT IS NECESSARY TO RUN AGAIN USING THE GROUND TRUTH DATASET (-u)"
            sys.exit(0)
        y = dataset_other["rel"].astype(bool).values
        X_test = dataset_now[credFieldsForML].astype(float).values
        #should not have for test2014
        if WORKING_DATASET == "dev2014":
            y_test = dataset_now["rel"].astype(bool).values

        model = LogisticRegression(random_state=42, fit_intercept=False, class_weight='auto', C=0.1)
        model.fit(X,y)
        
        #preds = model.predict(X_test)
        probas = model.predict_proba(X_test)
        preds = probas[:,1] > 0.5

        if WORKING_DATASET == "dev2014":
            print "Results: F1", metrics.f1_score(y_test, preds) 
            print "Results: Precision", metrics.precision_score(y_test, preds) 
            print "Results: Recall", metrics.recall_score(y_test, preds) 
            print "Results: Accuracy", metrics.accuracy_score(y_test, preds)
            print "Initial number of examples: %d - set as true: %d (%.3f) " % ( len(preds), preds.sum(), 1.0 * preds.sum() / len(preds))
    
            rel, nrel = (dataset_now["rel"] == 1).sum(), (dataset_now["rel"] == 0).sum()
            if rel + nrel > 0:
                print "Original stats: %d (%.3f) true relevant, %d (%.3f) not relevant" % (rel, 1.0*rel/(rel+nrel), nrel, 1.0*nrel/(rel+nrel))
            
        #remove the files
        #dataset_now = dataset_now[preds == 1]

        #just put them at a very low rank
        new_ranking = np.where(preds==True, 0, 999 + np.array(range(0,preds.shape[0])))
        dataset_now["rank"] = np.where(preds==True, dataset_now["rank"], new_ranking)
        #print dataset_now["rank"].head(5), preds[0:5]
            
        if WORKING_DATASET == "dev2014":
            rel, nrel = (dataset_now["rel"] == 1).sum(), (dataset_now["rel"] == 0).sum()
            if rel + nrel > 0:
                print "New stats: %d (%.3f) true relevant, %d (%.3f) not relevant" % (rel, 1.0*rel/(rel+nrel), nrel, 1.0*nrel/(rel+nrel))
            
        #dataset_now["rank"] = probas[probas[:,1] > 0.5][:,1]


    dataset_now.sort("rank", inplace=True)
    print "Calculating ", poi, ipoi + 1, "/", len(pois)
    #Add some randomness
    ids = map(int, dataset_now["id"].values)
    id_rank = get_id_rank(dataset_now)
    num_clusters = get_number_of_clusterss(dataset_now, DT_MAP['NUM_CLUSTERS'])

    if METHOD == "flickr":
        ranked = flickr(dataset_now)

    elif METHOD == "oracle":
        ranked = oracle(dataset_now)
    
    elif METHOD == "filterOnly":
        MAX_ELEMENTS = 10000
        ranked = flickr(dataset_now)
    
    elif METHOD == "metis":
        # It is hard to explain why, but it works better if you dont reverse sim matrix (1.0 - similarity)
        similarity = create_similarity_matrix(dataset_now, header, SIM_METRIC[0])
        G = createGraph(similarity, SIM_KEEP_ABOVE, GRAPH_RULE)
        
        parts = metis_clusters(G, DT_MAP['NUM_CLUSTERS'])
        ranked = sort_bt_rank(id_rank, parts, ids)
    
    elif METHOD == "kmeans":
        #This is just the original kmeans 
        values = dataset[header].values
        parts = KMeans(n_clusters=num_clusters, n_init=100, n_jobs=-1, random_state=29).fit(values).predict(values)
        ranked = sort_bt_rank(id_rank, parts, ids)
 
    elif METHOD == "simkmeans":
        #This is kmeans applied on the similiarity matrix 
        similarity = create_similarity_matrix(dataset_now, header, SIM_METRIC[0])
        parts = kmeans_on_similarity(similarity, DT_MAP['NUM_CLUSTERS'])
        ranked = sort_bt_rank(id_rank, parts, ids)
   
    elif METHOD == "min-max" or METHOD == "minmax":
        ranked = minmax(dataset_now, header, DT_MAP['NUM_CLUSTERS'], SIM_METRIC[0], MIN_MAX_RANK_THRESHOLD)

    elif METHOD == "textsim":
        textSimDT_now = textSimDT[textSimDT["tid"] == topic_map[poi]]
        similarity = get_text_similarity(textSimDT_now, ids)

        # Both are very good strategies, but metis impacts less in the precision score
        #parts = agglomerative(get_number_of_clusterss(dataset_now, DT_MAP["NUM_CLUSTERS"]), similarity, dataset_now, header, True)
        G = createGraph(similarity, SIM_KEEP_ABOVE, GRAPH_RULE)
        parts = metis_clusters(G, DT_MAP['NUM_CLUSTERS'])

        ranked = sort_bt_rank(id_rank, parts, ids)

    elif METHOD == "multiGraph" or METHOD == "multigraph":
        parts = {}
        sim_h = np.zeros([dataset_now.shape[0], dataset_now.shape[0]]) 

        if AUXILIAR_METHOD != "all":
            for metric in SIM_METRIC:
                for desc in VISUAL_DESCRIPTORS:
                    h = [h for h in header if re.match(desc + "\d+$", h) ]
                    #sim_h += create_similarity_matrix(dataset_now, h, metric)
                    sim_h += (create_similarity_matrix(dataset_now, h, metric))
                    #G[metric][desc] = createGraph(sim_h, SIM_KEEP_ABOVE, GRAPH_RULE)
                #parts[desc] = metis_clusters(G[desc], DT_MAP['NUM_CLUSTERS'])

        done = False
        if AUXILIAR_METHOD == "metis":
            G = createGraph(sim_h, SIM_KEEP_ABOVE, GRAPH_RULE)
            nparts = metis_clusters(G, DT_MAP['NUM_CLUSTERS'])
        elif AUXILIAR_METHOD == "affinity":
            nparts = AffinityPropagation(convergence_iter=20, damping=0.50, max_iter=500, verbose=True).fit_predict(sim_h)
        
        elif AUXILIAR_METHOD == "agglomerative":
            nparts = agglomerative(get_number_of_clusterss(dataset_now, DT_MAP["NUM_CLUSTERS"]), sim_h, dataset_now, header) 
        
        elif AUXILIAR_METHOD == "spectral":
            nparts = spectral(get_number_of_clusterss(dataset_now, DT_MAP["NUM_CLUSTERS"]), sim_h)  
        
        elif AUXILIAR_METHOD == "all":
            nparts = []
           
            for metric in SIM_METRIC:
                for desc in VISUAL_DESCRIPTORS:
                    h = [h for h in header if re.match(desc + "\d+$", h) ]
                    sim_h = (create_similarity_matrix(dataset_now, h, metric))
                    nclusters = get_number_of_clusterss(dataset_now, DT_MAP["NUM_CLUSTERS"])
                    nparts.append( agglomerative(nclusters, sim_h, dataset_now, header) )
                    try:
                        nparts.append( spectral(nclusters, sim_h) )
                    except ValueError:
                        print "Ignoring spectral"

                    G = createGraph(sim_h, SIM_KEEP_ABOVE, GRAPH_RULE)
                    nparts.append( metis_clusters(G, DT_MAP['NUM_CLUSTERS']) )
                    #nparts.append( AffinityPropagation(convergence_iter=20, damping=0.50, max_iter=500, verbose=True).fit_predict(sim_h) )

            if USE_TEXT_SIM:
                print "USING TEXT SIM"
                textSimDT_now = textSimDT[textSimDT["tid"] == topic_map[poi]]
                sim_h = get_text_similarity(textSimDT_now, ids)
                nparts.append( agglomerative(get_number_of_clusterss(dataset_now, DT_MAP["NUM_CLUSTERS"]), sim_h, dataset_now, header, True) )
                #nparts.append( spectral(get_number_of_clusterss(dataset_now, DT_MAP["NUM_CLUSTERS"]),  sim_h) )
                G = createGraph(sim_h, SIM_KEEP_ABOVE, GRAPH_RULE)
                nparts.append( metis_clusters(G, DT_MAP['NUM_CLUSTERS']) )

            forbmap = {}
            for i in ids:
                forbmap[i] = {}
            
            for part in nparts:
                for cluster in xrange(DT_MAP["NUM_CLUSTERS"]):
                    sameCluster = [i for i,c in enumerate(part) if c == cluster]
                    for element in sameCluster:
                        for oelement in sameCluster:
                            if ids[oelement] not in forbmap[ids[element]]:
                                forbmap[ ids[element] ][ids[oelement]] = 0
                            forbmap[ ids[element] ][ids[oelement]] += 1
                    
            #TODO: normalize forbmap
            MULTIGRAPH_MEAN = MULTIGRAPH_MEAN_THRESHOLD
            MULTIGRAPH_MIN = MULTIGRAPH_MIN_THRESHOLD
            MULTIGRAPH_MAX = MULTIGRAPH_MAX_THRESHOLD
            MULTIGRAPH_SUM = 0
            # this ids should be already ranked by the "rank" colunm
            ranked = []
            
            if args.useGT and WORKING_DATASET != "test2014":
                idgrp = get_id_group(dataset_now)
                groups = set()
                print "Grp Id0:%d" % (idgrp[ids[0]])
            
            alreadyIncluded = set()
            while len(ranked) < len(ids):
                #Early break, we just need the first 50 results
                if len(ranked) > 50:
                    break

                for nid, id in enumerate(ids):
                    if id in alreadyIncluded:
                        continue

                    possible = True
                    foundvect = []
                    
                    # Look at the past examples
                    for takenId in ranked:
                        if id in forbmap[takenId]:
                            foundvect.append(forbmap[takenId][id])
                        else:
                            foundvect.append(0)

                    if len(foundvect) > 0:
                        vmean, vmax = np.mean(foundvect), np.max(foundvect)
                        nozeros = [v for v in foundvect if v > 0]
                        if len(nozeros) > 0:
                            vmin = np.min(nozeros)
                        else:
                            vmin = 0
                        vsum = np.sum(foundvect)

                        if vmean > MULTIGRAPH_MEAN:
                            #pass
                            possible = False
                            #print "Rejected by the MEAN:", vmean, ">=", MULTIGRAPH_MEAN
                    
                        if vmax > MULTIGRAPH_MAX:
                            possible = False
                            #print "Rejected by the MAX:", vmax, ">=", MULTIGRAPH_MAX

                        if vmin > MULTIGRAPH_MIN:
                            #pass
                            possible = False
                            #print "Rejected by the MIN:", vmean, ">=", MULTIGRAPH_MIN
                        
                        """
                        if vsum > MULTIGRAPH_SUM_THRESHOLD:
                            possible = False
                            print "Rejected by the SUM:", vsum, ">=", MULTIGRAPH_SUM_THRESHOLD
                        """
                        #print "#Ranking: %d, Min: %.2f, Mean: %.2f, Max: %.2f, Sum: %.2f, Accepted:%d, grp:%d" % (nid, vmin, vmean, vmax, vsum, possible, idgrp[id])

                    if possible:
                        ranked.append(id)
                        alreadyIncluded.add(id)
                        if args.useGT and WORKING_DATASET != "test2014":
                            groups.add(idgrp[id])
                
                MULTIGRAPH_MEAN += MULTIGRAPH_MEAN_INCREMENT
                MULTIGRAPH_MAX += MULTIGRAPH_MAX_INCREMENT
                MULTIGRAPH_MIN += MULTIGRAPH_MIN_INCREMENT
                MULTIGRAPH_SUM += 0.00

                #print "MultiGraph - min %.2f, mean %.2f max %.2f - accepted: %d (%.2f)" % (MULTIGRAPH_MIN_THRESHOLD, MULTIGRAPH_MEAN_THRESHOLD, 
                #                                                                           MULTIGRAPH_MAX_THRESHOLD, len(ranked), len(ranked)/len(ids))
                #print "Ngroups", len(groups), "Present groups:", groups

            done = True        

        if not done:
            ranked = sort_bt_rank(id_rank, nparts, ids)

    elif METHOD == "affinity":
        #values = dataset[header].values
        distance = 1.0 - create_similarity_matrix(dataset_now, header, SIM_METRIC[0])
        parts = AffinityPropagation(convergence_iter=20, damping=0.50, max_iter=500, verbose=True).fit_predict(distance)
        print len(set(parts)), "graphs created by affinity"
        ranked = sort_bt_rank(id_rank, parts, ids)
   
    elif METHOD == "agglomerative":
        
        distance = create_similarity_matrix(dataset_now, header, SIM_METRIC[0])
        similarity = 1.0 - distance
        #beta = -1.0
        #similarity = np.exp(beta * distance / distance.std())
        parts = agglomerative(get_number_of_clusterss(dataset_now, DT_MAP['NUM_CLUSTERS']), similarity, dataset_now, header) 
        ranked = sort_bt_rank(id_rank, parts, ids)

    elif METHOD == "spectral":
        #similarity = np.exp(-beta * distance / distance.std())
        #http://scikit-learn.org/stable/modules/clustering.html#spectral-clustering
        distance = create_similarity_matrix(dataset_now, header, SIM_METRIC[0])
        similarity = 1.0 - distance
        #using beta seems a good alternative
        #beta = -0.1
        #beta = -1
        #similarity = np.exp(beta * distance / distance.std())
        
        parts = spectral(get_number_of_clusterss(dataset_now, DT_MAP['NUM_CLUSTERS']), similarity)  
        ranked = sort_bt_rank(id_rank, parts, ids)

    if args.useGT and WORKING_DATASET != "test2014" and METHOD not in ["joao","flickr","filterOnly","multigraph"]:
        true_values = dataset_now["grp"].astype(int).values
        ami = metrics.adjusted_mutual_info_score(true_values, parts)
        ars = metrics.adjusted_rand_score(true_values, parts)
        homogeneity = metrics.homogeneity_score(true_values, parts)
        completess = metrics.completeness_score(true_values, parts)
        v = metrics.v_measure_score(true_values, parts)
        amis.append(ami)
        arss.append(ars)
        homo.append(np.mean(homogeneity))
        comp.append(np.mean(completess))
        vmetric.append(np.mean(v))
        print ami, ars, np.mean(homo), np.mean(comp), np.mean(v)
       
    if args.useGT and WORKING_DATASET != "test2014":
        id_group = get_id_group(dataset_now)
        #show_original_rank_group(ranked, id_rank, id_group, header, dataset_now)

    print "Using at least", len(ranked), "examples" 
    if len(ranked) < 50:
        print "CHECK IT!!!!!!!!!!!! THERE ARE LESS THAN 50 EXAMPLES"
    write_output(OUTFILE, "a", ranked, topic_map, poi, MAX_ELEMENTS)

if args.useGT and WORKING_DATASET != "test2014" and METHOD not in ["joao", "multigraph"]:
    print "Mean adjusted rand score:" , np.mean(arss)
    print "Mean adjusted mutual info score:" , np.mean(amis)
    print "Mean homogeneity:" , np.mean(homo)
    print "Mean completeness:" , np.mean(comp)
    print "Mean V metric:" , np.mean(vmetric)

if WORKING_DATASET != "test2014":
    print "P10,CR10,F10,P20,CR20,F20"
    evaluate(OUTFILE, WORKING_DATASET, EVAL_CSV, [METHOD, WORKING_DATASET,USE_FILTER,SUPERVISED,\
                                             SUPERVISED_DATASETS,VISUAL_DESCRIPTORS,OTHER_DESCRIPTORS,\
                                             SIM_METRIC,USE_CREDIBILITY,APPLY_NEW_RANK_FILE,\
                                            AUXILIAR_METHOD, MULTIGRAPH_MIN_THRESHOLD, MULTIGRAPH_MEAN_THRESHOLD, MULTIGRAPH_MAX_THRESHOLD,\
                                             MULTIGRAPH_MIN_INCREMENT, MULTIGRAPH_MEAN_INCREMENT, MULTIGRAPH_MAX_INCREMENT])

"""
    if WORKING_DATASET == "test2013":
        if poi == "Castle Estense Ferrara":
            f = open("out", "w")
            for i in xrange(0, len(ids)):
                #print ids[i], " belongs to ", parts[i]
                #print "cp ../archive2013/testset/keywords/img/" + poi.replace(" ","\ ") + "/" + str(ids[i]) + ".jpg  outs/" + str(parts[i])
                f.write("cp ../archive2013/testset/keywords/img/" + poi.replace(" ","\ ") + "/" + str(ids[i]) + ".jpg  outs/" + str(parts[i]) + "\n")
            f.close()
"""
##########################################################################################################

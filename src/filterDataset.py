from __future__ import division
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import glob

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import ShuffleSplit, KFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.svm import SVC

from types import NoneType

##########################################################################################################
def filterDT(old_dataset, topics_path, supervised, rerank, train_dataset=None):
    #I found some 'rel' values set to -1 in 2013 collection. Treating them as 0
    dataset = old_dataset.copy()
    
    if "rel" in dataset.keys():
        dataset["rel"] = dataset["rel"].apply(lambda x: x == 1)
   
    if type(train_dataset) != NoneType and "rel" in train_dataset.keys():
        train_dataset["rel"] = train_dataset["rel"].apply(lambda x: x == 1)
    
    def getDiffGrps(dt):
        total = 0
        for poi in set(dt["poiName"]):
            total += len(set(dt[dt["poiName"] == poi]["grp"]))
        return total
    
    if not supervised:
        print "Filtering only based on GPS data"

        #rel, nrel = (dataset["rel"] == 1).sum(), (dataset["rel"] == 0).sum()
        #print "Initial stats: %d (%.3f) true relevant, %d (%.3f) not relevant" % (rel, 1.0*rel/(rel+nrel), nrel, 1.0*nrel/(rel+nrel))

        #print "Grps before: ", getDiffGrps(dataset)
        dataset = dataset[dataset["dist"] < 10 ]
        dataset = dataset[dataset["views"] > 0]
        dataset = dataset[dataset["descLength"] < 2000]
        #dataset = dataset[dataset["descLength"] < 500]
        #rel, nrel = (dataset["rel"] == 1).sum(), (dataset["rel"] == 0).sum()
        #print "Pos-filter stats: %d (%.3f) true relevant, %d (%.3f) not relevant" % (rel, 1.0*rel/(rel+nrel), nrel, 1.0*nrel/(rel+nrel))
        #print "Grps after filtering: ", getDiffGrps(dataset)

        #dataset = dataset[dataset["nbComments"] > 0]
        #dataset = dataset[(dataset.nbComments > 1) & (dataset.dist < 10)]
        #new_dataset = new_dataset[new_dataset["dist"] > 0]
        return dataset, None
    else:
        #values to be used for ML
        #header = ["dist", "views", "descLength", "license", "titleLength", "poiInDescription", "poiInTitle"]
        
        #header = ["dist", "views", "descLength"]
        header = ["dist", "views", "descLength", "titleLength", "poiInDescription", "poiInTitle", "partOfTheDay", "license"]
        #Not helped for test2013: rank, nbComments, partOfTheDay, poiInTags, poiInTitle, tagsLength

        def model32(value, maxv):
            return int(32*value/maxv)
        
        def makeBool(value, v):
            if value == v:
                return True
            return False
        
        # Configure the 32 bins for distance
        def makeBins(feature, train_dt, dt, h):
            maxv = max(train_dt[feature].max(), dt[feature].max())
            dt[feature] = dt[feature].apply(model32, args=(maxv,))
            train_dt[feature] = train_dt[feature].apply(model32, args=(maxv,))
            for v in xrange(0,32):
                dt[feature + str(v)] = dt[feature].apply(makeBool, args=(v,))
                h.append(feature + str(v))
        
            for v in xrange(0,32):
                train_dt[feature + str(v)] = train_dt[feature].apply(makeBool, args=(v,))
            h.remove(feature)
        
        makeBins("dist", train_dataset, dataset, header)
        makeBins("views", train_dataset, dataset, header)
        makeBins("descLength", train_dataset, dataset, header)
        #dataset["dist"] = dataset["dist"] < 10
        #dataset["views"] = dataset["views"] > 0
        #dataset["descLength"] = dataset["descLength"] < 2000
        
        if "nbComments" in header:
            makeBins("nbComments", train_dataset, dataset, header)
        
        dataset["numTags"] = dataset["tags"].apply(lambda x:len(x.split()))
        train_dataset["numTags"] = train_dataset["tags"].apply(lambda x:len(x.split()))
        
        def poiInTags(row):                                                                                                                                                 
            poi = set(row["poiName"].lower().split())
            tags = set(row["tags"].lower().split())
            return len(poi.intersection(tags))

        #TODO: remove stopwords
        dataset["poiInTags"] = dataset.apply(poiInTags, axis=1)
        train_dataset["poiInTags"] = train_dataset.apply(poiInTags, axis=1)
        
        def poiInTitle(row):                                                                                                                                                  
            poi = set(row["poiName"].lower().split())
            tags = set(row["title"].lower().split())
            return len(poi.intersection(tags)) >= 4

        dataset["poiInTitle"] = dataset.apply(poiInTitle, axis=1)
        train_dataset["poiInTitle"] = train_dataset.apply(poiInTitle, axis=1)
        
        def poiInDesc(row):                                                                                                                                                  
            poi = set(row["poiName"].lower().split())
            tags = set(row["description"].lower().split())
            return len(poi.intersection(tags)) >= 4

        dataset["poiInDescription"] = dataset.apply(poiInDesc, axis=1)
        train_dataset["poiInDescription"] = train_dataset.apply(poiInDesc, axis=1)
        
        train_dataset["titleLength"] = train_dataset["title"].apply((lambda x: len(x)))
        dataset["titleLength"] = dataset["title"].apply((lambda x: len(x)))
        
        train_dataset["tagsLength"] = train_dataset["tags"].apply((lambda x: len(x)))
        dataset["tagsLength"] = dataset["tags"].apply((lambda x: len(x)))
        
        #dataset["dist"] = dataset["dist"].apply(lambda x: x>10)
        #train_dataset["dist"] = train_dataset["dist"].apply(lambda x: x>10)
        #dataset["views"] = dataset["views"].apply(lambda x: x==0)
        #train_dataset["views"] = train_dataset["views"].apply(lambda x: x==0)
        
        print "Filtering based on ML"
        print "Features:", header

        if type(train_dataset) != NoneType:
            print "Train dataset given in the input."
            y_train = train_dataset["rel"].values
            X_train = train_dataset[header].values.astype(float)
            X_test  = dataset[header].values.astype(float)
            
            if "rel" in dataset.keys():
                y_test  = dataset["rel"].values
        else:
            print "TODO: CV 60% train, 40% test?"

        #model = LogisticRegression(random_state=42, fit_intercept=False, class_weight={0:0.6,1:0.4}, C=1)
        # 'auto' is less prone to overfitting 
        #model = LogisticRegression(random_state=42, fit_intercept=False, class_weight='auto', C=1)
        model = LogisticRegression(random_state=42, fit_intercept=False, class_weight=None, C=1)
        #model = ExtraTreesClassifier(random_state=42, n_jobs=-1, n_estimators=1000, min_samples_split=2,)
        # Needs a lot of parameter tunning
        #model = SVC(random_state=42, C=1, probability=False, kernel="rbf", class_weight={0:.6,1:.4})
        #model = SVC(random_state=42, C=1000, probability=False, kernel="rbf", class_weight="auto")
        
        model.fit(X_train,y_train)
        #weights = np.where(y_train == 1, 0.60, 0.40)
        #model.fit(X_train,y_train, sample_weight=weights)

        #predictions = model.predict(X_test)
        probas = model.predict_proba(X_test)
        predictions = (probas[:,1] >= 0.5)
        
        new_dataset = dataset[predictions == 1]

        print "Classifier:", model
        if "rel" in dataset.keys():
            print "Results: F1", f1_score(y_test, predictions) 
            print "Results: Precision", precision_score(y_test, predictions) 
            print "Results: Recall", recall_score(y_test, predictions) 
            print "Results: Accuracy", accuracy_score(y_test, predictions)
            print "Original number of examples: %d - set as true: %d (%.3f) " % ( len(predictions), predictions.sum(), 1.0 * predictions.sum() / len(predictions))
            rel, nrel = (dataset["rel"] == 1).sum(), (dataset["rel"] == 0).sum()
            print "Correct stats: %d (%.3f) true relevant, %d (%.3f) not relevant" % (rel, 1.0*rel/(rel+nrel), nrel, 1.0*nrel/(rel+nrel))
        
        if "rel" in new_dataset.keys():
            rel, nrel = (new_dataset["rel"] == 1).sum(), (new_dataset["rel"] == 0).sum()
            print "New stats: %d (%.3f) true relevant, %d (%.3f) not relevant" % (rel, 1.0*rel/(rel+nrel), nrel, 1.0*nrel/(rel+nrel))
        
        #print "Grps before: ", getDiffGrps(dataset)
        #print "Grps after: ", getDiffGrps(new_dataset)
        
        # Here I only take the ones that I am confident they worked
        #return dataset, predictions
        #stophere
        return new_dataset, model

##########################################################################################################



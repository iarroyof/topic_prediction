from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed
from scipy.sparse import vstack
from operator import itemgetter
import numpy as np
import re
import pandas as pd
import random
import logging
import pymongo
from elasticsearch import Elasticsearch

from pdb import set_trace as st

# 1. ¿Que tiempo minimo promedio necesita una reporte para influir en un nuevo
#    reporte, independientemente de la empresa?
# 2. ¿Que tiempo maximo promedio seria de ayuda para predecir topicos y
#    sentimientos de un nuevo reporte?

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

batch_size = 5
db_addr = "mongodb://192.168.1.165:27017/"
db_name = 'teLaResumo'
coll_name = 'snp500_10q'


class mongo_iterator(object):

    def __init__(self, db_addr, db_name, coll_name,
                        batch_size=1000, n_docs=1000,
                        llave='summary_textrank', sort_by='date'):
        self.db_addr = db_addr
        self.db_name = db_name
        self.coll_name = coll_name
        if isinstance(llave, list):
            self.llave = llave
        else:
            self.llave = [llave]
        client = pymongo.MongoClient(db_addr)
        db = client[db_name]
        collection = db[coll_name]
        self.convert_dec = []
        for ll in self.llave:
            if "decimal" in str(type(collection.find_one().pop(ll))):
                self.convert_dec.append(True)
            else:
                self.convert_dec.append(False)

        __cursor = collection.find().limit(2)
        self.column_names = __cursor.next().keys()
        if len(set(self.llave).intersection(self.column_names)
                ) != len(self.llave):
            print("ERROR: NO column name '{0}' in DB.\n"
                                .format(llave))
            raise NameError("Available keys are: {}".format(self.column_names))

        if sort_by is None:
            pos = set(['date', 'Date', 'Timestamp', 'TimeStamp', 'timestamp'])
            sort_by = list(pos.intersection(self.column_names))[0]

        _sort = [(sort_by, -1)]
        collection.create_index(_sort)
        if batch_size is None:
            self.batch_size = batch_size
            batch_size = 1000
        else:
            self.batch_size = batch_size
        if n_docs is None:
            self.n_docs = collection.count()
        else:
            self.n_docs = n_docs

        self.cursor = collection.find(batch_size=batch_size)\
                                .sort(_sort)\
                                .limit(self.n_docs)
        self.n_yielded = 0

    def chunker(self, seq, size):
        res = []
        for reg in seq:
            try:
                t = itemgetter(*self.llave)(reg)
                if len(self.llave) > 1:
                    res.append(tuple(float(k.to_decimal()) if r else k
                                        for k, r in zip(t, self.convert_dec)))
                else:
                    res.append(float(t.to_decimal()) \
                                        if self.convert_dec[0] else t)
               # res.append(itemgetter(*self.llave)(el))
            except KeyError:
                continue
            except AttributeError:
                continue

            if len(res) == size:
                yield res
                res = []
        if res:
            yield res

    def __iter__(self):
        if self.batch_size is None:
            for self.n_yielded, reg in zip(range(self.n_docs), self.cursor):
                try:
                    t = itemgetter(*self.llave)(reg)
                    if len(self.llave) > 1:
                        yield tuple(float(k.to_decimal()) if r else k
                                        for k, r in zip(t, self.convert_dec))
                    else:
                        yield float(t.to_decimal()) if self.convert_dec[0] \
                                                    else t
                except KeyError:
                    yield None
        else:
            for batch in self.chunker(self.cursor, self.batch_size):
                yield batch

    def __next__(self):

        if self.n_yielded > self.n_docs:
            raise StopIteration

        self.n_yielded += 1
        try:
            t = itemgetter(*self.llave)(self.cursor.next())
            yield tuple(float(k.to_decimal()) if r else k
                                        for k, r in zip(t, self.convert_dec))
        except KeyError:
            yield None


class ElasticIterator(object):

    def __init__(self, n_docs=1000, host_port='192.168.55.165:9200', 
                    index='newsfeed', collection='test', data_source='headlines',
                    query={'Headline':'india', 'HotLevel': '1'},
                    array_name='Story', time_field='Timestamp'):
        self.index = index
        self.time_field = time_field
        self.llave = list(query.keys())
        self.query = list(query.values())
        self.array_name = array_name
        self.paths = {k: [] for k in llave}
        self.root_path = 'hits.hits._source.'
        self.filter_paths = [self.root_path.split('_')[0] + '_type',
                            self.root_path + 'ADD_STORY',
                            self.root_path + time_field,
		                    ]
        self.llave = [self.root_path + array_name + '.' + ll for ll in llave] \
                        if isinstance(llave, list) \
		                else [llave]
        
        self.data_source = data_source
        self.collection = collection
        self.time_field = time_field
        self.n_docs = n_docs
        self.query = self.get_es_query()

        if self.query is None:
            print("Try with another query as the given one gives no results...")
        else:
            self.print_keypaths()    

    def print_keypaths(self):
        for ll in llave:
            self.key_path(self.query, ll, path=index + '.' + collection)
            key_paths = []
            for i, p in enumerate(self.paths[ll]):
                pa = re.sub('\d+', '%d', p)
                key_paths.append(pa.split('=>')[0])
            self.paths[ll] = key_paths
            print('Key paths in DB for "{}"'.format(ll))
            print(set(self.paths[ll]))
        

    def get_es_query(self):
        must_list = [   { "match":
            		        { "Event": "ADD_STORY" \
                                if self.data_source == 'headlines'\
						        else "ADD_1STPASS"}
        		            },
        	            { "match":
            		        { "_type": self.collection }
        		        }
              	    ]
        #assert len(llave) == len(query) # Each field needed needs a query term.
        for ll, q in zip(self.llave, self.query):
            match = '.'.join(ll.split('.')[-2:])
            must_list.append({"match": {match: q}})
            
        es = Elasticsearch([{'host': host_port.split(':')[0],
                                'port':int(host_port.split(':')[1])}])
        
        query = es.search(index=self.index, 
                                filter_path=self.filter_paths + self.llave,
				                body={"query": {
    					                "bool": {"must": must_list}
                                          }},
                                sort=self.time_field + ":desc",
                                size=self.n_docs)

        for k in self.root_path.split('.'):
            try:
                query = query[k]
                if isinstance(query[k], list):
                    query = query[k]
                    break
            except KeyError:
                print("Empty results returned by given query...")
                break

        return None if query == {} else query:

    def key_path(self, my_dict, key, path='query'):

        for k, v in my_dict.items():
            if isinstance(v,list):
                for i, item in enumerate(v):
                    self.key_path(item, key, path + "." + k + "." + str(i))
            elif isinstance(v,dict):
                self.key_path(v, key, path + "." + k)
            else:
                if k == key:
                    if isinstance(v, str):
                        va = v[:min(10, len(v))]
                        self.paths[key].append("=>".join([path + "." + k, str(va)]))
                    else:
                        self.paths[key].append("=>".join([path + "." + k, str(v)]))

    def __iter__(self):

        for q in self.query:
            qq = q[self.root_path.split('.')[-2]]
            result = {r: qq[self.array_name][0][r] 
                        for r in qq[self.array_name][0]}
            result[self.time_field] = qq[self.time_field]
            yield result

# Get dayly headlines/bodies example:
# https://www.interviewqs.com/ddi_code_snippets/select_pandas_dataframe_rows_between_two_dates

class DailyNews(object):
    def __init__(self, index, collection, time_field='Timestamp', n_docs=10000):
        self.index = index
        self.collection = collection
        self.time_field = time_field
        self.n_docs = n_docs

def build_query(self, query, n_days=1000, start_day="dd/mm/yyyy"):
    self.elastic_docs = ElasticIterator(index=self.index, collection=self.collection, 
                                query=query,  #{'Headline': keyword}, 
                                time_field=self.time_field, n_docs=self.n_docs)
    self.docsdf = pd.DataFrame([d for d in self.elastic_docs])
    self.docsdf[self.time_field] = pd.to_datetime(self.docsdf[self.time_field])
    if current_day == "dd/mm/yyyy":
        self.current_day = pd.Timestamp(self.docsdf[self.time_field][0].date())
    else:
        print("Parsed Timestamp:")
        print(pd.Timestamp(start_day))
        self.current_day = pd.Timestamp(start_day)
    self.n_days = n_days

def __iter__(self):
    for _ in range(self.n_days):
        mask = (self.docsdf[self.time_field].dt.date == self.current_day)
        day_headlines = self.docsdf.loc[mask]
        self.current_day = self.current_day - pd.DateOffset(1)
        if day_headlines.empty:
            continue
        yield day_headlines

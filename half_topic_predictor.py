from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed
from scipy.sparse import vstack
import numpy as np
import pandas as pd

import logging
#from logging import info
from pdb import set_trace as st


logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

class SplitVectorizer():
    def __init__(self, tfidf_model=None, input_file_name=None, Xy='X',
                 type_analyzer='word', n_gram_range=(1, 2), batch_size=100,
                                                             vectorize=False):
        if tfidf_model == None:
            assert input_file_name != None  # Give model or input text
            self.model = TfidfVectorizer(analyzer=type_analyzer,
                                                ngram_range=n_gram_range)
        elif input_file_name == None:
            assert tfidf_model != None  # Give model or input text
            self.model = tfidf_model

        elif not None in [input_file_name, tfidf_model]:
            self.model = tfidf_model

        self.XY = Xy
        self.input_file = input_file_name
        self.vectorize = vectorize
        self.batch_size = batch_size

    def fit(self, X=None, y=None):
        with open(self.input_file) as f:
            self.model.fit(f)

        self.analyzer = self.model.build_analyzer()
        self.prep = self.model.build_preprocessor()
        self.tokenizer = self.model.build_tokenizer()
        self.vocab = {self.model.vocabulary_[w]: w
				for w in self.model.vocabulary_}

        return self

    def get_matrices(self):
        self.docs_X = []
        self.docs_Y = []
        for a in open(self.input_file):
            x = self.tokenizer(self.prep(a))
            dl = len(x)
            self.docs_X.append(" ".join(x[:int(dl/2)]))
            self.docs_Y.append(" ".join(x[int(dl/2):]))
        return self.model.transform(self.docs_X), \
               self.model.transform(self.docs_Y)

    def Tx(self, x):
        if self.vectorize:
            if self.XY == 'join':
                return (self.model.transform([x_[0] for x_ in x]),
                        self.model.transform([x_[1] for x_ in x]))
            else:
                return self.model.transform([x] if isinstance(x, str) else x)
        else:
            return self.analyzer(x)

    def __iter__(self):
        if self.batch_size in [None, 0, 1]:
            for a in open(self.input_file):
                x = self.tokenizer(self.prep(a))
                dl = len(x)

                if self.XY == 'X':
                    yield self.Tx(" ".join(x[:int(dl/2)]))
                elif self.XY == 'Y':
                    yield self.Tx(" ".join(x[int(dl/2):]))
                elif self.XY == 'join':
                    yield self.Tx(" ".join(x[:int(dl/2)])), \
			              self.Tx(" ".join(x[int(dl/2):]))
        else:
            issue = 0
            for a in open(self.input_file):
                if issue == 0:
                    batch = []
                x = self.tokenizer(self.prep(a))
                dl = len(x)
                if self.XY == 'X':
                    batch.append(" ".join(x[:int(dl/2)]))
                elif self.XY == 'Y':
                    batch.append(" ".join(x[int(dl/2):]))
                elif self.XY == 'join':
                    batch.append((" ".join(x[:int(dl/2)]), \
                                        " ".join(x[int(dl/2):])))
                if issue == self.batch_size:
                    issue = -1
                    yield self.Tx(batch)

                issue += 1



def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

class MultivariateSparseSGDRegressor(BaseEstimator):
    def __init__(self, verbose=50, minf=2, estimator_names=None, partial=True,
                        binary=True, pos_th=None, maxf=50, e_batch_size=50, 
                        ret_neg_class_probs=False, 
			fit_b_size=1000):
        self.verbose = verbose
        self.minf = minf
        self.vocab = estimator_names  # format: {idx: 'token'}
        self.binary = binary
        self.pos_th = pos_th
        self.maxf = maxf
        self.ret_neg_class_probs = ret_neg_class_probs
        self.e_batch_size = e_batch_size
        self.partial = partial

    def create_datasets(self, X, y):

        def _select_nonzeros(e, y, th, lessgrea='great'):
            if lessgrea == 'great':
                return e, (y[:, e] > th).nonzero()[0].tolist()
            else:  # less than:
                return e, np.invert((y[:, e] > th).nonzero()[0]).tolist()

        def _remove_items(e):
            quitt = np.where(self.non_zeros_e==e)[0]
            self.non_zeros_e = np.delete(self.non_zeros_e, quitt)
            self.non_zeros_s = np.delete(self.non_zeros_s, quitt)
            #return 

        #def _remove_undersampled(ds, fmin):

        if self.pos_th is None:
            self.pos_th = np.asarray(y.mean(axis=0)).reshape(-1) * 0.2

        if self.verbose: print(pd.DataFrame({
                            'Pos_th': self.pos_th,
                            'Y_mean': np.asarray(y.mean(axis=0)).reshape(-1),
                            'Y_max': y.max(axis=0).toarray().reshape(-1),
                                             }))
        try:
            self.E.keys()
            self.first_fit = False
        except (NameError, AttributeError):
            self.first_fit = True
            self.E = {e: [] for e in list(self.vocab.keys())}

        self.non_zeros_s, self.non_zeros_e = y.nonzero()
        # Nonzeros associated to the given word (estimator) 'e' (positive class)
        if self.verbose: logging.info("Shrinking non-zeroes...")
        #NZs = {e: (y[:, e] > self.pos_th[e]).nonzero()[0].tolist()
        #                                        for e in set(non_zeros_e)}
        #st()
        NZs = dict(Parallel(n_jobs=-1, prefer="threads")(
                        delayed(_select_nonzeros)(e, y, self.pos_th[e])
                                    for e in set(self.non_zeros_e)))
        # Remove estimators that where nonzeroes, but did not hold the minimum
        # tfidf (informativeness) criterion
        if self.minf > 1:
            if self.verbose: logging.info("Remove undersampled non-zeroes...")
            Parallel(n_jobs=-1, prefer="threads")(
                        delayed(_remove_items)(e) for e in NZs 
				if len(NZs[e]) <= self.minf)
            #for e in NZs:
            #    if len(NZs[e]) <= self.minf:
            #        quitt = np.where(self.non_zeros_e==e)[0]
            #        non_zeros_e = np.delete(non_zeros_e, quitt)
            #        non_zeros_s = np.delete(non_zeros_s, quitt)

        if self.verbose: logging.info("Shrinking negative zeroes...")
        nNZs = dict(Parallel(n_jobs=-1, prefer="threads")(
                        delayed(_select_nonzeros)(e, y, self.pos_th[e],
                                                            lessgrea='less')
                             for e in set(self.non_zeros_e))) if self.binary else []
        # Nonzeros associated to different words others than 'e'
        if self.verbose: logging.info("Shrinking negatives...")
        #Nes = {e: list(set([s for (ee, s) in zip(non_zeros_e, non_zeros_s)
        #                if (ee != e and ee not in NZs[e] + nNZs[e])]))
        #                for e in set(non_zeros_e) }
        full_range = set(range(y.shape[0]))
        Nes = {e: list(full_range - set(NZs[e] + nNZs[e]))
                                            for e in set(self.non_zeros_e)}
        if self.verbose: logging.info("Sampling over sampled estimators...")
        for e in set(self.non_zeros_e):
            if len(NZs[e]) > self.maxf:
                NZs[e] = np.random.choice(NZs[e], self.maxf).tolist()
            if len(nNZs[e]) > self.maxf:
                nNZs[e] = np.random.choice(nNZs[e], self.maxf).tolist()
            if len(Nes[e]) > self.maxf:
                Nes[e] = np.random.choice(Nes[e], self.maxf).tolist()

        if self.verbose: logging.info("Integrating each estimator's datasets...")
        for e in set(self.non_zeros_e):
            Xe = [X[NZs[e], :].tocoo(), X[Nes[e], :].tocoo()]  #, X[Zes[e], :]]

            ye = [[1] * len(NZs[e]) if self.binary else y[NZs[e], e],
                 [0] * len(Nes[e]) if self.binary else y[Nes[e], e]]
            if nNZs[e] != []:
                Xe += [X[nNZs[e], :].tocoo()]
                ye += [0] * len(nNZs[e])

            Xe = vstack(Xe).tocsr()
            ye = np.hstack(ye)
            self.E[e] = {'x': Xe, 'y': ye}

        if self.first_fit:  # Remove despreciable/zero-probability samples
            self.E = {e: self.E[e]
                      for e in self.E if e in self.non_zeros_e}

        if self.verbose: logging.info("Number of estimators to train: %d" % len(self.E))

    def fit(self, X, y=None):
        # One estimator for each 'y' dimension and train batches of them

        if self.verbose: logging.info("Creating non-zero datasets...")
        self.create_datasets(X, y)  # Datasets are left in self.E[e]['x','y']
        if self.binary:
            for e in self.E:
                self.E[e]['e'] = SGDClassifier(loss='log',
                                                penalty='elasticnet',
                                                l1_ratio=0.25,
                                                #class_weight='balanced',
                                                 n_jobs=3)
        else:
            for e in self.E:
                self.E[e]['e'] = SGDRegressor()

        if self.verbose: logging.info("Training estimators...")
        # Send the fittig process to _fit() function independent of joblib to
        # avoid serializing the process
        if self.e_batch_size is None:
            self.estimator_vector = Parallel(n_jobs=-1, prefer="threads")(
                delayed(self._fit)(e, self.E[e]['e'],
                                self.E[e]['x'], self.E[e]['y'])
                        for e in self.E)
        else:
            self.estimator_vector = []
            for be in batch(list(self.E.items()), n=self.e_batch_size):
                de = dict(be)
                #if self.verbose: print("Training batch of Es: %s" % de.keys())
                self.estimator_vector.append(
                    Parallel(n_jobs=-1, prefer="threads")(
                        delayed(self._fit)(e, de[e]['e'],
                                de[e]['x'], de[e]['y'])
                            for e in de)
                        )
                trained = list(de.keys())
                list(map(self.E.pop, trained))
            self.estimator_vector = sum(self.estimator_vector, [])

        self.estimator_vector = dict(self.estimator_vector)
        return self

    def _fit(self, e, estimator, X, y):
        try:
            if self.partial:
                estimator.partial_fit(X, y, classes=np.array([1, 0]))
            else:
                estimator.fit(X, y)
            return e, estimator

        except RuntimeError:
            return e, None

    def predict(self, X, y=None):
        predictors = dict.fromkeys(self.estimator_vector, [0] * X.shape[0])
        # The 'predictors' structure is different for classification than for
        # regression (for the former each prediction is a 2-dimensional vector
        # and for the second it is a real number)
        for e in self.estimator_vector:
            if not self.estimator_vector[e] is None:
                if self.ret_neg_class_probs:
                    probabilities = self.estimator_vector[e].predict_proba(X)
                    predictors[e] = probabilities[:, 1]
                    predictors[str(e) + '_neg_class'] = probabilities[:, 0]
                else:
                    predictors[e] = self.estimator_vector[e]\
                                        .predict_proba(X)[:, 1]

        return pd.DataFrame(predictors).transpose()


if __name__ == '__main__':
    input_text = "/home/iarroyof/data/pruebaWikipedia_es.txt"
    to_file = True
    pos_th = None
    min_pred_prob = 0.6
    k_topic = 20
    batch_size = 100
    n_train = 2500
    n_test  = 1000
    minf = 2
    maxf = 30

    split_vectorizer = SplitVectorizer(
            type_analyzer='word', n_gram_range=(2, 3), Xy='join',
			input_file_name=input_text, batch_size=batch_size, vectorize=True)
    split_vectorizer.fit()
    # Get training set as multilabel regression, so like 'X', 'Y' is a tfidf
    # matrix also.
    if batch_size in [None, 0, 1]:
        X, Y = split_vectorizer.get_matrices()
        X_train, X_test, Y_train, Y_test, \
        docs_X_train, docs_X_test, \
        docs_Y_train, docs_Y_test = train_test_split(X, Y,
                                                    split_vectorizer.docs_X,
                                                    split_vectorizer.docs_Y,
                                                    train_size=n_train,
                                                    test_size=n_test)
    vocab = split_vectorizer.vocab

    mvsr = MultivariateSparseSGDRegressor(estimator_names=vocab, minf=minf,
                                    partial=True, maxf=maxf, pos_th=pos_th)
    if batch_size in [None, 0, 1]:
        mvsr.fit(X_train, Y_train)
    else:
    #batch(list(self.E.items()), n=self.e_batch_size)
    #for XY_batch in batch(list(zip(X_train, Y_train)), n=batch_size):
    #    X_batch = vstack([x[0] for x in XY_batch])
    #    Y_batch = vstack([y[0] for y in XY_batch])
    #    mvsr.fit(X_batch, Y_batch)
        for XY_batch in split_vectorizer:
            mvsr.fit(XY_batch[0], XY_batch[0])

    predictions = mvsr.predict(X_test)
    # Each estimator predicts the tfidf weight of the word it represents of
    # appearing in the target text. Therefore, for each input text, the
    # estimator of w_i will predict its importance for appearing next.
    # A simple approach to show the output is to rank the vocabulary according
    # to the predicted importances in order to build a predicted topic.
    # Another, and probably subsequent, approach is to sample the vocabulary
    # according to this rank simulating a 2nd-order Markov process, taking into
    # account the most frequent words assuming that they are connection nodes
    # among the items of the rank. This, unlike the previous approach, has the
    # aim of fully generating the target text.
    predicted_topics = []
    drawn_topics = []
    for d in predictions:
        d_sorted_probs = predictions.sort_values([d], ascending=[0])
        # Verify wheter the largest probability is greater than 'min_pred_prob'
        if d_sorted_probs[d].iloc[0] < min_pred_prob:
            predicted_topics.append([None] * k_topic)
            drawn_topics.append([None] * k_topic)
        else:
            w_i = d_sorted_probs[d].index.values[:k_topic]
            predicted_topics.append([vocab[w] for w in w_i])
            raw_probs = d_sorted_probs[d][:k_topic * 2]
            raw_probs /= raw_probs.sum()
            k_rdntopics = np.random\
                         .choice([vocab[w]
                            for w in d_sorted_probs[d].index\
                                                      .values[:k_topic * 2]],
                                 size=k_topic, p=raw_probs, replace=False)
            drawn_topics.append(k_rdntopics)

    shower_df = pd.DataFrame({'docs_X': docs_X_test,
                                'docs_Y': docs_Y_test})

    pred_drw_top_df = pd.DataFrame({'pred_Topic': predicted_topics,
                                    'drawn_Topic': drawn_topics})
    print(shower_df)
    print(pred_drw_top_df)
    if to_file:
        shower_df.to_csv("input_documents.csv")
        pred_drw_top_df.to_csv("predicted_documents.csv")
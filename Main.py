import numpy as np
from sklearn.base import clone
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import roc_auc_score as AUC
from scipy.stats import ks_2samp
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, RFE, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from boruta_py import BorutaPy as Boruta
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import numpy as np
from river import datasets
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import pandas as pd
from tqdm import tqdm
import json
import random
import time
import sys

def d3(X,clf = LogisticRegression(solver='liblinear')):
    y = np.ones(X.shape[0])
    y[:int(X.shape[0]/2)] = 0
    
    predictions = np.zeros(y.shape)
    
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = probs
    auc_score = AUC(y, predictions)
    
    return 1 - auc_score


def gen_window_matrix(l1,l2, n_perm, cach=dict()):
    if (l1,l2, n_perm) not in cach.keys():
        #print("compute window: "+str((l1,l2, n_perm)))
        w = np.array(l1*[1./l1]+(l2)*[-1./(l2)])
        W = np.array([w] + [np.random.permutation(w) for _ in range(n_perm)])
        cach[(l1,l2,n_perm)] = W
    return cach[(l1,l2,n_perm)]
def mmd(X, s=None, n_perm=2500):
    K = apply_kernel(X, metric="rbf")
    if s is None:
        s = int(X.shape[0]/2)
    W = gen_window_matrix(s,K.shape[0]-s, n_perm)
    s = np.einsum('ij,ij->i', np.dot(W, K), W)
    p = (s[0] < s).sum() / n_perm
    
    return s[0], p


def ks(X, s=None):
    if s is None:
        s = int(X.shape[0]/2)
    return min([ks_2samp(X[:,i][:s], X[:,i][s:],mode="exact")[1] for i in range(X.shape[1])])

class DataLoader():
    def __init__(self):
        self.X, self.y = None,None
        self.mod, self.n = -1, -1
    def setXy(self,n,Xy):
        self.n = n
        self.mod = np.random.randint(0,n)
        
        if Xy is not None:
            self.X,self.y = Xy[0],Xy[1]
            if len(self.y.shape) == 2 and self.y.shape[1] == 1:
                self.y = self.y.ravel()
    def get_mod(self):
        return self.mod
    def generate_drift(self):
        mod_old = self.mod
        while mod_old == self.mod:
            self.mod = (np.random.randint(1,self.n)+self.mod)%self.n
        
    def next_sample(self, samp_size=1):
        sel = np.random.randint(0,int(self.X.shape[0]/self.n), samp_size) + int(self.mod * self.X.shape[0] / self.n)
        return self.X[sel],self.y[sel]


    
def load_weather(data_folder="res/data/weather/", **kwds):
        df_labels = pd.read_csv(os.path.join(data_folder, "NEweather_class.csv"))
        y = df_labels.values.flatten()

        df_data = pd.read_csv(os.path.join(data_folder, "NEweather_data.csv"))
        X = df_data.values

        return X, y.reshape(-1, 1)
class WeatherLoader(DataLoader):
    def __init__(self, **kwds):
        self.setXy(2,load_weather(**kwds))
        
def load_forest(data_folder="res/data/", **kwds):
        df = pd.read_csv(os.path.join(data_folder, "forestCoverType.csv"))
        data = df.values
        X, y = data[:, 1:-1], data[:, -1]

        return X, y.reshape(-1, 1)
class ForestLoader(DataLoader):
    def __init__(self, **kwds):
        self.setXy(2,load_forest(**kwds))
        
def load_elec(data_folder="res/data/", **kwds):
        df = pd.read_csv(os.path.join(data_folder, "elecNormNew.csv"))
        data = df.values
        X, y = data[:, 1:-1], data[:, -1]

        X = X.astype(float)
        label = ["UP", "DOWN"]
        le = LabelEncoder()
        le.fit(label)
        y = le.transform(y)

        return X, y.reshape(-1, 1)
class ElecLoader(DataLoader):
    def __init__(self, **kwds):
        self.setXy(2,load_elec(**kwds))

def load_HTTP(total_size=50000):
    data = list(datasets.HTTP().take(total_size))
    labels = ['dst_bytes', 'duration', 'src_bytes']
    X = np.array([tuple([entry[0][l] for l in labels]) for entry in data])
    y = np.array([entry[1] for entry in data]).reshape(-1,1)
    return X,y
class HTTPLoader(DataLoader):
    def __init__(self, **kwds):
        self.setXy(2,load_HTTP(**kwds))
        
def load_CreditCard(total_size=50000):
    data = list(datasets.CreditCard().take(total_size))
    labels = ['V6', 'V3', 'V19', 'V20', 'V25', 'V24', 'V11', 'V7', 'V17', 'V10', 'V27', 'V2', 'V26', 'Time', 'V23', 'Amount', 'V1', 'V8', 'V13', 'V16', 'V21', 'V14', 'V9', 'V15', 'V22', 'V18', 'V12', 'V5', 'V28', 'V4']
    X = np.array([tuple([entry[0][l] for l in labels]) for entry in data])
    y = np.array([entry[1] for entry in data]).reshape(-1,1)
    return X,y
class CreditCardLoader(DataLoader):
    def __init__(self, **kwds):
        self.setXy(2,load_CreditCard(**kwds)) 
        
class AgrawalLoader(DataLoader):
    def __init__(self):
        self.setXy(10,None)
    def next_sample(self, samp_size=1):
        data = list(datasets.synth.Agrawal(classification_function=self.get_mod()).take(samp_size))
        labels = ['age', 'car', 'loan', 'commission', 'elevel', 'hyears', 'hvalue', 'zipcode', 'salary']
        X = np.array([tuple([entry[0][l] for l in labels]) for entry in data])
        y = np.array([entry[1] for entry in data]).flatten()
        return X,y
    
class SEALoader(DataLoader):
    def __init__(self):
        self.setXy(4,None)
    def next_sample(self, samp_size=1):
        data = list(datasets.synth.SEA(variant=self.get_mod()).take(samp_size))
        labels = [0, 1, 2]
        X = np.array([tuple([entry[0][l] for l in labels]) for entry in data])
        y = np.array([entry[1] for entry in data]).flatten()
        return X,y

class MixedLoader(DataLoader):
    def __init__(self):
        self.setXy(2,None)
    def next_sample(self, samp_size=1):
        data = list(datasets.synth.Mixed(classification_function=self.get_mod()).take(samp_size))
        labels = [0, 1, 2, 3]
        X = np.array([tuple([entry[0][l] for l in labels]) for entry in data])
        y = np.array([entry[1] for entry in data]).flatten()
        return X,y
    
class TestLoader(DataLoader):
    def __init__(self):
        self.setXy(5,None)
    def next_sample(self, samp_size=1):
        X = np.ones( (samp_size,3) ) * self.get_mod()
        X[:,1:] /= 10
        X[:,2:] /= 10
        return X, np.random.random(size=samp_size)

data_loader = {"Weather": WeatherLoader, "Elec": ElecLoader, "Forest": ForestLoader, 
               "HTTP": HTTPLoader, "CreditCard": CreditCardLoader, 
               "Agrawal": AgrawalLoader, "SEA": SEALoader, "Mixed": MixedLoader,
               "Test": TestLoader}

def generate_usup_stream(loader, chunks):
    data = []
    generator = loader()

    for chunk_size in chunks:
        generator.generate_drift()
        if chunk_size > 0:
            X,y = generator.next_sample(chunk_size)
            if len(y.shape) == 1:
                y = y.reshape(-1,1)
            #print(X.shape, y.shape, loader, chunk_size)
            data.append( np.hstack( (X, y) ) )
    
    return np.vstack(tuple(data))

def generate_data(dataset, drift_pos, num_add_features, add_feature_cor, length, mul=10):
    n1 = int(length*drift_pos)
    n2 = length - n1
    
    if num_add_features == 0:
        X = generate_usup_stream(data_loader[dataset], [n1,n2])
    else:
        X_base = generate_usup_stream(data_loader[dataset], [mul*n1,mul*n2])

        X0 = X_base[::mul]
        X1 = X_base[np.random.permutation(X_base.shape[0])]

        if add_feature_cor:
            X2 = [X1[:,c].reshape(-1,1) for c in np.random.randint(0,X1.shape[1],num_add_features)]
        else:
            X2 = [np.random.permutation(X1[:,c]).reshape(-1,1) for c in np.random.randint(0,X1.shape[1],num_add_features)]
        X2 = np.hstack(X2)
        X2 = X2[np.random.choice(X2.shape[0],size=X0.shape[0],replace=False)]
        X = np.hstack( (X0, X2) )
    
    X = RobustScaler().fit_transform(X)
    
    return X, {"dataset":dataset, "before drift (size)": n1, "after drift (size)": n2, "additinal features": num_add_features, "additional features correlated": add_feature_cor}

def check_dd(dd):
    from sklearn.preprocessing import RobustScaler
    for name, loader in data_loader.items():
        res = {True: [], False: []}
        for drift in [True,False]:
            for _ in range(50):
                data = generate_usup_stream(loader, [500,500] if drift else [500+500])
                data = RobustScaler().fit_transform(data)

                res[drift].append( dd(data) )
        print(name,":",AUC(len(res[True])*[0]+len(res[False])*[1], res[True]+res[False]))

def sanity_check():
    print(" == d3 ==");check_dd(d3)
    print(" == MMD ==");check_dd(lambda X:mmd(X)[1])
    print(" == KS ==");check_dd(ks)
#sanity_check()

def sklearn_model_type(model):
    if is_classifier(model):
        return "cls"
    elif is_regressor(model):
        return "reg"
    else:
        raise ValueError()
        
def create_target(model, size, degree=5):
    if model == "cls":
        n = int(size/2)
        return np.array(n*[0]+(size-n)*[1], dtype=int)
    elif model == "reg":
        t = np.linspace(0,2*np.pi, size)
        y = np.array([t]+[np.sin(d*t+o) for d in range(1,degree+1) for o in [0,np.pi/2]]).T
        if y.shape[1] == 1:
            return y.ravel()
        else:
            return y
    else:
        raise ValueError()

def select_best_k(scores, k):
    return np.argsort(scores)[-k:]

def run_dd(X, desc):
    if X.flatten().shape[0] == 0:
        X = None
    elif len(X.shape) == 1:
        X = X.reshape(-1,1)
    
    desc = list(desc.items())
    res = []
    for name,dd in {"MMD":lambda x:mmd(x)[1], "D3 (lin)": d3, "D3 (kNN)": lambda x:d3(x,KNeighborsClassifier()), "D3 (DT)": lambda x:d3(x,DecisionTreeClassifier()), "KS":ks}.items():
        val = dd(X) if X is not None else np.nan
        res.append(dict(desc + list({"drift detector":name, "drift score": val}.items())))
    return res

def filter_setup(name=None,tpe=None,model=None,select=None, filter_off=True):
    if filter_off:
        return True
    pattern = {"name":name, "type":tpe, "model": model, "select": select}
    
    allowed = [
        {"name":"all", "type":"usup", "model": None, "select": -1},
        {"name":"PCA", "type":"usup", "model": None, "select": 15},
        {"name":"GRP", "type":"usup", "model": None, "select": 15},
        {"name":"URP", "type":"usup", "model": None, "select": 10},
        {"name":"SRP", "type":"usup", "model": None, "select": 15},
        {"name":"MI", "type":"reg", "model": None, "select": 10},
        {"name":"RFE", "type":"cls", "model":"ET", "select": 10},
        {"name":"Boruta", "type":"reg", "model":"ET", "select": "strong"},
        {"name":"PFI", "type":"reg", "model":"ET", "select": 3},
        {"name":"FI", "type":"cls", "model":"ET", "select": 5}
    ]
    
    for k,v in pattern.items():
        if v is not None:
            v = str(v)
            allowed = [a for a in allowed if str(a[k]) == v]
    return len(allowed) > 0

def run_experiment(setup, filter_off=True):
    X, setup = generate_data(**setup)
    size = X.shape[0]
    selections = [1,2,3,5,10,15]; selections.sort(); selections.reverse()
    
    results = []
    
    results += run_dd(X,{"method":{"name":"all","select":-1,"type":"usup"}, "time":0})
    
    for method_name,method,tpe in [("MI",mutual_info_regression,"reg"), ("MI",mutual_info_classif, "cls")]:
        t0 = time.time()
        if filter_setup(method_name,tpe,None,None,filter_off=filter_off):
            scores = method(X, create_target(tpe, size, degree=0))
            tm = time.time() - t0
            for n_sel in selections:
                if filter_setup(method_name,tpe,None,n_sel,filter_off=filter_off):
                    sel = select_best_k(scores, n_sel)
                    results += run_dd(X[:,sel] ,{"method": {"name": method_name, "select": n_sel, "type": tpe}, "time": tm, "features": n_sel})
    
    for model_name, model in [("ET",ExtraTreesClassifier()), ("RF",RandomForestClassifier()), ("kNN",KNeighborsClassifier()), 
                              ("DT",DecisionTreeClassifier()), ("Lin",LogisticRegression()), ("Ridge",KernelRidge(kernel="rbf")),
                              ("ET",ExtraTreesRegressor()), ("RF",RandomForestRegressor()), ("kNN",KNeighborsRegressor()), 
                             ]:
        model = clone(model)
        tpe = sklearn_model_type(model)
        y = create_target(tpe, size)
        t0 = time.time()
        if filter_setup(None,tpe,model_name,None,filter_off=filter_off):
            model.fit( X, y )
            tm = time.time()-t0
            for method_name,method in  [("PFI",lambda mdl_X_y:permutation_importance(mdl_X_y[0], mdl_X_y[1],mdl_X_y[2], n_repeats=10).importances_mean)]+([] if model_name not in ["ET","DT","RF"] else [("FI",lambda mdl_X_y:mdl_X_y[0].feature_importances_)]):
                scores = method( (model, X, y) )
                for n_sel in selections:
                    if filter_setup(method_name,tpe,model_name,n_sel,filter_off=filter_off):
                        sel = select_best_k(scores, n_sel)
                        results += run_dd(X[:,sel],{"method": {"name": method_name, "model": model_name, "select": n_sel, "type": tpe}, "time": tm, "features": sel})
    
    for model_name, model in [("ET",ExtraTreesClassifier(min_samples_leaf=8)), ("RF",RandomForestClassifier(min_samples_leaf=8)), ("ET",ExtraTreesRegressor(min_samples_leaf=8)), ("RF",RandomForestRegressor(min_samples_leaf=8))]:
        model = clone(model)
        tpe = sklearn_model_type(model)
        y = create_target(tpe, size)
        model = Boruta(model, n_estimators="auto", early_stopping=True)
        t0 = time.time()
        if filter_setup("Boruta",tpe,model_name,None,filter_off=filter_off):
            model.fit(X,y)
            tm = time.time()-t0
            for n_sel in selections:
                if filter_setup("Boruta",tpe,model_name,n_sel,filter_off=filter_off):
                    sel = select_best_k(model.stat, n_sel)
                    results += run_dd(X[:,sel],{"method": {"name": "Boruta", "model": model_name, "select": n_sel, "type": tpe}, "time": tm, "features": sel})
            sel = np.where(model.support_)[0]
            if filter_setup("Boruta",tpe,model_name,"strong",filter_off=filter_off):
                results += run_dd(X[:,sel],{"method": {"name": "Boruta", "model": model_name, "select": "strong", "type": tpe}, "time": tm, "features": sel})

            sel = np.where(np.logical_or(model.support_weak_,model.support_))[0]
            if filter_setup("Boruta",tpe,model_name,"relevant",filter_off=filter_off):
                results += run_dd(X[:,sel],{"method": {"name": "Boruta", "model": model_name, "select": "relevant", "type": tpe}, "time": tm, "features": sel})
    
    for model_name, model in [("ET",ExtraTreesClassifier(min_samples_leaf=8)), ("RF",RandomForestClassifier(min_samples_leaf=8)),
                              ("Lin",LogisticRegression()),
                              ("ET",ExtraTreesRegressor(min_samples_leaf=8)), ("RF",RandomForestRegressor(min_samples_leaf=8)), 
                             ]:
        model = clone(model)
        tpe = sklearn_model_type(model)
        y = create_target(tpe, size)
        if filter_setup("RFE",tpe,model_name,None,filter_off=filter_off):
            for method_name,selector in [("RFE", lambda x,n_sel: RFE(model,n_features_to_select=n_sel).fit(x,y)),
                                           #("SFE (b)", lambda x,n_sel: SequentialFeatureSelector(model,n_features_to_select=n_sel,direction="backward").fit(x,y)),
                                           #("SFE (f)", lambda x,n_sel: SequentialFeatureSelector(model,n_features_to_select=n_sel,direction="forward").fit(x,y))
                                        ]:
                X_ = X.copy()
                t0 = time.time()
                for n_sel in [s for s in selections if filter_setup(method_name,tpe,model_name,s,filter_off=filter_off)]:
                    if method_name == "SFE (f)":
                        X_ = X.copy()
                        t0 = time.time()
                    if n_sel < X_.shape[1]:
                        sel = np.where(selector( X_, n_sel ).support_)[0]
                    else:
                        sel = list(range(X.shape[1]))
                    t1 = time.time()
                    X_ = X_[:,sel]
                    if filter_setup(method_name,tpe,model_name,n_sel,filter_off=filter_off):
                        results += run_dd(X_,{"method": {"name": method_name, "model": model_name, "select": n_sel, "type": tpe}, "time": t1-t0})
    
    
    for n_sel in selections:
        t0 = time.time()
        X_proj = PCA(n_components=min(n_sel,X.shape[1])).fit_transform(X)
        tm = time.time() - t0
        if filter_setup("PCA","usup",None,n_sel,filter_off=filter_off):
            results += run_dd(X_proj,{"method": {"name": "PCA", "select": n_sel, "type": "usup"}, "time": tm})
        
        P = np.random.normal(size=(n_sel,X.shape[1])).T
        tm,X_proj = 0, X @ P
        if filter_setup("GRP","usup",None,n_sel,filter_off=filter_off):
            results += run_dd(X_proj,{"method": {"name": "GRP", "select": n_sel, "type": "usup"}, "time": tm})
        
        P = np.random.random(size=(n_sel,X.shape[1])).T
        tm,X_proj = 0, X @ P
        if filter_setup("URP","usup",None,n_sel,filter_off=filter_off):
            results += run_dd(X_proj,{"method": {"name": "URP", "select": n_sel, "type": "usup"}, "time": tm})
        
        P = np.random.random(size=(n_sel,X.shape[1])).T; P = 1.*(P < 1/X.shape[1])+1e-32; P /= P.sum(axis=0)[None,:]
        tm,X_proj = 0, X @ P
        if filter_setup("SRP","usup",None,n_sel,filter_off=filter_off):
            results += run_dd(X_proj,{"method": {"name": "SRP", "select": n_sel, "type": "usup"}, "time": tm})
    return [dict(list(entry.items())+list(setup.items())) for entry in results]        

if len(sys.argv) not in [2,3] or sys.argv[1] not in ["--make","--setup","--run_exp"]:
    print("Use --make | --setup #n | --run_exp #i")
    exit(1)
if sys.argv[1] == "--make":
    if len(sys.argv) != 2:
        print("WARN: --make does not take argument")
    pass
    
elif len(sys.argv) != 3:
    print("Use --make | --setup #n | --run_exp #i")
    exit(1)
if sys.argv[1] == "--setup":
    try:
        n_setups = int(sys.argv[2])
        assert str(n_setups) == sys.argv[2]
        assert n_setups >= 1
    except:
        print("number must be an integer >= 1")
        exit(1)
    
    setups = [{"dataset": dataset, "drift_pos": drift_pos, "num_add_features": num_add_features, "add_feature_cor": add_feature_cor, "length": length} 
              for dataset in data_loader.keys() 
              for drift_pos in [0,0.1,0.25,0.5] 
              for num_add_features in [0,1,2,3,5,10,15,25,35,50,75,100,150,200,250]
              for add_feature_cor in [True, False] 
              for length in [100,250]]
    random.shuffle(setups)
    setups *= 200
    
    n = int(len(setups)/n_setups)
    
    with open("setups.json", "w") as file:
        json.dump({i:setups[i*n:(i+1)*n] for i in range(n_setups)}, file)
    
elif sys.argv[1] == "--run_exp" or sys.argv[1] == "--run_exp_best":
    filter_off = not(sys.argv[1] == "--run_exp_best")

    try:
        setup_id = int(sys.argv[2])
        assert str(setup_id) == sys.argv[2]
    except:
        print("id must be an integer >= 1")
        exit(1)
    
    with open("setups.json", "r") as file:
        setups = json.load(file)[str(setup_id)]
    print("running setup %i (%s)"%(setup_id,str(setups)))
    
    results = []
    for res in tqdm(map(lambda setup: run_experiment(setup,filter_off), setups), total=len(setups)):
        results += res
    
    pd.DataFrame(results).to_pickle("./out/result_%i.pkl.xz"%setup_id)

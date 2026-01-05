import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from timeview.data import BaseDataset
from abc import ABC, abstractmethod
import json
import os
import numpy as np
import pandas as pd
import glob
from scipy.stats import beta
from timeview.knot_selection import *
from experiments.dynamic_modeling import *
from timeview.basis import BSplineBasis
from typing import Optional
from dataclasses import dataclass
from scipy.special import betaln


def save_dataset(dataset_name, dataset_builder, dataset_dictionary, notes="", dataset_description_path="dataset_descriptions"):
    # Check if a dataset description directory exists. If not, create it.
    if not os.path.exists(dataset_description_path):
        os.makedirs(dataset_description_path)
    
    # Check if a dataset description file already exists. If so, raise an error.
    path = os.path.join(dataset_description_path, dataset_name + ".json")
    if os.path.exists(path):
        raise ValueError(f"A dataset description file with this name already exists at {path}.")

    dataset_description = {
        'dataset_name': dataset_name,
        'dataset_builder': dataset_builder,
        'dataset_dictionary': dataset_dictionary,
        'notes': notes
    }
    with open(path, 'w') as f:
        json.dump(dataset_description, f, indent=4)

def load_dataset_description(dataset_name, dataset_description_path="dataset_descriptions"):
    path = os.path.join(dataset_description_path, dataset_name + ".json")
    # Check if a dataset description file exists. If not, raise an error.
    if not os.path.exists(path):
        raise ValueError(f"A dataset description file with this name does not exist at {path}.")

    with open(path, 'r') as f:
        dataset_description = json.load(f)
    return dataset_description


def load_dataset(dataset_name, dataset_description_path="dataset_descriptions", data_folder=None):
    dataset_description = load_dataset_description(dataset_name, dataset_description_path=dataset_description_path)
    dataset_builder = dataset_description['dataset_builder']
    dataset_dictionary = dataset_description['dataset_dictionary']
    if data_folder is not None:
        dataset_dictionary['data_folder'] = data_folder
    dataset = get_class_by_name(dataset_builder)(**dataset_dictionary)
    return dataset


def get_class_by_name(class_name):
    """
    This function takes a class name as an argument and returns a python class with this name that is implemented in this module.
    """
    return globals()[class_name]

class SimpleLinearDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        n_points = 10
        n_samples = 100
        self.X = pd.DataFrame({'x':np.linspace(0,1,n_samples)})
        coeffs = np.random.uniform(-1, 1, size=n_points)
        coeffs[0] = 1
        self.ts = [np.linspace(0,1,n_points) for i in range(n_samples)]
        y0s = [np.random.uniform(-1, 1) for i in range(n_samples)]
        self.ys = [coeffs*y0 for y0 in y0s]

    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['x']
    
    def get_feature_ranges(self):
        return {
            'x': (0, 1)
        } 

class ExponentialDataset(BaseDataset):

    def __init__(self, log_t=False):
        super().__init__(log_t=log_t)
        self.X = pd.DataFrame({'x':np.linspace(0,1,100)})
        self.ts = [np.linspace(0,1,20) for i in range(100)]
        self.ys = [np.exp(t*x) for t, x in zip(self.ts, self.X['x'])]
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['x']
    
    def get_feature_ranges(self):
        return {
            'x': (0, 1)
        }
    
class BetaDataset(BaseDataset):

    def __init__(self, n_samples=100, n_timesteps=20):
        super().__init__(n_samples=n_samples, n_timesteps=n_timesteps)
        n_samples_per_dim = int(np.sqrt(n_samples))
        n_samples = n_samples_per_dim**2
        alphas = np.linspace(1.0,4.0,n_samples_per_dim)
        betas = np.linspace(1.0,4.0,n_samples_per_dim)

        grid = np.meshgrid(alphas, betas)
    
        # stack along the last axis and then reshape into 2 columns
        cart_prod = np.stack(grid, axis=-1).reshape(-1, 2)

        self.X = pd.DataFrame({'alpha':cart_prod[:,0], 'beta':cart_prod[:,1]})
        self.ts = [np.linspace(0,1,n_timesteps) for i in range(len(self.X))]
        self.ys = [np.array([beta.pdf(t,alpha, betap) for t in np.linspace(0,1,n_timesteps)]) for alpha, betap in zip(self.X['alpha'], self.X['beta'])]
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['alpha', 'beta']
    
    def get_feature_ranges(self):
        return {
            'alpha': (1.0, 4.0),
            'beta': (1.0, 4.0)
        }
    
class Exponential2Dataset(BaseDataset):

    def __init__(self, n_samples=100, n_timesteps=20):
        super().__init__()
        self.X = pd.DataFrame({'x':np.linspace(-1,1,n_samples)})
        self.ts = [np.linspace(0,1,n_timesteps) for i in range(n_samples)]
        self.ys = [np.exp((t-1)*x) for t, x in zip(self.ts, self.X['x'])]
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['x']
    
    def get_feature_ranges(self):
        return {
            'x': (-1, 1)
        }

class SineDataset(BaseDataset):

    def __init__(self, n_samples=100, n_timesteps=20):
        super().__init__(n_samples=n_samples, n_timesteps=n_timesteps)
        self.X = pd.DataFrame({'x':np.linspace(-1,1,n_samples)})
        self.ts = [np.linspace(0,1,n_timesteps) for i in range(n_samples)]
        self.ys = [np.sin(t*x*np.pi) for t, x in zip(self.ts, self.X['x'])]
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['x']
    
    def get_feature_ranges(self):
        return {
            'x': (-1, 1)
        }
    
class SineTransDataset(BaseDataset):

    def __init__(self, n_samples=100, n_timesteps=20):
        super().__init__(n_samples=n_samples, n_timesteps=n_timesteps)
        self.X = pd.DataFrame({'x':np.linspace(1.0,3.0,n_samples)})
        self.ts = [np.linspace(0,1,n_timesteps) for i in range(n_samples)]
        self.ys = [np.sin(2*t*np.pi/x) for t, x in zip(self.ts, self.X['x'])]
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['x']
    
    def get_feature_ranges(self):
        return {
            'x': (1, 2.5)
        }

class AirfoilDataset(BaseDataset):

    def __init__(self, log_t=False):
        super().__init__(log_t=log_t)
        df = pd.read_csv('data/airfoil/airfoil_self_noise.dat', sep='\t', header=None)
        df.columns = ['t', 'angle', 'chord', 'velocity', 'thickness', 'y']

        # We need to assign ids
        df['id'] = 0
        prev = 10000000
        curr_id = 0
        for index, row in df.iterrows():
            if row['t'] < prev:
                curr_id += 1
            prev = row['t']
            df.at[index, 'id'] = curr_id
        
        df = df[['id', 'angle', 'chord', 'velocity', 'thickness', 't', 'y']]

        df['t'] = df['t'] / 200 # scale to min 1

        if log_t:
            df['t'] = np.log(df['t']) # now it will be in range 0-4.7
        else:
            df['t'] = df['t'] / 100 # scale to range 0-1

        self.X, self.ts, self.ys = self._extract_data_from_one_dataframe(df)
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['angle', 'chord', 'velocity', 'thickness']
    
    def get_feature_ranges(self):
        return {
            'angle': (0, 22),
            'chord': (0.025, 0.30),
            'velocity': (31, 71),
            'thickness': (0.0004, 0.05)
        }


class CelgeneDataset(BaseDataset):

    def __init__(self):
        super().__init__()
        df = pd.read_csv(os.path.join("data", "celgene", "celgene.csv"))
        df = df[df['t'] <= 365]
        df['t'] = df['t'] / 365 # scale

        df['y'] = df['y'] / 100 # scale
        # Filter out patients with fewer than 3 observations
        df = df.groupby('id').filter(lambda x: len(x) > 4)
        # Filter out duplicate observation times (keep the first one)
        df = df.groupby(['id', 't']).first().reset_index()
        df = df[['id'] + self.get_feature_names() + ['t','y']]


        self.X, self.ts, self.ys = self._extract_data_from_one_dataframe(df)
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['age', 'race', 'ECOG', 'location', 'bmi', 'sysbp', 'diabp']
    
    def get_feature_ranges(self):
        return {
            'age': (50, 85),
            'race': ['White', 'Other', 'Black or African American'],
            'ECOG': [0, 1, 2],
            'location': ['lymph nodes', 'organ or soft tissue'],
            'bmi': (18, 50),
            'sysbp': (85, 180),
            'diabp': (50, 115),
        }

class FLChainDataset(BaseDataset):

    def __init__(self, subset='all'):
        super().__init__(subset=subset)
        df = pd.read_csv(os.path.join("data", "flchain", "flchain.csv"))
        df = df[['id'] + self.get_feature_names() + ['t','y']]
        df['t'] = df['t'] / 5000 # scale
        if subset != 'all':
            all_ids = df['id'].unique()
            # Randomly select a subset of patients
            gen = np.random.default_rng(0)
            ids = gen.choice(all_ids, size=subset, replace=False)
            df = df[df['id'].isin(ids)]

        self.X, self.ts, self.ys = self._extract_data_from_one_dataframe(df)
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['age', 'sex', 'creatinine', 'kappa', 'lambda', 'flc.grp', 'mgus']
    
    def get_feature_ranges(self):
        return {
            'age': (50, 100),
            'sex': ['M', 'F'],
            'creatinine': (0.4, 2.0),
            'kappa': (0.01, 5.0),
            'lambda': (0.04, 5.0),
            'flc.grp': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'mgus': ['no', 'yes']
        }


class StressStrainDataset(BaseDataset):

    def __init__(self, lot='all', include_lot_as_feature=False, more_samples=0, downsample=True, specimen='all', max_strain=0.3):
        super().__init__(lot=lot, include_lot_as_feature=include_lot_as_feature, more_samples=more_samples, downsample=downsample, specimen=specimen, max_strain=max_strain)
        if lot == 'all':
            path = os.path.join("data", "stress-strain-curves", "T*.csv")
        else:
            path = os.path.join("data", "stress-strain-curves", f"T*{lot}*.csv")
        filenames = glob.glob(path)

        dfs = []
        for filename in filenames:
            df = pd.read_csv(filename)
            dataset_name = filename.split('T_')[1].split('.csv')[0]
            parts = dataset_name.split('_')
            temp = parts[0]
            lot = parts[1]
            specimen = parts[2]
            if (self.args.specimen != 'all') and (int(specimen) != self.args.specimen):
                continue
            df.columns = ['t', 'y']
            df.drop(df.tail(1).index,inplace=True) # drop last row because it's an outlier
            if downsample:
                df = df.iloc[::3,:] # downsample
            df['temp'] = float(temp)
            if include_lot_as_feature:
                df['lot'] = lot
            df['id'] = lot + '_' + specimen + '_' + temp
            df.reset_index(inplace=True)
            if more_samples > 0:
                for ind in range(more_samples):
                    df.loc[ind::more_samples, 'id'] = df.loc[ind::more_samples, 'id'] + '_' + str(ind)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        if include_lot_as_feature:
            df = df[['id', 'temp', 'lot', 't', 'y']]
        else:
            df = df[['id', 'temp', 't', 'y']]
        df.drop(index=df[df['t'] < 0].index, inplace=True) # drop rows where t is < 0
        df.drop(index=df[df['y'] < 0].index, inplace=True) # drop rows where y is < 0
        df.drop(index=df[df['t'] > max_strain].index, inplace=True) # drop rows where t is > max_strain
        df['y'] = df['y'] / 300 # scale

        self.X, self.ts, self.ys = self._extract_data_from_one_dataframe(df)

    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)

    def get_feature_names(self):
        if self.args.include_lot_as_feature:
            return ['temp', 'lot']
        else:
            return ['temp']
    
    def get_feature_ranges(self):
        if self.args.include_lot_as_feature:
            return {'temp': (20, 300), 'lot': ['A','B','C','D','E','F','G','H', 'I']}
        else:
            return {'temp': (20, 300)}


class TacrolimusDataset(BaseDataset):

    def __init__(self, granularity, normalize=False, max_t=25, data_folder='data'):
        super().__init__(granularity=granularity, normalize=normalize)
        if granularity == 'visit':

            df = pd.read_csv(os.path.join(data_folder, "tacrolimus", "tac_pccp_mr4_250423.csv"))
            dosage_rows = df[df['DOSE'] != 0]
            assert dosage_rows['visit_id'].is_unique
            df.drop(columns=['DOSE', 'EVID','II', 'AGE'], inplace=True) # we drop age because many missing values. the other columns are not needed
            df.drop(index=dosage_rows.index, inplace=True) # drop dosage rows
            # Merge df with dosage rows on visit_id
            df = df.merge(dosage_rows[['visit_id', 'DOSE']], on='visit_id', how='left') # add dosage as a feature
            df.loc[df['TIME'] >= 168, 'TIME'] -= 168 # subtract 168 from time to get time since last dosage
            missing_24h = df[(df['TIME'] == 0) & (df['DV'] == 0)].index
            df.drop(index=missing_24h, inplace=True) # drop rows where DV is 0 and time is 0 - they correspond to missing 24h measurements

            dv_0 = df[df['TIME'] == 0][['visit_id', 'DV']]
            assert dv_0['visit_id'].is_unique
            df = df.merge(dv_0, on='visit_id', how='left', suffixes=('', '_0')) # add DV_0 as a feature

            more_than_t = df[df['TIME'] > max_t].index
            df.drop(index=more_than_t, inplace=True) # drop rows where time is greater than max_t

            df.dropna(inplace=True) # drop rows with missing values

            df = df[['visit_id'] + ['DOSE', 'DV_0', 'SEXE', 'POIDS', 'HT', 'HB', 'CREAT', 'CYP', 'FORMULATION'] + ['TIME', 'DV']]

            df.columns = ['id'] + self.get_feature_names() + ['t', 'y']

            X, ts, ys = self._extract_data_from_one_dataframe(df)

            if normalize:
                # Make each column of X between 0 and 1
                X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            
            self.X, self.ts, self.ys = X, ts, ys
        else:
            raise NotImplementedError("Only visit granularity is implemented for this dataset.")
        

    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['DOSE', 'DV_0', 'SEX', 'WEIGHT', 'HT', 'HB', 'CREAT', 'CYP', 'FORM']
    
    def get_feature_ranges(self):
        if self.args.normalize:
            return {
                'DOSE': (0, 1),
                'DV_0': (0, 1),
                'SEX': [0,1],
                'WEIGHT': (0, 1),
                'HT': (0, 1),
                'HB': (0, 1),
                'CREAT': (0, 1),
                'CYP': [0, 1],
                'FORM': [0, 1]
            }
        else:
            return {
                'DOSE': (0, 10),
                'DV_0': (0, 20),
                'SEX': [0, 1],
                'WEIGHT': (45, 110),
                'HT': (20, 47),
                'HB': (6, 16),
                'CREAT': (60, 830),
                'CYP': [0, 1],
                'FORM': [0, 1]
            }





class WindDataset(BaseDataset):

    def __init__(self, company, granularity='daily', rolling=False):
        super().__init__(company=company, granularity=granularity, rolling=rolling)

        files = {
            "50Hertz": "50Hertz.csv",
            "Amprion": "Amprion.csv",
            "TenneTTSO": "TenneTTSO.csv",
            "TransnetBW": "TransnetBW.csv"
        }

        def load_company(name):
            file_path = os.path.join("data", "wind_data", files[name])
            df = pd.read_csv(file_path,index_col=0,parse_dates=True,dayfirst=True)

            if self.args.rolling == True:
                df = df.rolling(window=7).mean()
            elif type(self.args.rolling) == int:
                df = df.rolling(window=self.args.rolling, center=True).mean()

            df = df.loc['2019-09-01':'2020-08-31',:].copy()
            df['id'] = [f"{name}{id}" for id in list(range(len(df)))]
            df['day_number'] = list(range(len(df)))
            df['month'] = df.index.month
            df = df.melt(id_vars=['id','day_number','month'],var_name='time', value_name='y')
            df['t'] = pd.to_timedelta(df['time']).dt.total_seconds() / 3600
            df.drop(columns=['time'],inplace=True)

            # Standardize the data between 0 and 1
            # df['y'] = (df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())
            # Normalize the data so that the mean is 0 and the standard deviation is 1
            df['y'] = (df['y'] - df['y'].mean()) / df['y'].std()
            df['day_number'] = df['day_number'] / 365

            return df


        if company == 'all':
            df = pd.concat([load_company(name) for name in files.keys()])
        else:
            df = load_company(company)

        if self.args.granularity == 'daily':
            df = df[['id','day_number','t','y']]
        elif self.args.granularity == 'monthly':
            df = df[['id','month','t','y']]

        X, ts, ys = self._extract_data_from_one_dataframe(df)
        self.X, self.ts, self.ys = X, ts, ys

    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        if self.args.granularity == 'daily':
            return ['day_number']
        elif self.args.granularity == 'monthly':
            return ['month']

    def get_feature_ranges(self):
        if self.args.granularity == 'daily':
            return {
                'day_number': (0, 1)
            }
        elif self.args.granularity == 'monthly':
            return {
                'month': (1, 12)
            }

class MIMICDataset(BaseDataset):

    def __init__(self, subset=0.1, seed=0):
        super().__init__(subset=subset,seed=seed)
        df = pd.read_csv(os.path.join("data", "mimic", "processed_sepsis3_tts.csv"))

        selected_cols = [
                'traj',
                'o:gender', 'o:mechvent', 'o:re_admission', 'o:age',
                'o:Weight_kg', 'o:GCS', 'o:HR', 'o:SysBP', 'o:MeanBP', 'o:DiaBP',
                'o:RR', 'o:Temp_C', 'o:FiO2_1', 'o:Potassium', 'o:Sodium', 'o:Chloride',
                'o:Glucose', 'o:Magnesium', 'o:Calcium', 'o:Hb', 'o:WBC_count',
                'o:Platelets_count', 'o:PTT', 'o:PT', 'o:Arterial_pH', 'o:paO2',
                'o:paCO2', 'o:Arterial_BE', 'o:HCO3', 'o:Arterial_lactate', 'o:SOFA',
                'o:SIRS', 'o:Shock_Index', 'o:PaO2_FiO2', 'o:cumulated_balance',
                'o:SpO2', 'o:BUN', 'o:Creatinine', 'o:SGOT', 'o:SGPT', 'o:Total_bili',
                'o:INR', 'a:action',
                'step', 'true_score'] 

        df = df[selected_cols]

        df.columns = ['id'] + df.columns[1:-2].tolist() + ['t','y']

        # Filter out patients with less than 5 observations
        df = df.groupby('id').filter(lambda x: len(x) > 4)

        X, ts, ys = self._extract_data_from_one_dataframe(df)

        subset = self.args.subset
        seed = self.args.seed

        n = len(X)

        gen = np.random.default_rng(seed)
        subset_indices = gen.choice(n, int(n*subset), replace=False)
        subset_indices = [i.item() for i in subset_indices]

        X = X.iloc[subset_indices, :]
        ts = [ts[i] for i in subset_indices]
        ys = [ys[i] for i in subset_indices]

        self.X = X
        self.ts = ts
        self.ys = ys

    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys

    def __len__(self):
        return len(self.X)
    
    def get_feature_ranges(self):
        return {
            'o:gender': (0,1),
            'o:mechvent': (0,1), 
            'o:re_admission': (0,1), 
            'o:age': (0,1),
            'o:Weight_kg': (0,1), 
            'o:GCS': (0,1), 
            'o:HR': (0,1), 
            'o:SysBP': (0,1), 
            'o:MeanBP': (0,1), 
            'o:DiaBP': (0,1),
            'o:RR': (0,1), 
            'o:Temp_C': (0,1), 
            'o:FiO2_1': (0,1), 
            'o:Potassium': (0,1), 
            'o:Sodium': (0,1), 
            'o:Chloride': (0,1),
            'o:Glucose': (0,1), 
            'o:Magnesium': (0,1), 
            'o:Calcium': (0,1), 
            'o:Hb': (0,1), 
            'o:WBC_count': (0,1),
            'o:Platelets_count': (0,1), 
            'o:PTT': (0,1), 
            'o:PT': (0,1), 
            'o:Arterial_pH': (0,1), 
            'o:paO2': (0,1),
            'o:paCO2': (0,1), 
            'o:Arterial_BE': (0,1), 
            'o:HCO3': (0,1), 
            'o:Arterial_lactate': (0,1), 
            'o:SOFA': (0,1),
            'o:SIRS': (0,1), 
            'o:Shock_Index': (0,1), 
            'o:PaO2_FiO2': (0,1), 
            'o:cumulated_balance': (0,1),
            'o:SpO2': (0,1), 
            'o:BUN': (0,1), 
            'o:Creatinine': (0,1), 
            'o:SGOT': (0,1), 
            'o:SGPT': (0,1), 
            'o:Total_bili': (0,1),
            'o:INR': (0,1), 
            'a:action': (0,1),
        }

    def get_feature_names(self):
        return [
            'o:gender',
            'o:mechvent', 
            'o:re_admission', 
            'o:age',
            'o:Weight_kg', 
            'o:GCS', 
            'o:HR', 
            'o:SysBP', 
            'o:MeanBP', 
            'o:DiaBP',
            'o:RR', 
            'o:Temp_C', 
            'o:FiO2_1', 
            'o:Potassium', 
            'o:Sodium', 
            'o:Chloride',
            'o:Glucose', 
            'o:Magnesium', 
            'o:Calcium', 
            'o:Hb', 
            'o:WBC_count',
            'o:Platelets_count', 
            'o:PTT', 
            'o:PT', 
            'o:Arterial_pH', 
            'o:paO2',
            'o:paCO2', 
            'o:Arterial_BE', 
            'o:HCO3', 
            'o:Arterial_lactate', 
            'o:SOFA',
            'o:SIRS', 
            'o:Shock_Index', 
            'o:PaO2_FiO2', 
            'o:cumulated_balance',
            'o:SpO2', 
            'o:BUN', 
            'o:Creatinine', 
            'o:SGOT', 
            'o:SGPT', 
            'o:Total_bili',
            'o:INR', 
            'a:action'
        ]




class TumorDataset(BaseDataset):

    FILE_LIST = [
        "input celgene09.csv",
        "input centoco06.csv",
        "input cougar06.csv",
        "input novacea06.csv",
        "input pfizer08.csv",
        "input sanfi00.csv",
        "input sanofi79.csv",
        "inputS83OFF.csv",
        "inputS83ON.csv",
    ]

    def __init__(self, **args):
        super().__init__(**args)
        df_list = list()
        for f in TumorDataset.FILE_LIST:
            df = pd.read_csv(os.path.join("data",'tumor',f))
            df["name"] = df["name"].astype(str) + f
            df_list.append(df)

        df = pd.concat(df_list)

        # Filter out duplicate observation times (keep the first one)
        df = df.groupby(['name', 'date']).first().reset_index()

        # Take the log transform of the tumor volume
        def protected_log(x):
            return np.log(x + 1e-6)

        df['size'] = protected_log(df['size'])

        first_time = df.groupby('name')[['date']].min()
        first_measurements = pd.merge(first_time, df[['name','date','size']], on=['name', 'date'])
        df = pd.merge(df, first_measurements, on='name', suffixes=('', '_first'))

        df['t'] = df['date'] - df['date_first']

        # Filter only to date 365 (1 year)
        df = df[df['t'] <= 365.0]

        # Filter only to patients with at least 10 time steps
        df = df.groupby('name').filter(lambda x: len(x) >= 10)

        df['t'] = df['t'] / 365.0


        df = df[['name','size_first','t','size']]
        df.columns = ['id','y0','t','y']

        self.X, self.ts, self.ys = self._extract_data_from_one_dataframe(df)


    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys

    def __len__(self):
        return len(self.X)
    
    def get_feature_ranges(self):
        return {
            'y0': (-3, 8),
        }
    
    def get_feature_names(self):
        return ['y0']



# ============================================================
# Helper functions for Van der Schaar recruitment task
# ============================================================

@dataclass
class SemanticConstraintConfig:
    """
    Controls probabilistic enforcement of semantic constraints
    on spline intervals.

    Each interval is independently sampled to be:
      - constant
      - linear
      - unconstrained

    A global cap avoids over-constraining a single spline.
    """
    p_force_linear: float = 0.25
    p_force_constant: float = 0.10
    max_forced_intervals: int = 3


def _project_to_linear_constraints(c0: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Orthogonal projection of coefficient vector c0 onto the affine subspace:

        { c : A c = b }

    Solves:
        argmin ||c - c0||₂  subject to  A c = b

    Closed-form solution:
        c = c0 - Aᵀ (A Aᵀ)⁻¹ (A c0 - b)

    Uses least-squares for numerical stability.
    """
    if A.size == 0:
        return c0

    rhs = (A @ c0) - b
    M = A @ A.T
    lam, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    c = c0 - (A.T @ lam)
    return c


def sample_inputs_from_bsplinebasis(
    n_samples: int,
    t: np.ndarray,
    n_basis: int = 8,
    coeff_std: float = 1.0,
    seed: int = 0,
    random_internal_knots: bool = True,
    knot_jitter: float = 0.15,
    semantic_cfg: Optional[SemanticConstraintConfig] = None,
):
    """
    Samples spline-valued input trajectories u(t) by drawing
    B-spline coefficients and optionally enforcing semantic
    (constant / linear) constraints on selected intervals.

    Returns:
      u:      (n_samples, T) spline evaluations
      coeffs: (n_samples, n_basis) spline coefficients
      basis:  BSplineBasis instance used
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)

    # Number of internal knots required by cubic B-splines
    n_internal = n_basis - 3 + 1

    if random_internal_knots:
        # Jittered knot placement around uniform grid
        base = np.linspace(0.0, 1.0, n_internal)
        jitter = rng.normal(0.0, knot_jitter, size=n_internal)
        internal = np.clip(base + jitter, 0.0, 1.0)
        internal.sort()

        # Explicitly pin endpoints for numerical robustness
        internal[0] = 0.0
        internal[-1] = 1.0
    else:
        internal = np.linspace(0.0, 1.0, n_internal)

    basis = BSplineBasis(n_basis=n_basis, t_range=(0.0, 1.0), internal_knots=internal)
    B = basis.get_matrix(t)  # (T, n_basis)

    # Tensor mapping spline coefficients to per-interval monomial coefficients
    P = basis.monomial_tensor()  # (4, n_intervals, n_basis)
    n_intervals = P.shape[1]

    if semantic_cfg is None:
        semantic_cfg = SemanticConstraintConfig()

    coeffs = np.zeros((n_samples, n_basis), dtype=float)
    u = np.zeros((n_samples, len(t)), dtype=float)

    for i in range(n_samples):
        c0 = rng.normal(0.0, coeff_std, size=n_basis)

        # Sample which intervals are constrained
        forced_all = []
        for j in range(n_intervals):
            r = rng.random()
            if r < semantic_cfg.p_force_constant:
                forced_all.append((j, "constant"))
            elif r < semantic_cfg.p_force_constant + semantic_cfg.p_force_linear:
                forced_all.append((j, "linear"))

        # Enforce a maximum number of constrained intervals
        if len(forced_all) > semantic_cfg.max_forced_intervals:
            idx = rng.choice(len(forced_all), size=semantic_cfg.max_forced_intervals, replace=False)
            forced = [forced_all[k] for k in idx]
        else:
            forced = forced_all

        # Build linear constraint system A c = b
        rows = []
        rhs = []

        for (j, kind) in forced:
            # Cubic monomial: a0 + a1 t + a2 t² + a3 t³
            if kind == "linear":
                rows.append(P[2, j, :])  # a2 = 0
                rhs.append(0.0)
                rows.append(P[3, j, :])  # a3 = 0
                rhs.append(0.0)
            elif kind == "constant":
                rows.append(P[1, j, :])  # a1 = 0
                rhs.append(0.0)
                rows.append(P[2, j, :])  # a2 = 0
                rhs.append(0.0)
                rows.append(P[3, j, :])  # a3 = 0
                rhs.append(0.0)

        if rows:
            A = np.vstack(rows)
            bvec = np.asarray(rhs, float)
            c = _project_to_linear_constraints(c0, A, bvec)
        else:
            c = c0

        coeffs[i] = c
        u[i] = B @ c

    return u, coeffs, basis

def make_ts_0_1(n_timesteps: int) -> np.ndarray:
    """Uniform grid on [0, 1] with endpoint included."""
    return np.linspace(0.0, 1.0, n_timesteps)


def coeffs_to_frame(coeffs: np.ndarray, prefix: str = "c") -> pd.DataFrame:
    """Stores spline coefficients in a DataFrame."""
    return pd.DataFrame({f"{prefix}{j}": coeffs[:, j] for j in range(coeffs.shape[1])})


def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def beta_fn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes Beta(a, b) via log-space evaluation."""
    return np.exp(betaln(a, b))


# Three Synthetic Datasets used for van Der Schaar task given below
class DynamicTumorDataset(BaseDataset):
    """
    Tumour-growth synthetic dataset with:
      - Static covariates: [age, weight, initial_tumor_volume, dosage]
      - One dynamic exogenous input u(t) (e.g., glucose), generated as a B-spline per sample

    Dynamics are a bounded “Wilkerson-style” split-compartment update:
      - Vs: “sensitive” compartment decays with rate d(dose, weight)
      - Vr: “resistant” compartment grows with rate g_t(t), but is limited by crowding

    Parameter maps:
      phi(dose)  = sigmoid(PHI_0 * dose)           (initial split between Vs and Vr)
      d(dose,w)  = D_0 * dose / w                  (dose effect scaled by weight)
      g0(age)    = G_0 * sqrt(age/20)              (baseline growth from age)
      g_t(t)     = clip(g0 + alpha*u(t), g_clip)   (input-modulated growth, clipped)

    Discrete-time update (dt from the [0,1] grid unless overridden):
      Vs_{k+1} = Vs_k * exp(-dt * d)
      Vr_{k+1} = Vr_k * exp( dt * g_tk * crowd_k )
      crowd_k  = clip(1 - (Vs_k + Vr_k)/K, 0, 1)
      y_k      = Vs_k + Vr_k

    Carrying capacity K is *not* tied to V0:
      - K_mode="const":  K = K_const
      - K_mode="weight": K = K_0 + K_1 * weight

    Important constraint to avoid “starting above capacity”:
      pick V0_range so that V0_max < K_min (computed from the minimum possible weight).
      This avoids resampling or awkward initial conditions.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        n_timesteps: int = 60,
        seed: int = 0,
        n_basis: int = 8,
        coeff_std: float = 1.0,
        semantic_cfg: Optional["SemanticConstraintConfig"] = None,

        # How strongly u(t) perturbs the baseline growth
        alpha: float = 3.5,

        # Baseline parameter scales for the “Wilkerson-style” maps
        G_0: float = 2.0,
        D_0: float = 180.0,
        PHI_0: float = 10.0,
        g_clip: tuple = (-5.0, 5.0),

        # Carrying capacity config (kept independent of V0 by design)
        K_mode: str = "weight",          # {"const", "weight"}
        K_const: float = 1.4,            # used if K_mode=="const"
        K_0: float = 1.0,                # used if K_mode=="weight"
        K_1: float = 0.01,               # used if K_mode=="weight"
        K_clip: Optional[tuple] = None,  # optional extra cap/floor on K

        # Ranges for static covariates
        age_range=(20.0, 80.0),
        weight_range=(40.0, 100.0),
        dose_range=(0.0, 1.0),

        # Choose this so V0_max < K_min (so crowding starts sensible)
        V0_range=(0.1, 0.5),

        # Numerical safeguards
        V_min: float = 1e-4,
        dt: Optional[float] = None,
        hard_cap_to_K: bool = False,      # optional: clip output y(t) to K
    ):
        super().__init__(n_samples=n_samples, n_timesteps=n_timesteps)
        rng = np.random.default_rng(seed)

        self.alpha = float(alpha)

        self.G_0 = float(G_0)
        self.D_0 = float(D_0)
        self.PHI_0 = float(PHI_0)
        self.g_clip = (float(g_clip[0]), float(g_clip[1]))

        self.K_mode = str(K_mode)
        self.K_const = float(K_const)
        self.K_0 = float(K_0)
        self.K_1 = float(K_1)
        self.K_clip = K_clip

        self.age_range = (float(age_range[0]), float(age_range[1]))
        self.weight_range = (float(weight_range[0]), float(weight_range[1]))
        self.dose_range = (float(dose_range[0]), float(dose_range[1]))
        self.V0_range = (float(V0_range[0]), float(V0_range[1]))

        self.V_min = float(V_min)

        # Time grid is normalised to [0,1] across n_timesteps
        t = make_ts_0_1(n_timesteps)
        self.ts = [t.copy() for _ in range(n_samples)]
        self.dt = float(1.0 / (n_timesteps - 1) if dt is None else dt)

        # Compute K from statics (either constant or weight-dependent)
        def K_from_statics(weight_i: float) -> float:
            if self.K_mode == "const":
                K = self.K_const
            elif self.K_mode == "weight":
                K = self.K_0 + self.K_1 * weight_i
            else:
                raise ValueError(f"K_mode must be 'const' or 'weight', got {self.K_mode}")

            if self.K_clip is not None:
                K = float(np.clip(K, float(self.K_clip[0]), float(self.K_clip[1])))

            # Ensure K stays above a tiny positive floor
            return float(max(K, self.V_min + 1e-6))

        # Sanity check: ensure V0_range won't violate capacity at the smallest K
        w_min = self.weight_range[0]
        K_min = K_from_statics(w_min)

        if not (self.V0_range[1] < K_min):
            raise ValueError(
                f"V0_range must satisfy V0_max < K_min to avoid starting above carrying capacity.\n"
                f"Got V0_max={self.V0_range[1]:.4g}, but K_min={K_min:.4g}.\n"
                f"Fix by reducing V0_range[1] or increasing K_const / K_0 / K_1 (or K_clip min)."
            )

        # Sample static covariates (i.i.d. within the given ranges)
        age = rng.uniform(self.age_range[0], self.age_range[1], size=n_samples)
        weight = rng.uniform(self.weight_range[0], self.weight_range[1], size=n_samples)
        init_vol = rng.uniform(self.V0_range[0], self.V0_range[1], size=n_samples)
        dosage = rng.uniform(self.dose_range[0], self.dose_range[1], size=n_samples)

        self.X = pd.DataFrame(
            {"age": age, "weight": weight, "initial_tumor_volume": init_vol, "dosage": dosage}
        )

        # Sample u(t) as a spline and keep both values and coefficients
        u, c, basis = sample_inputs_from_bsplinebasis(
            n_samples=n_samples,
            t=t,
            n_basis=n_basis,
            coeff_std=coeff_std,
            seed=seed,
            random_internal_knots=True,
            semantic_cfg=semantic_cfg,
        )

        self.basis = basis
        self.X_dynamic = pd.DataFrame({"x_dynamic": [u[i] for i in range(n_samples)]})
        self.X_dynamic_coeffs = coeffs_to_frame(c, prefix="c_u_")

        # Convenience maps for parameterisation (kept local for readability)
        def g0_from_age(age_i: float) -> float:
            return self.G_0 * (age_i / 20.0) ** 0.5

        def d_from_dose_weight(dose_i: float, weight_i: float) -> float:
            return self.D_0 * dose_i / (weight_i + 1e-8)

        def phi_from_dose(dose_i: float) -> float:
            return 1.0 / (1.0 + np.exp(-self.PHI_0 * dose_i))

        # Simulate per-sample trajectories
        self.ys = []
        for i in range(n_samples):
            age_i = float(self.X.loc[i, "age"])
            w_i = float(self.X.loc[i, "weight"])
            V0 = float(self.X.loc[i, "initial_tumor_volume"])
            dose = float(self.X.loc[i, "dosage"])

            g0 = float(g0_from_age(age_i))
            d = float(d_from_dose_weight(dose, w_i))
            phi = float(phi_from_dose(dose))
            K = float(K_from_statics(w_i))

            Vs = np.zeros(n_timesteps, dtype=float)
            Vr = np.zeros(n_timesteps, dtype=float)

            # Initialise compartments so Vs(0)+Vr(0)=V0 (with a small positive floor)
            Vs[0] = max(self.V_min, phi * V0)
            Vr[0] = max(self.V_min, (1.0 - phi) * V0)

            ui = u[i]
            for k in range(n_timesteps - 1):
                # Input-modulated growth (optionally clipped via g_clip)
                g_t = g0 + self.alpha * float(ui[k])

                # Crowding limits growth as volume approaches K
                Vtot = float(Vs[k] + Vr[k])
                crowd = float(np.clip(1.0 - Vtot / K, 0.0, 1.0))

                Vs[k + 1] = max(self.V_min, Vs[k] * np.exp(-self.dt * d))
                Vr[k + 1] = max(self.V_min, Vr[k] * np.exp(self.dt * g_t * crowd))

            y = Vs + Vr
            if hard_cap_to_K:
                y = np.clip(y, 0.0, K)

            self.ys.append(y)

        # Precompute semantic decomposition for u(t) so downstream models can reuse it
        self._semantics = self._build_all_semantics()

    def __len__(self):
        return len(self.X)

    def get_X_ts_ys(self):
        return self.X, self.X_dynamic_coeffs, self.ts, self.ys

    def _build_all_semantics(self):
        return [self._build_semantics_for_sample(i) for i in range(len(self.X))]

    def _build_semantics_for_sample(self, i):
        template, transition_points = self.basis.get_template_from_coeffs(self.X_dynamic_coeffs.iloc[i])
        t = np.asarray(self.ts[i], dtype=float)
        u = np.asarray(self.X_dynamic.loc[i, "x_dynamic"], dtype=float)

        semantics_i = []
        for k, cls in enumerate(template):
            t0 = float(transition_points[k])
            t1 = float(transition_points[k + 1])

            # Map semantic endpoints (t0,t1) onto nearest indices on the discrete grid
            idx0 = int(np.searchsorted(t, t0, side="left"))
            idx1 = int(np.searchsorted(t, t1, side="right") - 1)
            idx0 = int(np.clip(idx0, 0, len(t) - 1))
            idx1 = int(np.clip(idx1, 0, len(t) - 1))

            # Store class + continuous endpoints and their sampled values
            semantics_i.append((int(cls), t0, t1, float(u[idx0]), float(u[idx1])))
        return semantics_i

    def get_semantics(self, i):
        return self._semantics[i]

    def get_feature_names(self):
        return ["age", "weight", "initial_tumor_volume", "dosage"]

    def get_feature_ranges(self):
        # x_dynamic is a per-sample normalised spline (most values typically within ~[-3,3])
        return {
            "age": self.age_range,
            "weight": self.weight_range,
            "initial_tumor_volume": self.V0_range,
            "dosage": self.dose_range,
            "x_dynamic": (-3.0, 3.0),
        }

    def _semantic_midpoint_derivative_magnitudes(self, i: int):
        """
        Compute |f'(mid)| and |f''(mid)| for each semantic segment of u(t),
        where mid = (t0+t1)/2.

        Uses:
          - semantic endpoints already stored in self._semantics[i]
          - the spline reconstructed directly from stored coefficients (no refit)
        """
        semantics_i = self.get_semantics(i)  # list of (cls, t0, t1, y0, y1)

        coeffs_1d = np.asarray(self.X_dynamic_coeffs.iloc[i], dtype=float).ravel()
        spline = self.basis.get_spline_with_coeffs(coeffs_1d)

        t0 = np.array([m[1] for m in semantics_i], dtype=float)
        t1 = np.array([m[2] for m in semantics_i], dtype=float)
        mids = 0.5 * (t0 + t1)

        f1_mag = np.abs(spline.derivative(nu=1)(mids)).astype(float)
        f2_mag = np.abs(spline.derivative(nu=2)(mids)).astype(float)
        return f1_mag, f2_mag, mids

    def encode_dynamic_interleaved_irregular_for_sample(
    self,
    i: int,
    append_last_endpoint_property: bool = True,
    ):
        """
        Build a flat, RNN-friendly encoding from semantic segments (irregular endpoints).

        Per-semantic token (length 5):
          [cls, |f'(mid)|, |f''(mid)|, dt, y0]
        Optionally append the very final endpoint value y1 once, so the sequence has
        one more “property” value than number of semantics.

        Output length:
          K*5 (+1 if append_last_endpoint_property)
        """
        semantics_i = self.get_semantics(i)  # (cls, t0, t1, y0, y1)
        if len(semantics_i) == 0:
            return np.zeros((0,), dtype=float)

        f1_mag, f2_mag, _ = self._semantic_midpoint_derivative_magnitudes(i)

        tokens = []
        for k, (cls, t0, t1, y0, y1) in enumerate(semantics_i):
            dt = float(t1) - float(t0)
            tokens.append([float(cls), float(f1_mag[k]), float(f2_mag[k]), float(dt), float(y0)])

        token_mat = np.asarray(tokens, dtype=float)  # (K, 5)

        flat = token_mat.reshape(-1)
        if append_last_endpoint_property:
            # Add y1 of the final semantic as a single trailing “endpoint property”
            flat = np.concatenate([flat, np.asarray([float(semantics_i[-1][4])], dtype=float)])

        return flat

    def get_X_dynamic_interleaved_irregular(
        self,
        append_last_endpoint_property: bool = True,
    ):
        encoded = [
            self.encode_dynamic_interleaved_irregular_for_sample(
                i, append_last_endpoint_property=append_last_endpoint_property
            )
            for i in range(len(self))
        ]
        return pd.DataFrame({"x_dynamic": encoded})


class DynamicSineTransDataset(BaseDataset):
    """
    Phase-modulated sine dataset.

      - Static covariate: x ~ Uniform(1,3)
      - Dynamic input: u(t) spline (one per sample)
      - Output: y(t) = sin(2*pi*t/x + alpha*u(t))

    “1-K pairing” means:
      - you draw N_static unique static values
      - each static value is repeated K times (with different u(t) per repetition)
      - total samples = N_static * K (any remainder is truncated for determinism)
    """
    def __init__(
        self,
        n_samples: int = 2000,   # TOTAL samples requested (will be truncated to a multiple of K)
        n_timesteps: int = 60,
        seed: int = 0,
        n_basis: int = 8,
        coeff_std: float = 1.0,
        semantic_cfg: Optional["SemanticConstraintConfig"] = None,
        alpha: float = 0.5,
        K: int = 5,
    ):
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")

        self.K = int(K)
        self.N_total = int(n_samples)
        self.N_static = self.N_total // self.K

        if self.N_static == 0:
            raise ValueError(
                f"n_samples={n_samples} too small for K={K}"
            )

        # Force N_total to be an exact multiple of K (keeps pairing exact and repeatable)
        self.N_total = self.N_static * self.K

        super().__init__(n_samples=self.N_total, n_timesteps=n_timesteps)

        rng = np.random.default_rng(seed)
        self.alpha = float(alpha)

        # Time grid in [0,1]
        t = make_ts_0_1(n_timesteps)
        self.ts = [t.copy() for _ in range(self.N_total)]

        # Unique static draws (these will be repeated K times each)
        X_static = pd.DataFrame({
            "x": rng.uniform(1.0, 3.0, size=self.N_static)
        })

        # Map each repeated sample back to its static row
        static_id = np.repeat(np.arange(self.N_static), self.K)

        # Expand statics to total samples (only true covariates are stored in self.X)
        self.X = X_static.iloc[static_id].reset_index(drop=True)

        # Dynamic inputs are unique per sample (even when statics repeat)
        u, c, basis = sample_inputs_from_bsplinebasis(
            n_samples=self.N_total,
            t=t,
            n_basis=n_basis,
            coeff_std=coeff_std,
            seed=seed,
            random_internal_knots=True,
            semantic_cfg=semantic_cfg,
        )
        self.basis = basis
        self.X_dynamic = pd.DataFrame({"x_dynamic": [u[i] for i in range(self.N_total)]})
        self.X_dynamic_coeffs = coeffs_to_frame(c, prefix="c_u_")

        # Generate trajectories
        self.ys = []
        for i in range(self.N_total):
            x_i = float(self.X.loc[i, "x"])   # single static feature
            y = np.sin(2.0 * np.pi * t / x_i + self.alpha * u[i])
            self.ys.append(y)

        # Cache semantic decomposition for u(t)
        self._semantics = self._build_all_semantics()

    def __len__(self):
        return self.N_total

    def get_X_ts_ys(self):
        return self.X, self.X_dynamic_coeffs, self.ts, self.ys

    def _build_all_semantics(self):
        return [self._build_semantics_for_sample(i) for i in range(self.N_total)]

    def _build_semantics_for_sample(self, i):
        template, transition_points = self.basis.get_template_from_coeffs(
            self.X_dynamic_coeffs.iloc[i]
        )
        t = np.asarray(self.ts[i], dtype=float)
        u = np.asarray(self.X_dynamic.loc[i, "x_dynamic"], dtype=float)

        semantics_i = []
        for k, cls in enumerate(template):
            t0 = float(transition_points[k])
            t1 = float(transition_points[k + 1])

            # Snap continuous endpoints to discrete indices for storing endpoint values
            idx0 = int(np.searchsorted(t, t0, side="left"))
            idx1 = int(np.searchsorted(t, t1, side="right") - 1)
            idx0 = int(np.clip(idx0, 0, len(t) - 1))
            idx1 = int(np.clip(idx1, 0, len(t) - 1))

            semantics_i.append(
                (int(cls), t0, t1, float(u[idx0]), float(u[idx1]))
            )
        return semantics_i

    def get_semantics(self, i):
        return self._semantics[i]

    def get_feature_names(self):
        return ["x"]

    def get_feature_ranges(self):
        return {"x": (1.0, 3.0), "x_dynamic": (-3.0, 3.0)}

    def _semantic_midpoint_derivative_magnitudes(self, i: int):
        """
        For each semantic segment of u(t), compute |f'(mid)| and |f''(mid)| with
        mid = (t0+t1)/2, using the stored spline coefficients (no refit).
        """
        semantics_i = self.get_semantics(i)  # list of (cls, t0, t1, y0, y1)

        coeffs_1d = np.asarray(self.X_dynamic_coeffs.iloc[i], dtype=float).ravel()
        spline = self.basis.get_spline_with_coeffs(coeffs_1d)

        t0 = np.array([m[1] for m in semantics_i], dtype=float)
        t1 = np.array([m[2] for m in semantics_i], dtype=float)
        mids = 0.5 * (t0 + t1)

        f1_mag = np.abs(spline.derivative(nu=1)(mids)).astype(float)
        f2_mag = np.abs(spline.derivative(nu=2)(mids)).astype(float)
        return f1_mag, f2_mag, mids

    def encode_dynamic_interleaved_irregular_for_sample(
    self,
    i: int,
    append_last_endpoint_property: bool = True,
    ):
        """
        Same interleaved “semantic token” encoding as the tumour dataset.

        Per semantic:
          [cls, |f'(mid)|, |f''(mid)|, dt, y0]
        Optionally append final y1.
        """
        semantics_i = self.get_semantics(i)  # (cls, t0, t1, y0, y1)
        if len(semantics_i) == 0:
            return np.zeros((0,), dtype=float)

        f1_mag, f2_mag, _ = self._semantic_midpoint_derivative_magnitudes(i)

        tokens = []
        for k, (cls, t0, t1, y0, y1) in enumerate(semantics_i):
            dt = float(t1) - float(t0)
            tokens.append([float(cls), float(f1_mag[k]), float(f2_mag[k]), float(dt), float(y0)])

        token_mat = np.asarray(tokens, dtype=float)  # (K, 5)

        flat = token_mat.reshape(-1)
        if append_last_endpoint_property:
            flat = np.concatenate([flat, np.asarray([float(semantics_i[-1][4])], dtype=float)])

        return flat

    def get_X_dynamic_interleaved_irregular(
        self,
        append_last_endpoint_property: bool = True,
    ):
        encoded = [
            self.encode_dynamic_interleaved_irregular_for_sample(
                i, append_last_endpoint_property=append_last_endpoint_property
            )
            for i in range(len(self))
        ]
        return pd.DataFrame({"x_dynamic": encoded})


class DynamicBetaDataset(BaseDataset):
    """
    Time-varying Beta PDF dataset.

      - Static covariates: (alpha, beta) laid out on a 2D grid
      - Dynamic input: u(t) spline (one per sample)
      - Convert statics + input into Beta shape parameters:
            a(t) = 1 + delta + softplus(alpha + lam*u(t))
            b(t) = 1 + delta + softplus(beta  - lam*u(t))
      - Output: y(t) = BetaPDF(t; a(t), b(t))

    “1-K pairing”:
      - build ~N_static_target unique static grid points (rounded to a square grid)
      - repeat each grid point K times with different u(t)
      - total samples = N_static * K
    """
    def __init__(
        self,
        n_samples: int = 2000,          # TOTAL samples requested (will be rounded to N_static*K)
        n_timesteps: int = 60,
        seed: int = 0,
        n_basis: int = 8,
        coeff_std: float = 1.0,
        semantic_cfg: Optional["SemanticConstraintConfig"] = None,
        lam: float = 0.8,
        delta: float = 1e-3,
        K: int = 5,
        alpha_range=(1.0, 4.0),
        beta_range=(1.0, 4.0),
    ):
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")

        self.K = int(K)
        self.N_total = int(n_samples)
        self.N_static_target = self.N_total // self.K
        if self.N_static_target == 0:
            raise ValueError(f"n_samples={n_samples} too small for K={K}")

        self.lam = float(lam)
        self.delta = float(delta)

        t = make_ts_0_1(n_timesteps)

        # Choose a square grid size close to the target number of unique statics
        n_per_dim = int(np.floor(np.sqrt(self.N_static_target)))
        n_per_dim = max(n_per_dim, 1)

        # Unique static points = perfect square, so we can form a clean meshgrid
        self.N_static = n_per_dim * n_per_dim

        # Total samples = N_static * K (exact pairing)
        self.N_total = self.N_static * self.K

        super().__init__(n_samples=self.N_total, n_timesteps=n_timesteps)
        self.ts = [t.copy() for _ in range(self.N_total)]

        # Build the (alpha,beta) grid
        a_grid = np.linspace(float(alpha_range[0]), float(alpha_range[1]), n_per_dim)
        b_grid = np.linspace(float(beta_range[0]), float(beta_range[1]), n_per_dim)
        A, B = np.meshgrid(a_grid, b_grid)
        cart = np.stack([A, B], axis=-1).reshape(-1, 2)  # (N_static, 2)
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        X_static = pd.DataFrame({"alpha": cart[:, 0], "beta": cart[:, 1]})

        # Map repeated samples back to static grid points
        static_id = np.repeat(np.arange(self.N_static), self.K)

        # Expand statics to all samples (only true covariates are stored in self.X)
        self.X = X_static.iloc[static_id].reset_index(drop=True)  # only alpha, beta

        # Optional bookkeeping for pairing / plotting (kept out of self.X)
        self.static_id = static_id
        self.rep_id = np.tile(np.arange(self.K), self.N_static)

        # Dynamic inputs are unique per sample
        u, c, basis = sample_inputs_from_bsplinebasis(
            n_samples=self.N_total,
            t=t,
            n_basis=n_basis,
            coeff_std=coeff_std,
            seed=seed,
            random_internal_knots=True,
            semantic_cfg=semantic_cfg,
        )
        self.basis = basis
        self.X_dynamic = pd.DataFrame({"x_dynamic": [u[i] for i in range(self.N_total)]})
        self.X_dynamic_coeffs = coeffs_to_frame(c, prefix="c_u_")

        # Avoid exactly 0 or 1 in the Beta PDF evaluation (numerical stability)
        eps = 1e-6
        tt = np.clip(t, eps, 1.0 - eps)

        self.ys = []
        for i in range(self.N_total):
            a0 = float(self.X.loc[i, "alpha"])
            b0 = float(self.X.loc[i, "beta"])

            # Time-varying Beta shape parameters
            a_t = 1.0 + self.delta + softplus(a0 + self.lam * u[i])
            b_t = 1.0 + self.delta + softplus(b0 - self.lam * u[i])

            y = (tt ** (a_t - 1.0)) * ((1.0 - tt) ** (b_t - 1.0)) / beta_fn(a_t, b_t)
            self.ys.append(y)

        # Cache semantic decomposition for u(t)
        self._semantics = self._build_all_semantics()

    def __len__(self):
        return self.N_total

    def get_X_ts_ys(self):
        return self.X, self.X_dynamic_coeffs, self.ts, self.ys

    def _build_all_semantics(self):
        return [self._build_semantics_for_sample(i) for i in range(self.N_total)]

    def _build_semantics_for_sample(self, i):
        template, transition_points = self.basis.get_template_from_coeffs(self.X_dynamic_coeffs.iloc[i])
        t = np.asarray(self.ts[i], dtype=float)
        u = np.asarray(self.X_dynamic.loc[i, "x_dynamic"], dtype=float)

        semantics_i = []
        for k, cls in enumerate(template):
            t0 = float(transition_points[k])
            t1 = float(transition_points[k + 1])

            # Snap semantic endpoints to discrete indices to store endpoint values
            idx0 = int(np.searchsorted(t, t0, side="left"))
            idx1 = int(np.searchsorted(t, t1, side="right") - 1)
            idx0 = int(np.clip(idx0, 0, len(t) - 1))
            idx1 = int(np.clip(idx1, 0, len(t) - 1))

            semantics_i.append((int(cls), t0, t1, float(u[idx0]), float(u[idx1])))
        return semantics_i

    def get_semantics(self, i):
        return self._semantics[i]

    def get_feature_names(self):
        return ["alpha", "beta"]

    def get_feature_ranges(self):
        return {
            "alpha": (float(self.alpha_range[0]), float(self.alpha_range[1])),
            "beta": (float(self.beta_range[0]), float(self.beta_range[1])),
            "x_dynamic": (-3.0, 3.0),
        }

    def _semantic_midpoint_derivative_magnitudes(self, i):
        """
        For each semantic segment of u(t), compute |f'(mid)| and |f''(mid)| with
        mid = (t0+t1)/2, using the stored spline coefficients (no refit).
        """
        semantics_i = self.get_semantics(i)  # list of (cls, t0, t1, y0, y1)

        coeffs_1d = np.asarray(self.X_dynamic_coeffs.iloc[i], dtype=float).ravel()
        spline = self.basis.get_spline_with_coeffs(coeffs_1d)

        t0 = np.array([m[1] for m in semantics_i], dtype=float)
        t1 = np.array([m[2] for m in semantics_i], dtype=float)
        mids = 0.5 * (t0 + t1)

        f1_mag = np.abs(spline.derivative(nu=1)(mids)).astype(float)
        f2_mag = np.abs(spline.derivative(nu=2)(mids)).astype(float)
        return f1_mag, f2_mag, mids

    def encode_dynamic_interleaved_irregular_for_sample(
    self,
    i: int,
    append_last_endpoint_property: bool = True,
    ):
        """
        Same interleaved “semantic token” encoding as the other datasets.

        Per semantic:
          [cls, |f'(mid)|, |f''(mid)|, dt, y0]
        Optionally append final y1.
        """
        semantics_i = self.get_semantics(i)  # (cls, t0, t1, y0, y1)
        if len(semantics_i) == 0:
            return np.zeros((0,), dtype=float)

        f1_mag, f2_mag, _ = self._semantic_midpoint_derivative_magnitudes(i)

        tokens = []
        for k, (cls, t0, t1, y0, y1) in enumerate(semantics_i):
            dt = float(t1) - float(t0)
            tokens.append([float(cls), float(f1_mag[k]), float(f2_mag[k]), float(dt), float(y0)])

        token_mat = np.asarray(tokens, dtype=float)  # (K, 5)

        flat = token_mat.reshape(-1)
        if append_last_endpoint_property:
            flat = np.concatenate([flat, np.asarray([float(semantics_i[-1][4])], dtype=float)])

        return flat

    def get_X_dynamic_interleaved_irregular(
        self,
        append_last_endpoint_property: bool = True,
    ):
        encoded = [
            self.encode_dynamic_interleaved_irregular_for_sample(
                i, append_last_endpoint_property=append_last_endpoint_property
            )
            for i in range(len(self))
        ]
        return pd.DataFrame({"x_dynamic": encoded})

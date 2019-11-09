from pomegranate import BayesianNetwork
import data_operations
import constants

if __name__ == '__main__':
    df_val = data_operations.load_dataframe(constants.NORMALIZED_SLICED_SMPL).values[:20]
    model = BayesianNetwork.from_samples(df_val, algorithm='chow-liu',max_parents=1,n_jobs=10)
    model.plot()
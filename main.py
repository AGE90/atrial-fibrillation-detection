import afdetection.utils.paths as path

from afdetection.data.make_dataset import MakeDataset
from afdetection.features.build_features import BuildFeatures
from afdetection.models.train_model import TrainModel

if __name__=="__main__":
    
    make_data = MakeDataset()
    dataset_DIR = path.data_raw_dir('dataset.csv')
    dataset = make_data.read_from_csv(dataset_DIR)
    
    build_features = BuildFeatures()
    X, y = build_features.features_target_split(
        dataset=dataset,
        drop_cols=['diagnosi', 'ritmi'],
        target='ritmi'
    )
    
    training = TrainModel()
    training.genopt_training(X, y)

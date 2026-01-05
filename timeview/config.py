from types import SimpleNamespace
import torch

ACTIVATION_FUNCTIONS = {
    'relu': torch.nn.ReLU,
    'sigmoid': torch.nn.Sigmoid,
    'tanh': torch.nn.Tanh,
    'leaky_relu': torch.nn.LeakyReLU,
    'elu': torch.nn.ELU,
    'selu': torch.nn.SELU
}

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop
}


class Config():

    def __init__(self,
                 n_features=1,
                 n_basis=5,
                 T=1,
                 seed=42,
                 encoder={'hidden_sizes': [32, 64, 32],
                          'activation': 'relu', 'dropout_p': 0.2},
                 training={'optimizer': 'adam', 'lr': 1e-3,
                           'batch_size': 32, 'weight_decay': 1e-5},
                 dataset_split={'train': 0.8, 'val': 0.1, 'test': 0.1},
                 dataloader_type='iterative',
                 device='cpu',
                 num_epochs=200,
                 internal_knots=None,
                 n_basis_tunable=False,
                 dynamic_bias=False):

        if not isinstance(n_features, int):
            raise ValueError("n_features must be an integer > 0")
        if n_basis < 4:
            raise ValueError("num_basis must be at least 4")
        if not isinstance(encoder['hidden_sizes'], list):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if not all([isinstance(x, int) for x in encoder['hidden_sizes']]):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if encoder['activation'] not in ACTIVATION_FUNCTIONS:
            raise ValueError("encoder['activation'] must be one of {}".format(
                list(ACTIVATION_FUNCTIONS.keys())))
        if training['optimizer'] not in OPTIMIZERS:
            raise ValueError("optimizer['name'] must be one of {}".format(
                list(OPTIMIZERS.keys())))
        if not isinstance(training['lr'], float):
            raise ValueError("optimizer['lr'] must be a float")
        if not isinstance(training['batch_size'], int):
            raise ValueError("training['batch_size'] must be an integer")
        if not isinstance(training['weight_decay'], float):
            raise ValueError("training['weight_decay'] must be a float")
        if not isinstance(dataset_split['train'], float):
            raise ValueError("dataset_split['train'] must be a float")
        if not isinstance(dataset_split['val'], float):
            raise ValueError("dataset_split['val'] must be a float")
        if not isinstance(dataset_split['test'], float):
            raise ValueError("dataset_split['test'] must be a float")
        if dataset_split['train'] + dataset_split['val'] + dataset_split['test'] != 1.0:
            raise ValueError(
                "dataset_split['train'] + dataset_split['val'] + dataset_split['test'] must equal 1.0")
        if dataloader_type not in ['iterative', 'tensor']:
            raise ValueError(
                "dataloader_type must be one of ['iterative','tensor']")

        self.n_basis = n_basis
        self.n_features = n_features
        self.T = T
        self.seed = seed
        self.encoder = SimpleNamespace(**encoder)
        self.training = SimpleNamespace(**training)
        self.dataset_split = SimpleNamespace(**dataset_split)
        self.dataloader_type = dataloader_type
        self.device = device
        self.num_epochs = num_epochs
        self.internal_knots = internal_knots
        self.n_basis_tunable = n_basis_tunable
        self.dynamic_bias = dynamic_bias

class DynamicConfig(Config):

    def __init__(self,
                 n_features=1,
                 n_features_dynamic=3,
                 n_basis=5,
                 T=1,
                 seed=42,
                 encoder={'hidden_sizes': [32, 64, 32],
                          'activation': 'relu', 'dropout_p': 0.2},
                 dynamic_encoder={'hidden_size': 64,  # Hidden size of the RNN
                                  'rnn_layers': 2,  # Number of RNN layers
                                  'dropout_p': 0.2,  # Dropout probability for RNN
                                  'rnn_type': 'lstm'},
                 training={'optimizer': 'adam', 'lr': 1e-3,
                           'batch_size': 32, 'weight_decay': 1e-5},
                 dataset_split={'train': 0.8, 'val': 0.1, 'test': 0.1},
                 dataloader_type='iterative',
                 device='cpu',
                 num_epochs=400,
                 internal_knots=None,
                 n_basis_tunable=False,
                 dynamic_bias=False):

        if not isinstance(n_features, int):
            raise ValueError("n_features must be an integer > 0")
        if n_basis < 4:
            raise ValueError("num_basis must be at least 4")
        if not isinstance(encoder['hidden_sizes'], list):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if not all([isinstance(x, int) for x in encoder['hidden_sizes']]):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if encoder['activation'] not in ACTIVATION_FUNCTIONS:
            raise ValueError("encoder['activation'] must be one of {}".format(
                list(ACTIVATION_FUNCTIONS.keys())))
        if training['optimizer'] not in OPTIMIZERS:
            raise ValueError("optimizer['name'] must be one of {}".format(
                list(OPTIMIZERS.keys())))
        if not isinstance(training['lr'], float):
            raise ValueError("optimizer['lr'] must be a float")
        if not isinstance(training['batch_size'], int):
            raise ValueError("training['batch_size'] must be an integer")
        if not isinstance(training['weight_decay'], float):
            raise ValueError("training['weight_decay'] must be a float")
        if not isinstance(dataset_split['train'], float):
            raise ValueError("dataset_split['train'] must be a float")
        if not isinstance(dataset_split['val'], float):
            raise ValueError("dataset_split['val'] must be a float")
        if not isinstance(dataset_split['test'], float):
            raise ValueError("dataset_split['test'] must be a float")
        if dataset_split['train'] + dataset_split['val'] + dataset_split['test'] != 1.0:
            raise ValueError(
                "dataset_split['train'] + dataset_split['val'] + dataset_split['test'] must equal 1.0")
        if dataloader_type not in ['iterative', 'tensor']:
            raise ValueError(
                "dataloader_type must be one of ['iterative','tensor']")

        self.n_basis = n_basis
        self.n_features = n_features
        self.n_features_dynamic = n_features_dynamic
        self.T = T
        self.seed = seed
        self.encoder = SimpleNamespace(**encoder)
        self.dynamic_encoder = SimpleNamespace(**dynamic_encoder)
        self.training = SimpleNamespace(**training)
        self.dataset_split = SimpleNamespace(**dataset_split)
        self.dataloader_type = dataloader_type
        self.device = device
        self.num_epochs = num_epochs
        self.internal_knots = internal_knots
        self.n_basis_tunable = n_basis_tunable
        self.dynamic_bias = dynamic_bias



class TuningConfig(Config):
    def __init__(
        self,
        trial,
        n_features=1,
        n_basis=5,
        T=1,
        seed=42,
        dataset_split={'train': 0.8, 'val': 0.1, 'test': 0.1},
        dataloader_type='iterative',
        device='cpu',
        num_epochs=200,
        internal_knots=None,
        n_basis_tunable=False,
        dynamic_bias=False
    ):

        # define hyperparameter search space
        hidden_sizes = [trial.suggest_int(
            f'hidden_size_{i}', 16, 128) for i in range(3)]
        # the activation search range might be a bit excessive, but it's a good example
        activation = trial.suggest_categorical(
            'activation',
            ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
        )
        dropout_p = trial.suggest_float('dropout_p', 0.0, 0.5)

        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128])
        weight_decay = trial.suggest_float(
            'weight_decay', 1e-6, 1e-1, log=True)
        
        if n_basis_tunable:
            n_basis = trial.suggest_int('n_basis', 5, 16)

        encoder = {
            'hidden_sizes': hidden_sizes,
            'activation': activation,
            'dropout_p': dropout_p
        }
        training = {
            'optimizer': 'adam',
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay
        }

        if not isinstance(n_features, int):
            raise ValueError("n_features must be an integer > 0")
        if n_basis < 4:
            raise ValueError("num_basis must be at least 4")
        if not isinstance(encoder['hidden_sizes'], list):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if not all([isinstance(x, int) for x in encoder['hidden_sizes']]):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if encoder['activation'] not in ACTIVATION_FUNCTIONS:
            raise ValueError("encoder['activation'] must be one of {}".format(
                list(ACTIVATION_FUNCTIONS.keys())))
        if training['optimizer'] not in OPTIMIZERS:
            raise ValueError("optimizer['name'] must be one of {}".format(
                list(OPTIMIZERS.keys())))
        if not isinstance(training['lr'], float):
            raise ValueError("optimizer['lr'] must be a float")
        if not isinstance(training['batch_size'], int):
            raise ValueError("training['batch_size'] must be an integer")
        if not isinstance(training['weight_decay'], float):
            raise ValueError("training['weight_decay'] must be a float")
        if not isinstance(dataset_split['train'], float):
            raise ValueError("dataset_split['train'] must be a float")
        if not isinstance(dataset_split['val'], float):
            raise ValueError("dataset_split['val'] must be a float")
        if not isinstance(dataset_split['test'], float):
            raise ValueError("dataset_split['test'] must be a float")
        if dataset_split['train'] + dataset_split['val'] + dataset_split['test'] != 1.0:
            raise ValueError(
                "dataset_split['train'] + dataset_split['val'] + dataset_split['test'] must equal 1.0")
        if dataloader_type not in ['iterative', 'tensor']:
            raise ValueError(
                "dataloader_type must be one of ['iterative','tensor']")

        self.n_basis = n_basis
        self.n_features = n_features
        self.T = T
        self.seed = seed
        self.encoder = SimpleNamespace(**encoder)
        self.training = SimpleNamespace(**training)
        self.dataset_split = SimpleNamespace(**dataset_split)
        self.dataloader_type = dataloader_type
        self.device = device
        self.num_epochs = num_epochs
        self.internal_knots = internal_knots
        self.n_basis_tunable = n_basis_tunable
        self.dynamic_bias = dynamic_bias
        

class DynamicTuningConfig(Config):
    def __init__(
        self,
        trial,
        n_features=1,
        n_features_dynamic=3,
        n_basis=5,
        T=1,
        seed=0,
        dataset_split={'train': 0.8, 'val': 0.1, 'test': 0.1},
        dataloader_type='iterative',
        device='cpu',
        num_epochs=200,
        internal_knots=None,
        n_basis_tunable=False,
        dynamic_bias=False
    ):

        # define hyperparameter search space
        hidden_sizes = [trial.suggest_int(
            f'hidden_size_{i}', 16, 128) for i in range(3)]
        # the activation search range might be a bit excessive, but it's a good example
        activation = trial.suggest_categorical(
            'activation',
            ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
        )
        dropout_p = trial.suggest_float('dropout_p', 0.0, 0.5)

        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128])
        weight_decay = trial.suggest_float(
            'weight_decay', 1e-6, 1e-1, log=True)
        
        if n_basis_tunable:
            n_basis = trial.suggest_int("rnn_n_basis_out", 5, 16)

        dynamic_hidden_size = trial.suggest_int('dynamic_hidden_size', 16, 128)
        rnn_type = trial.suggest_categorical('rnn_type', ['lstm', 'gru'])
        rnn_layers = trial.suggest_int('rnn_layers', 1, 3)
        dynamic_dropout_p = trial.suggest_float('dynamic_dropout_p', 0.0, 0.5)

        encoder = {
            'hidden_sizes': hidden_sizes,
            'activation': activation,
            'dropout_p': dropout_p
        }
        dynamic_encoder = {
            'hidden_size': dynamic_hidden_size,
            'rnn_type': rnn_type,
            'rnn_layers': rnn_layers,
            'dropout_p': dynamic_dropout_p
        }
        training = {
            'optimizer': 'adam',
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay
        }

        if not isinstance(n_features, int):
            raise ValueError("n_features must be an integer > 0")
        if n_basis < 4:
            raise ValueError("num_basis must be at least 4")
        if not isinstance(encoder['hidden_sizes'], list):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if not all([isinstance(x, int) for x in encoder['hidden_sizes']]):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if encoder['activation'] not in ACTIVATION_FUNCTIONS:
            raise ValueError("encoder['activation'] must be one of {}".format(
                list(ACTIVATION_FUNCTIONS.keys())))
        if training['optimizer'] not in OPTIMIZERS:
            raise ValueError("optimizer['name'] must be one of {}".format(
                list(OPTIMIZERS.keys())))
        if not isinstance(training['lr'], float):
            raise ValueError("optimizer['lr'] must be a float")
        if not isinstance(training['batch_size'], int):
            raise ValueError("training['batch_size'] must be an integer")
        if not isinstance(training['weight_decay'], float):
            raise ValueError("training['weight_decay'] must be a float")
        if not isinstance(dataset_split['train'], float):
            raise ValueError("dataset_split['train'] must be a float")
        if not isinstance(dataset_split['val'], float):
            raise ValueError("dataset_split['val'] must be a float")
        if not isinstance(dataset_split['test'], float):
            raise ValueError("dataset_split['test'] must be a float")
        if dataset_split['train'] + dataset_split['val'] + dataset_split['test'] != 1.0:
            raise ValueError(
                "dataset_split['train'] + dataset_split['val'] + dataset_split['test'] must equal 1.0")
        if dataloader_type not in ['iterative', 'tensor']:
            raise ValueError(
                "dataloader_type must be one of ['iterative','tensor']")

        self.n_basis = n_basis
        self.n_features = n_features
        self.n_features_dynamic = n_features_dynamic
        self.T = T
        self.seed = seed
        self.encoder = SimpleNamespace(**encoder)
        self.dynamic_encoder = SimpleNamespace(**dynamic_encoder)
        self.training = SimpleNamespace(**training)
        self.dataset_split = SimpleNamespace(**dataset_split)
        self.dataloader_type = dataloader_type
        self.device = device
        self.num_epochs = num_epochs
        self.internal_knots = internal_knots
        self.n_basis_tunable = n_basis_tunable
        self.dynamic_bias = dynamic_bias



#Below are config classes needed for semantic transformer and hyperparam optimisation

class SemanticTransformerConfig(Config):

    def __init__(
        self,
        n_features=2,                    # dimension of static covariate vector X_static
        n_semantic_features=4,           # dimension of continuous semantic descriptor per token
        n_motif_classes=7,               # number of discrete semantic / motif classes
        n_basis=5,                       # number of spline coefficients predicted per sample
        T=1,
        seed=0,
        internal_knots=None,             # fixed internal knots for spline basis (set externally)

        # Static covariates encoder: maps X_static -> static token
        static_encoder={
            'hidden_sizes': [64],
            'activation': 'relu',
            'dropout_p': 0.1
        },

        # Semantic token encoder:
        # - discrete class id -> embedding
        # - continuous semantic features -> MLP
        semantic_encoder={
            'class_emb_dim': 32,         # embedding size for semantic class id
            'cont_hidden': 64,           # hidden size for continuous-feature MLP
            'dropout_p': 0.1
        },

        # Transformer backbone operating over static + semantic tokens
        transformer={
            'd_model': 128,              # token dimension used throughout the transformer
            'n_heads': 4,                # number of attention heads
            'n_layers': 2,
            'd_ff': 512,                 # feed-forward network width
            'dropout_p': 0.1
        },

        # Optimiser and training-related hyperparameters
        training={
            'optimizer': 'adam',
            'lr': 1e-3,
            'batch_size': 32,
            'weight_decay': 1e-5
        },

        dataset_split={'train': 0.8, 'val': 0.1, 'test': 0.1},
        dataloader_type='iterative',
        device='cpu',
        num_epochs=100,
        dynamic_bias=False              # whether an additional bias term is included in the output head
    ):

        # Structural validity checks
        if n_basis < 4:
            raise ValueError("n_basis must be >= 4")
        if transformer['d_model'] % transformer['n_heads'] != 0:
            raise ValueError("d_model must be divisible by n_heads")

        # Core problem dimensions
        self.n_features = n_features
        self.n_semantic_features = n_semantic_features
        self.n_motif_classes = n_motif_classes
        self.n_basis = n_basis
        self.T = T
        self.seed = seed

        # Sub-configs stored as namespaces for dot access
        self.static_encoder = SimpleNamespace(**static_encoder)
        self.semantic_encoder = SimpleNamespace(**semantic_encoder)
        self.transformer = SimpleNamespace(**transformer)

        # Training and runtime settings
        self.training = SimpleNamespace(**training)
        self.dataset_split = SimpleNamespace(**dataset_split)
        self.dataloader_type = dataloader_type
        self.device = device
        self.num_epochs = num_epochs
        self.dynamic_bias = dynamic_bias

        # Spline knot configuration (should be derived from training data only)
        self.internal_knots = internal_knots



# Configuration used during hyperparameter tuning with Optuna.
# This class defines the search space and resolves trial-dependent values,
# but does not apply seeding or perform any training itself.

class SemanticTransformerTuningConfig(Config):
    def __init__(
        self,
        trial,
        n_features: int,
        n_semantic_features: int = 4,
        n_motif_classes: int = 7,
        n_basis: int = 8,
        T: float = 1.0,
        seed: int = 0,
        internal_knots=None,
        dataset_split={'train': 0.8, 'val': 0.1, 'test': 0.1},
        dataloader_type: str = 'iterative',
        device: str = 'cpu',
        num_epochs: int = 100,
        n_basis_tunable: bool = False,
        dynamic_bias: bool = False,
    ):
        # Static token encoder hyperparameters
        static_hidden = trial.suggest_int("static_hidden", 16, 128)
        static_activation = trial.suggest_categorical(
            "static_activation",
            ["relu", "sigmoid", "tanh", "leaky_relu", "elu", "selu"],
        )
        static_dropout = trial.suggest_float("static_dropout", 0.0, 0.5)

        static_encoder = {
            "hidden_sizes": [static_hidden],
            "activation": static_activation,
            "dropout_p": static_dropout,
        }

        # Semantic token encoder hyperparameters
        class_emb_dim = trial.suggest_categorical("class_emb_dim", [16, 32, 64])
        cont_hidden = trial.suggest_categorical("cont_hidden", [32, 64, 128, 256])
        semantic_dropout = trial.suggest_float("semantic_dropout", 0.0, 0.5)

        semantic_encoder = {
            "class_emb_dim": class_emb_dim,
            "cont_hidden": cont_hidden,
            "dropout_p": semantic_dropout,
        }

        # Transformer architecture hyperparameters
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        n_layers = trial.suggest_int("n_layers", 1, 4)
        ff_mult = trial.suggest_categorical("ff_mult", [2, 4, 8])
        d_ff = int(ff_mult * d_model)
        tr_dropout = trial.suggest_float("tr_dropout", 0.0, 0.5)

        transformer = {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "dropout_p": tr_dropout,
        }

        # Optimisation hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

        training = {
            "optimizer": "adam",
            "lr": float(lr),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
        }

        # Optional tuning of spline basis size
        if n_basis_tunable:
            n_basis = trial.suggest_int("tf_n_basis", 5, 16)

        # Basic configuration validation
        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError("n_features must be an integer > 0")
        if n_basis < 4:
            raise ValueError("n_basis must be >= 4")
        if dataset_split["train"] + dataset_split["val"] + dataset_split["test"] != 1.0:
            raise ValueError("dataset_split fractions must sum to 1.0")
        if dataloader_type not in ["iterative", "tensor"]:
            raise ValueError("dataloader_type must be one of ['iterative','tensor']")

        # Resolved configuration values
        self.n_features = n_features
        self.n_semantic_features = n_semantic_features
        self.n_motif_classes = n_motif_classes
        self.n_basis = n_basis
        self.T = T
        self.seed = seed
        self.internal_knots = internal_knots

        self.static_encoder = SimpleNamespace(**static_encoder)
        self.semantic_encoder = SimpleNamespace(**semantic_encoder)
        self.transformer = SimpleNamespace(**transformer)
        self.training = SimpleNamespace(**training)

        self.dataset_split = SimpleNamespace(**dataset_split)
        self.dataloader_type = dataloader_type
        self.device = device
        self.num_epochs = num_epochs
        self.n_basis_tunable = n_basis_tunable
        self.dynamic_bias = dynamic_bias
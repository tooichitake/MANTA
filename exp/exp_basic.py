import importlib
import os
import torch

# Lazy model registry: model_name -> module_path
_MODEL_REGISTRY = {
    "TimesNet": "models.TimesNet",
    "Autoformer": "models.Autoformer",
    "Transformer": "models.Transformer",
    "Nonstationary_Transformer": "models.Nonstationary_Transformer",
    "DLinear": "models.DLinear",
    "FEDformer": "models.FEDformer",
    "Informer": "models.Informer",
    "LightTS": "models.LightTS",
    "Reformer": "models.Reformer",
    "ETSformer": "models.ETSformer",
    "Pyraformer": "models.Pyraformer",
    "PatchTST": "models.PatchTST",
    "MICN": "models.MICN",
    "Crossformer": "models.Crossformer",
    "FiLM": "models.FiLM",
    "iTransformer": "models.iTransformer",
    "Koopa": "models.Koopa",
    "TiDE": "models.TiDE",
    "FreTS": "models.FreTS",
    "MambaSimple": "models.MambaSimple",
    "TimeMixer": "models.TimeMixer",
    "TSMixer": "models.TSMixer",
    "SegRNN": "models.SegRNN",
    "TemporalFusionTransformer": "models.TemporalFusionTransformer",
    "SCINet": "models.SCINet",
    "PAttn": "models.PAttn",
    "TimeXer": "models.TimeXer",
    "WPMixer": "models.WPMixer",
    "MultiPatchFormer": "models.MultiPatchFormer",
    "KANAD": "models.KANAD",
    "MANTA": "models.MANTA",
    "MANTA_CUDA": "models.MANTA_CUDA",
    "MANTA_Ablation": "models.MANTA_Ablation",
    "MANTA_origin": "models.MANTA_origin",
    "MANTA3": "models.MANTA3",
    "Mamba": "models.Mamba",
}

# Cache for already-imported model classes
_loaded_models = {}


def get_model_class(model_name):
    """Lazily import and return the Model class for the given model name."""
    if model_name not in _loaded_models:
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {sorted(_MODEL_REGISTRY.keys())}"
            )
        module = importlib.import_module(_MODEL_REGISTRY[model_name])
        _loaded_models[model_name] = module.Model
    return _loaded_models[model_name]


class _LazyModelDict:
    """Dict-like object that imports model classes on first access."""

    def __getitem__(self, model_name):
        return get_model_class(model_name)

    def __contains__(self, model_name):
        return model_name in _MODEL_REGISTRY


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = _LazyModelDict()
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == "mps":
            device = torch.device("mps")
            print("Use GPU: mps")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

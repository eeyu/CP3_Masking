import paths
import pickle
import torch
from torch.nn import Module
from model.ModelParameters import ModelParameters
from data.GenreateTopologies import Constraint

def save_model(model : Module, name):
    torch.save(model.state_dict(), paths.PARAM_PATH + "_" + name)

def load_model(model : Module, name):
    model.load_state_dict(torch.load(paths.PARAM_PATH + "_" + name))
    model.eval()

def save_model_and_hp(model : Module, hyperparam : ModelParameters, batch_size, name, constraint: Constraint | None = None):
    dict = {}
    dict["param"] = model.state_dict()
    dict["hyperparam"] = hyperparam
    dict["batch_size"] = batch_size
    if constraint is not None:
        dict["constraint"] = constraint
    with open(paths.PARAM_PATH + "_" + name, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model_from_file(file: str | None = None, start_path = None):
    if file is None:
        filename = paths.select_file(choose_file=True)
    else:
        filename = file
    with open(filename, 'rb') as handle:
        dict = pickle.load(handle)
        hyperparam : ModelParameters = dict["hyperparam"]
        param = dict["param"]
        batch_size = dict["batch_size"]
        constraint = None
        if "constraint" in dict:
            constraint = dict["constraint"]

        model = hyperparam.instantiate_new_model()
        model.load_state_dict(param)
        model = model.to(device=paths.device)
        if constraint is None:
            return model, batch_size
        else:
            return model, batch_size, constraint

def save_gan_model_and_hp(generator : Module, discriminator: Module, hyperparam : ModelParameters, batch_size, name, constraint: Constraint):
    dict = {}
    dict["param_generator"] = generator.state_dict()
    dict["param_discriminator"] = discriminator.state_dict()
    dict["hyperparam"] = hyperparam
    dict["batch_size"] = batch_size
    dict["constraint"] = constraint
    with open(paths.PARAM_PATH + "_" + name, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_gan_model_from_file(file: str | None = None, start_path = None):
    if file is None:
        filename = paths.select_file(choose_file=True)
    else:
        filename = file
    with open(filename, 'rb') as handle:
        dict = pickle.load(handle)
        hyperparam : ModelParameters = dict["hyperparam"]
        param_generator = dict["param_generator"]
        param_discriminator = dict["param_discriminator"]
        batch_size = dict["batch_size"]
        constraint = dict["constraint"]

        generator, discriminator = hyperparam.instantiate_new_model()
        generator.load_state_dict(param_generator)
        generator = generator.to(device=paths.device)

        discriminator.load_state_dict(param_discriminator)
        discriminator = discriminator.to(device=paths.device)

        return generator, discriminator, constraint

def get_save_name(advisor, modelName, algName, run_index, extension):
    return "A" + str(advisor) + "_" + "R" + str(run_index) + "_" + modelName + "_" + algName  + extension

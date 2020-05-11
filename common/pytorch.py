import torch


def save_model(path, model):
    # Save a trained model
    print("** ** * Saving fine - tuned model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = path
    torch.save(model_to_save.state_dict(), output_model_file)


def load_model(path, model):
    # load pretrained model and update weights
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model


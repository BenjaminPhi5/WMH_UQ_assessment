import torch
from tqdm import tqdm
from trustworthai.utils.losses_and_metrics.evidential_bayes_risks import softplus_evidence, get_alpha, get_S, get_mean_p_hat


def punet_mean_and_samples(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    
    model_raw(x.swapaxes(0,1).cuda(), y.cuda(), training=False)
    mean = model_raw.sample(use_prior_mean=True).cpu()
    
    ind_samples = []
    for j in range(num_samples):
                ind_samples.append(model_raw.sample(testing=False).cpu())

    ind_samples = torch.stack(ind_samples, dim=0)
    
    return mean, ind_samples


def evid_mean(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    logits = model_raw(x.swapaxes(0,1).cuda()).cpu()
    evidence = softplus_evidence(logits)
    alpha = get_alpha(evidence)
    # print(alpha.shape)
    S = get_S(alpha)
    K = alpha.shape[1]
    mean_p_hat = get_mean_p_hat(alpha, S)
    return mean_p_hat, None

def deterministic_mean(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    pass

def ssn_mean_and_samples(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    mean, sample = model_raw.mean_and_sample(x.swapaxes(0,1).cuda(), num_samples=num_samples, temperature=1)
    
    return mean, sample

def mc_drop_mean_and_samples(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    x = x.swapaxes(0,1).cuda()
    mean = model_raw(x)
    model_raw.train()
    ind_samples = []
    for j in range(num_samples):
        ind_samples.append(model_raw(x))
    model_raw.eval()
    ind_samples = torch.stack(ind_samples, dim=0)
    return mean, ind_samples

def ensemble_mean_and_sample(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    pass

def ssn_ensemble_mean_and_sample(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    pass

def get_means_and_samples(model_raw, eval_ds, num_samples, model_func, extra_kwargs={}):
    means = []
    samples = []

    model_raw.eval()
    for i, data in enumerate(tqdm(eval_ds, position=0, leave=True)):
        x = data[0]
        y = data[1]
        inputs = {**{"model":model_raw, "x":x, "y":y, "num_samples":num_samples}, **extra_kwargs}
        with torch.no_grad():
            mean, sample = model_func(inputs)
            means.append(mean.cpu())
            if sample != None:
                sample = sample.cpu()
            samples.append(sample)
            
    return means, samples

def reorder_samples(sample):
    sample = sample.cuda()
    slice_volumes = sample.argmax(dim=2).sum(dim=(-1, -2))
    slice_volume_orders = torch.sort(slice_volumes.T, dim=1)[1]
    
    # rearrange the samples into one...
    new_sample = torch.zeros(sample.shape).to(sample.device)
    for i, slice_volumes_orders in enumerate(slice_volume_orders):
        for j, sample_index in enumerate(slice_volumes_orders):
            new_sample[j][i] = sample[sample_index][i]
            
    return new_sample.cpu()
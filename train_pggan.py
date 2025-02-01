# %%
import torch
from models.PGGAN import GeneratorPGGAN, DiscriminatorPGGAN
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F
from collections import namedtuple
import torch.autograd as autograd
from torchvision import datasets
import torchvision.transforms as transforms


# %%
out_res = (256,256)
latent_dim = 512
device = "mps"
check_point_dir = "trained_nets/pggan"
lr = 1E-3
resume = False
resume_seq = 3

# Set training schedule

# %%
schedule = namedtuple('schedule', ['n_warm_up', 'n_train', 'batch_size', 'size'])

def set_schedule(batch_size, num_train_imgs, num_warm_up_imgs):
    n_iter_warm_up = num_warm_up_imgs//batch_size
    n_iter_train = num_train_imgs//batch_size
    return n_iter_warm_up, n_iter_train

training_schedule = [  
                    schedule(n_warm_up=0,       n_train=50_000,   batch_size=16,   size=4), # 4 x 4
                    schedule(n_warm_up=50_000,  n_train=50_000,   batch_size=16,   size=8), # 8 x 8
                    schedule(n_warm_up=50_000,  n_train=50_000,   batch_size=16,   size=16), # 16 x 16
                    schedule(n_warm_up=30_000,  n_train=30_000,   batch_size=16,  size=32), # 32 x 32
                    schedule(n_warm_up=20_000,  n_train=20_000,   batch_size=16,  size=64), # 64 x 64
                    schedule(n_warm_up=20_000,  n_train=20_000,   batch_size=8,  size=128), # 128 x 128
                    schedule(n_warm_up=20_000,  n_train=20_000,   batch_size=8,  size=256), # 256 x 256           
                    ]


# Set transforms

# %%
transform = transforms.Compose([
			transforms.Resize(out_res),
			transforms.CenterCrop(out_res),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])

train_set = dataset = datasets.ImageFolder("Data/celeb", transform=transform)

# %%
g = GeneratorPGGAN(latent_dim).to(device)
d = DiscriminatorPGGAN(latent_dim).to(device)

# %%
d_optimizer = torch.optim.Adam(d.parameters(), lr=lr, betas=(0, 0.99))
g_optimizer = torch.optim.Adam(g.parameters(), lr=lr, betas=(0, 0.99))

# Gradient penalty for Wasserstein Loss

# %%
def compute_grad_penalty(discriminator, imgs, fake_imgs):
    epsilon = torch.rand(imgs.size(0), 1, 1, 1).to(device).expand_as(imgs)
    x_hat = (epsilon * imgs + ((1 - epsilon) * fake_imgs.detach())).requires_grad_(True)
    out = discriminator(x_hat)
    grad = autograd.grad(
            outputs=out,
            inputs = x_hat,
            grad_outputs = torch.ones_like(out).to(device),
            retain_graph = True,
            create_graph = True,
            only_inputs = True
        )[0]
    # compute norm of the gradient
    grad_l2norm = grad.norm(2, dim=[1,2,3])
    # compute expected values
    gradient_penalty = torch.mean((grad_l2norm - 1) ** 2)

    return gradient_penalty

# %%
def epoch(imgs, lambda_gp=10):
    
    # train discriminator
    for p in d.parameters():
        p.requires_grad = True  # to avoid computation
    d_optimizer.zero_grad()
    z = torch.randn(imgs.size(0), latent_dim, 1, 1, device=device)
    fake_imgs = g(z)
    
    fake_validity = d(fake_imgs.detach())
    real_validity = d(imgs)

    # Wasserstein Loss + gradient penalty
    gradient_penalty = compute_grad_penalty(d, imgs, fake_imgs)
    d_loss = fake_validity.mean() -real_validity.mean() + lambda_gp * gradient_penalty

    d_loss.backward()
    d_optimizer.step()

    # train generator
    for p in d.parameters():
        p.requires_grad = False  # to avoid computation
    g_optimizer.zero_grad()

    fake_imgs = g(z)
    fake_validity = d(fake_imgs)
    
    g_loss = - torch.mean(fake_validity)
    g_loss.backward()
    g_optimizer.step()

    return d_loss.item(), g_loss.item()

# %%
def save_checkpoint(check_point_dir, n_seq):
	check_point = {'G_net' : g.state_dict(), 
				   'G_optimizer' : g_optimizer.state_dict(),
				   'D_net' : d.state_dict(),
				   'D_optimizer' : d_optimizer.state_dict(),
				   'depth': g.depth,
				   'alpha':g.alpha,
				   'resume_seq' : n_seq + 1
				   }
	with torch.no_grad():
		torch.save(check_point, os.path.join(check_point_dir ,'check_point_nseq_%d.pth'%(n_seq)))
		torch.save(g.state_dict(), os.path.join(check_point_dir , 'G_weight_nseq_%d.pth' %(n_seq)))

def resume_calculation(check_point_dir, n_seq):
	check_point = torch.load(os.path.join(check_point_dir ,'check_point_nseq_%d.pth'%(n_seq)))
	g.load_state_dict(check_point['G_net'])
	d.load_state_dict(check_point['D_net'])
	g_optimizer.load_state_dict(check_point['G_optimizer'])
	d_optimizer.load_state_dict(check_point['D_optimizer'])
	g.depth = check_point['depth']
	d.depth = check_point['depth']
	g.alpha = check_point['alpha']
	resume_seq = check_point['resume_seq']
	return resume_seq
	

# %%
start_schedule = 0
d_running_loss = 0.0
g_running_loss = 0.0
d_losses_warm, d_losses_train, g_losses_warm, g_losses_train = [], [], [], []
fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

if resume:
    start_schedule = resume_calculation(check_point_dir, resume_seq)

for n_schedule in range(start_schedule, len(training_schedule)):
    g.train()
    d.train()
    n_it_warm, n_it_train = set_schedule(training_schedule[n_schedule].batch_size, training_schedule[n_schedule].n_train, training_schedule[n_schedule].n_warm_up)
    data_loader = DataLoader(dataset=train_set, batch_size=training_schedule[n_schedule].batch_size, shuffle=True, num_workers=0, drop_last=True)

    if n_schedule > 0:
        g.increase_net(n_it_warm)
        d.increase_net(n_it_warm)

    # FADE-IN training
    bar = tqdm(range(n_it_warm))
    for i in bar:
        imgs, _ = next(iter(data_loader))
        if training_schedule[n_schedule].size != out_res:
            imgs = F.interpolate(imgs, size=training_schedule[n_schedule].size)
        imgs = imgs.to(device)
        d_loss, g_loss = epoch(imgs)
        # increase alpha
        d.step()
        g.step()
        d_running_loss += d_loss
        g_running_loss += g_loss
        if i%50 == 0 and i > 0:
            bar.set_description('d loss: %.3f   g loss: %.3f' % (d_running_loss/i , g_running_loss/i))
    if n_it_warm > 0:
        d_losses_warm.append(d_running_loss/n_it_warm )
        g_losses_warm.append(g_running_loss/n_it_warm )
        print(f"End warm up   d: {d_losses_warm[-1]:4.4f}\t  g:{g_losses_warm[-1]:4.4f}")

    # ensure no fade in
    d.alpha = 1
    g.alpha = 1

    # NO FADE-IN training
    bar = tqdm(range(n_it_train))
    for i in bar:
        imgs, _ = next(iter(data_loader))
        if training_schedule[n_schedule].size != out_res:
            imgs = F.interpolate(imgs, size=training_schedule[n_schedule].size)
        imgs = imgs.to(device)
        d_loss, g_loss = epoch(imgs)
        d_running_loss += d_loss
        g_running_loss += g_loss
        if i%50 == 0 and i > 0:
            bar.set_description('d loss: %.3f   g loss: %.3f' % (d_running_loss/i , g_running_loss/i))
    d_losses_train.append(d_running_loss/n_it_train )
    g_losses_train.append(g_running_loss/n_it_train )
    print(f"End train   d: {d_losses_train[-1]:4.4f}\t  g:{g_losses_train[-1]:4.4f}")

    # save checkpoints and plot generated images
    with torch.no_grad():
        g.eval()
        fig = plt.figure(figsize=(15,6))
        out_imgs = g(fixed_noise)
        out_grid = make_grid(out_imgs, normalize=True, nrow=8, scale_each=True, padding=int(0.5*(2**g.depth))).permute(1,2,0)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(out_grid.cpu())
        plt.savefig(os.path.join(check_point_dir , 'img_schedule_%i'%(n_schedule)))
    
    if n_schedule < len(training_schedule):
        print(f"End schedule {n_schedule:2.0f}\t start size {training_schedule[n_schedule+1].size} x {training_schedule[n_schedule+1].size}")
    save_checkpoint(check_point_dir, n_seq=n_schedule)
 


# %%




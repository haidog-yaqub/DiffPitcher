from models.unet import UNetVC
import torch

if __name__ == "__main__":
    model = UNetVC(256, 128, False, False, True, pitch_type='log')
    mean = torch.rand(4, 100, 128)
    x = torch.rand(4, 100, 128)
    f0 = torch.rand(4, 128)
    # f0 = torch.randint(0, 256, (4, 128), ).long()
    t = torch.randint(0, 256, (4,),).long()

    y = model(x, mean, f0, t, ref=None, embed=None)

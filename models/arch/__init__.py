import torch

from models.arch.dsrnet import DSRNet, MDSRNet, MXDSRNet, MGDSRNet


def dsrnet_s(in_channels=3, out_channels=3, width=32):
    enc_blks = [2, 2, 2]
    middle_blk_num = 4
    dec_blks = [2, 2, 2]

    return DSRNet(in_channels, out_channels, width=width,
                  middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks,
                  dec_blk_nums=dec_blks,
                  shared_b=False)


def dsrnet_l(in_channels=3, out_channels=3, width=64):
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]
    # print(f">>>>>>>>>>>{in_channels}, {out_channels}, {width}")
    return DSRNet(in_channels, out_channels, width=width,
                  middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks,
                  dec_blk_nums=dec_blks,
                  shared_b=True)

def dsrnet_m(in_channels=3, out_channels=3, width=64):
    enc_blks = [2, 2, 2, 2]
    middle_blk_num = 4
    dec_blks = [2, 2, 2, 2]
    # print(f">>>>>>>>>>>{in_channels}, {out_channels}, {width}")
    return DSRNet(in_channels, out_channels, width=width,
                  middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks,
                  dec_blk_nums=dec_blks,
                  shared_b=True)


# Our models are below

# MDSRNet v1 used the same function, since it failed and there was no intuition behind
# how Mamba was placed, I just updated the MDSRNet instead of writing a new one

# M2DSRNet, numbers indicate the number of MuGI blocks - similar to DSRNet
def mdsrnet_l(in_channels=3, out_channels=3, width=64):
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    return MDSRNet(in_channels, out_channels, width=width,
                  middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks,
                  dec_blk_nums=dec_blks,
                  shared_b=True)

# MXDSRNet
def mxdsrnet_l(in_channels=3, out_channels=3, width=64):
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]
    print("MXDSRNET...")
    return MXDSRNet(in_channels, out_channels, width=width,
                  middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks,
                  dec_blk_nums=dec_blks,
                  shared_b=True)

# MGDSRNet -> a failed model in the report - needs more work
def mgdsrnet_l(in_channels=3, out_channels=3, width=64):
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]
    print("MGDSRNET...")
    return MGDSRNet(in_channels, out_channels, width=width,
                  middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks,
                  dec_blk_nums=dec_blks,
                  shared_b=True)

if __name__ == '__main__':
    from tools import mutils

    x = torch.ones(1, 3, 256, 256).cuda()
    feats = [
        torch.ones(1, 64, 128, 128).cuda(),
        torch.ones(1, 192, 64, 64).cuda(),
        torch.ones(1, 384, 32, 32).cuda(),
        torch.ones(1, 768, 16, 16).cuda(),
        torch.ones(1, 2560, 8, 8).cuda(),
    ]
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    model = dsrnet_l(3, 3).cuda()
    mutils.count_parameters(model)
    mutils.count_parameters(model.intro)

    out_l, out_r = model(x, feats)
    print(out_l.shape, out_r.shape)

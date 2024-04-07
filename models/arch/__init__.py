import torch

from models.arch.dsrnet import DSRNet, MDSRNet, MXDSRNet, MGDSRNet, MG2DSRNet


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

def mdsrnet_l(in_channels=3, out_channels=3, width=64):
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    return MDSRNet(in_channels, out_channels, width=width,
                  middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks,
                  dec_blk_nums=dec_blks,
                  shared_b=True)

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

def mg2dsrnet_l(in_channels=3, out_channels=3, width=64):
    enc_blks = [2, 2, 2, 3]
    middle_blk_num = 2
    dec_blks = [2, 2, 2, 2]
    print("MG2DSRNET... 352")
    return MG2DSRNet(in_channels, out_channels, width=width,
                  middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks,
                  dec_blk_nums=dec_blks,
                  shared_b=True)

def mdsrnet_m(in_channels=3, out_channels=3, width=64):
    enc_blks = [2, 2, 2, 2]
    middle_blk_num = 4
    dec_blks = [2, 2, 2, 2]

    return MDSRNet(in_channels, out_channels, width=width,
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

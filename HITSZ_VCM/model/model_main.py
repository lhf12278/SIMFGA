import copy
import torch
import torch.nn as nn
from torch.nn import init
from model.resnet import resnet50, resnet18
from model.clip_model import Transformer



class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)





class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.layer4 = copy.deepcopy(self.base.layer4)

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        # t_x = self.layer4(x)
        x = self.base.layer4(x)
        return x


def conv1x1(conv, x):
    x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
    x = conv(x)
    x = x.squeeze()
    return x


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

# local feature
# class SpatialFeature(nn.Module):
#     def __init__(self,args, num_stripes, embed_dim, seq_len, class_num):
#         super(SpatialFeature, self).__init__()
#         self.num_part = num_stripes
#         self.embed_dim = embed_dim
#         self.class_num = class_num
#         self.seq_len = seq_len
#         self.linear_layer = nn.Linear(self.embed_dim * 2, self.embed_dim)
#         for i in range(self.num_part):
#             name = 'classifier' + str(i)
#             setattr(self, name, nn.Linear(embed_dim, self.num_part))
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.TransLocal = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)
#         self.TransGlobal = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         stripe_h = int(H/self.num_part)
#         local_feat = []
#         Avglocal_feat = []
#         feat_list = []
#         weight_feat = []
#         for i in range(self.num_part):
#             part_feat = x[:, :, i * stripe_h:(i + 1) * stripe_h, :]
#             part_feat = self.avgpool(part_feat).squeeze()
#             part_feat = part_feat.view(part_feat.size(0) // self.seq_len, self.seq_len, -1)
#             local_feat.append(part_feat)
#             Avglocal_feat.append(torch.mean(part_feat, dim=1))
#             part_feat = part_feat.permute(1,0,2)
#             part_attention = part_feat@Avglocal_feat[0].transpose(-2, -1)
#             part_attention = part_attention.softmax(dim=-1)
#             att_feat = (part_attention @ part_feat).permute(1,0,2)
#             att_feat = torch.sum(att_feat, dim=1)
#             weight_feat.append(att_feat)
#
#         part1 = self.TransLocal(local_feat[0])
#         part1 = torch.mean(part1,dim=1)
#         part1 = torch.cat((part1,weight_feat[0]), dim=-1)
#         part1 = self.linear_layer(part1)
#
#         part2 = self.TransLocal(local_feat[1])
#         part2 = torch.mean(part2, dim=1)
#         part2 = torch.cat((part2, weight_feat[1]), dim=-1)
#         part2 = self.linear_layer(part2)
#
#         part3 = self.TransLocal(local_feat[2])
#         part3 = torch.mean(part3, dim=1)
#         part3 = torch.cat((part3, weight_feat[2]), dim=-1)
#         part3 = self.linear_layer(part3)
#
#         feat_list.append(part1)
#         feat_list.append(part2)
#         feat_list.append(part3)
#         global_feat = torch.stack([part1,part2,part3], dim=1)
#         global_feat = self.TransGlobal(global_feat)
#         global_feat = torch.mean(global_feat,dim=1)
#
#         logits_list = []
#         for i in range(self.num_part):
#             classifier_i = getattr(self, 'classifier' + str(i))
#             logits_i = classifier_i(feat_list[i])
#             logits_list.append(logits_i)#(Batch, 3)
#
#         return global_feat, logits_list
# local feature
class SpatialFeature(nn.Module):
    def __init__(self,args, num_stripes, embed_dim, seq_len, class_num):
        super(SpatialFeature, self).__init__()
        self.num_part = num_stripes
        self.embed_dim = embed_dim
        self.class_num = class_num
        self.seq_len = seq_len
        self.linear_layer = nn.Linear(self.embed_dim * 2, self.embed_dim)
        for i in range(self.num_part):
            name = 'classifier' + str(i)
            setattr(self, name, nn.Linear(embed_dim, self.num_part))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.TransLocal = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)
        self.TransGlobal = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)

    def forward(self, x):
        B, C, H, W = x.shape
        stripe_h = int(H/self.num_part)
        local_feat = []
        Avglocal_feat = []
        feat_list = []
        weight_feat = []
        for i in range(self.num_part):
            part_feat = x[:, :, i * stripe_h:(i + 1) * stripe_h, :]
            part_feat = self.avgpool(part_feat).squeeze()
            part_feat = part_feat.view(part_feat.size(0) // self.seq_len, self.seq_len, -1)
            local_feat.append(part_feat)
            Avglocal_feat.append(torch.mean(part_feat, dim=1))
            part_feat = part_feat.permute(1,0,2)
            part_attention = torch.mul(part_feat, Avglocal_feat[i])
            part_attention = part_attention.softmax(dim=-1)
            att_feat = torch.mul(part_attention, part_feat)
            att_feat = torch.sum(att_feat, dim=0)
            weight_feat.append(att_feat)

        part1 = self.TransLocal(local_feat[0])
        part1 = torch.mean(part1,dim=1)
        #cat
        # part1 = torch.cat((part1,weight_feat[0]), dim=-1)
        # part1 = self.linear_layer(part1)
        #加
        part1 = part1 + weight_feat[0]

        part2 = self.TransLocal(local_feat[1])
        part2 = torch.mean(part2, dim=1)
        #cat
        # part2 = torch.cat((part2, weight_feat[1]), dim=-1)
        # part2 = self.linear_layer(part2)
        #加
        part2 = part2 + weight_feat[1]

        part3 = self.TransLocal(local_feat[2])
        part3 = torch.mean(part3, dim=1)
        #cat
        # part3 = torch.cat((part3, weight_feat[2]), dim=-1)
        # part3 = self.linear_layer(part3)
        # 加
        part3 = part3 + weight_feat[2]

        feat_list.append(part1)
        feat_list.append(part2)
        feat_list.append(part3)
        global_feat = torch.stack([part1,part2,part3], dim=1)
        global_feat = self.TransGlobal(global_feat)
        global_feat = torch.mean(global_feat,dim=1)

        logits_list = []
        for i in range(self.num_part):
            classifier_i = getattr(self, 'classifier' + str(i))
            logits_i = classifier_i(feat_list[i])
            logits_list.append(logits_i)#(Batch, 3)

        return global_feat, logits_list

#Fine-grained feature mining
class FFM(nn.Module):
    def __init__(self, num_stripes=6, pool_dim=2048, class_num=500):
        super(FFM, self).__init__()
        self.num_stripes = num_stripes
        self.embed_dim = pool_dim
        # classifier
        for i in range(self.num_stripes):
            name = 'classifier' + str(i)
            setattr(self, name, nn.Linear(pool_dim, class_num))
        self.local_conv_list = nn.ModuleList()
        for _ in range(self.num_stripes):
            conv = nn.Conv2d(pool_dim, pool_dim, 1)
            conv.apply(weights_init_kaiming)
            self.local_conv_list.append(nn.Sequential(
                conv,
                nn.BatchNorm2d(pool_dim),
                nn.ReLU(inplace=True)
            ))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, seq_len):
        b, c, h, w = x.shape
        t = seq_len
        #PCB
        feat_part = x
        b = feat_part.shape[0] // seq_len
        assert feat_part.size(2) % self.num_stripes == 0
        stripe_h = int(feat_part.size(2) / self.num_stripes)
        local_featpart_list = []
        local_list = []
        for i in range(self.num_stripes):
            local_feat = feat_part[:, :, i * stripe_h:(i + 1) * stripe_h, :]
            local_feat = self.avgpool(local_feat).squeeze()
            local_feat = self.local_conv_list[i](local_feat.view(local_feat.size(0), local_feat.size(1), 1, 1))
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat = local_feat.view(local_feat.size(0) // t, t, -1)
            local_feat = local_feat.permute(1, 0, 2)  # [8,2048,6]
            local_featpart_list.append(local_feat)
            local_list.append(torch.mean(local_featpart_list[i], dim=0))

        logits_list = []
        for i in range(self.num_stripes):
            classifier_i = getattr(self, 'classifier' + str(i))
            logits_i = classifier_i(local_list[i])
            logits_list.append(logits_i)

        return logits_list



class embed_net(nn.Module):
    def __init__(self, args, class_num, drop=0.2, no_local='on', gm_pool='on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        pool_dim = 2048
        self.embed_dim = pool_dim
        self.dropout = drop
        self.non_local = no_local
        self.gm_pool = gm_pool

        if self.non_local == 'on':
            layers = [3, 4, 6, 3]
            non_layers = [0, 2, 3, 0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        self.l2norm = Normalize(2)
        self.bottleneck0 = nn.BatchNorm1d(pool_dim)
        self.bottleneck0.bias.requires_grad_(False)  # no shift
        self.bottleneck0.apply(weights_init_kaiming)
        self.bottleneck1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck1.bias.requires_grad_(False)
        self.bottleneck1.apply(weights_init_kaiming)
        self.bottleneck2 = nn.BatchNorm1d(pool_dim)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)
        self.bottleneck3 = nn.BatchNorm1d(pool_dim)
        self.bottleneck3.bias.requires_grad_(False)
        self.bottleneck3.apply(weights_init_kaiming)
        self.bottleneck4 = nn.BatchNorm1d(pool_dim)
        self.bottleneck4.bias.requires_grad_(False)
        self.bottleneck4.apply(weights_init_kaiming)

        self.classifier0 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier3 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier4 = nn.Linear(pool_dim, class_num, bias=False)


        self.classifier0.apply(weights_init_classifier)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)
        self.classifier3.apply(weights_init_classifier)
        self.classifier4.apply(weights_init_classifier)

        self.fc = nn.Linear(pool_dim * 2, pool_dim)
        nn.init.normal_(self.fc.weight.data, std=0.001)
        nn.init.constant_(self.fc.bias.data, val=0.0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.PCB = FFM()

        self.cat_predictor = nn.Sequential(
            nn.Linear(pool_dim*2, pool_dim),
            nn.BatchNorm1d(pool_dim),
            nn.ReLU(),
            # nn.Linear(pool_dim, class_num),
        )
        # self.cat_predictor.apply(weights_init_kaiming)

        self.encoder = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)
        self.decoder = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)

        self.SpatialFeature = SpatialFeature(args, num_stripes=3, embed_dim=pool_dim,  seq_len=6, class_num=class_num)


    def forward(self, x1, x2, modal=0, seq_len=6):
        b, c, h, w = x1.size()
        t = seq_len
        x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
        x2 = x2.view(int(b * seq_len), int(c / seq_len), h, w)

        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)
        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0 : self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)#(192,256,72,36)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)#(192,512,36,18)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)#(192,1024,18,9)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)#(192,2048,18,9)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1

        else:
            x = self.base_resnet(x)

        if self.gm_pool == 'on':
            b, c, h, w = x.shape
            x_ = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x_ ** p, dim=-1) + 1e-12) ** (1 / p)
            x_pool = x_pool.view(x_pool.size(0) // t, t, -1)

            spatial_feat, logits_list1 = self.SpatialFeature(x)
            logits_list = self.PCB(x, t)
            feat = torch.mean(x_pool, dim=1)
            # local_feat = local_feat.permute(0, 2, 1)
            # TAP
            # local_feat = F.avg_pool1d(local_feat, t)
            # local_feat = local_feat.squeeze(-1)
            # globle_feat = local_feat + x_pool

            if self.training:
                en_feature = self.encoder(x_pool)
                en_feature = torch.mean(en_feature, dim=1).squeeze()
                en_rgb, en_ir = en_feature.chunk(2, 0)
                # 拼接
                feat1 = torch.cat((en_rgb, en_ir), dim=1)
                # feat1 = self.fc(feat1)
                feat1 = self.cat_predictor(feat1)
                # 加和
                # feat1 = feat1_ir + feat1_rgb

                en_norm1d = self.bottleneck0(en_feature)
                feat1_norm1d = self.bottleneck1(feat1)

            de_feature = self.decoder(x_pool)
            de_feature = torch.mean(de_feature, dim=1).squeeze()
            feat2_norm1d = self.bottleneck2(de_feature)
            feat_norm1d = self.bottleneck3(feat)
            spatial_feat_norm1d = self.bottleneck4(spatial_feat)



        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

            spatial_feat, logits_list1 = self.SpatialFeature(x)

            # logits_list = self.PCB(x, x_pool, t)
            feat = torch.mean(x_pool, dim=1)
            # local_feat = local_feat.permute(0, 2, 1)
            # TAP
            # local_feat = F.avg_pool1d(local_feat, t)
            # globle_feat = globle_feat.squeeze(-1)
            # globle_feat = local_feat + x_pool

            if self.training:
                en_feature = self.encoder(x_pool)
                en_feature = torch.mean(en_feature, dim=1).squeeze()
                en_rgb, en_ir = en_feature.chunk(2, 0)
                # 拼接
                feat1 = torch.cat((en_rgb, en_ir), dim=1)
                # feat1 = self.fc(feat1)
                feat1 = self.cat_predictor(feat1)
                # 加和
                # feat1 = feat1_ir + feat1_rgb

                en_norm1d = self.bottleneck3(en_feature)
                feat1_norm1d = self.bottleneck4(feat1)

            de_feature = self.decoder(x_pool)
            de_feature = torch.mean(de_feature, dim=1).squeeze()
            feat2_norm1d = self.bottleneck5(de_feature)
            feat_norm1d = self.bottleneck3(feat)
            spatial_feat_norm1d = self.bottleneck4(spatial_feat)


        if self.training:
            return feat, self.classifier0(feat_norm1d),logits_list, en_feature, self.classifier1(en_norm1d), feat1, self.classifier2(feat1_norm1d), de_feature, self.classifier3(feat2_norm1d), spatial_feat, self.classifier4(spatial_feat_norm1d), logits_list1
        else:
            return self.l2norm(feat2_norm1d + spatial_feat_norm1d)
# if __name__ == '__main__':
#
#     net = embed_net(args, n_class, no_local='on', gm_pool='on', arch=args.arch)
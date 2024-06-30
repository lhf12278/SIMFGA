import numpy as np
import os
import os.path as osp
import shutil
import sys
from PIL import Image, ImageOps, ImageDraw



def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Successfully make dirs: {}'.format(dir))
    else:
        print('Existed dirs: {}'.format(dir))


def visualize_ranked_results(distmat, dataset, save_dir='/mnt/data/user/zzg/New_ReID/model_17/visualize', topk=10, sort='descend', mode='all', only_show=None):
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    query, gallery = dataset
    assert num_q == len(query[1])
    assert num_g == len(gallery[1])
    assert sort in ['descend', 'ascend']
    assert mode in ['intra-camera', 'inter-camera', 'all']

    if sort is 'ascend':
        indices = np.argsort(distmat, axis=1)
    elif sort is 'descend':
        indices = np.argsort(distmat, axis=1)[:, ::-1]#从大到小索引

    for i in range(0, 12, 1):#!!!!!!!!!!!!!!!!!!
        Tdir = '{}'.format(i)
        path = os.path.join(save_dir, Tdir)
        make_dirs(path)


    image_ge = Image.new('RGB', (10, 132), 'white')
    image_ge2 = Image.new('RGB', (2, 132), 'white')


    def cat_imgs_to(image_list, hit_list, text_list, target_dir):
        images = []
        for img, hit, text in zip(image_list, hit_list, text_list):
            img = Image.open(img).resize((64, 128))
            d = ImageDraw.Draw(img)
            d.text((3, 1), "{:.3}".format(text), fill=(255, 255, 0))
            if hit:
                img = ImageOps.expand(img, border=2, fill='green')
            else:
                img = ImageOps.expand(img, border=2, fill='red')
            img = np.concatenate((img, image_ge2), 1)
            img = Image.fromarray(img)
            images.append(img)

        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        new_im.save(target_dir)


    counts = 0


    for q_idx in range(num_q):
        image_list = []
        hit_list = []
        text_list = []

        qimg_path = query[0][q_idx]
        qpid = query[1][q_idx]
        id_dir = '{}'.format(qpid)#!!!!!!!!!!!!!!!!

        image_list.append(qimg_path)
        hit_list.append(True)
        text_list.append(1.0)

        if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
            qdir = osp.join(save_dir, osp.basename(qimg_path[0]))
        else:
            temp = '{}.jpg'.format(q_idx)

        rank_idx = 1
        for ii, g_idx in enumerate(indices[q_idx, :]):
            gimg_path = gallery[0][g_idx]
            gpid = gallery[1][g_idx]

            image_list.append(gimg_path)
            hit_list.append(qpid == gpid)
            text_list.append(distmat[q_idx, g_idx])
            rank_idx += 1
            if rank_idx > topk:
                break

        counts += 1

        if hit_list == [True,True,True,True,True,True,True,True,True,True,True]:
            dir = os.path.join(save_dir, '10', id_dir)
            img = os.path.join(dir, temp)
            isExists = os.path.exists(dir)
            if not isExists:
                os.makedirs(dir)
            cat_imgs_to(image_list, hit_list, text_list, img)
        elif hit_list == [True,True,True,True,True,True,True,True,True,True,False]:
            dir = os.path.join(save_dir, '9', id_dir)
            img = os.path.join(dir, temp)
            isExists = os.path.exists(dir)
            if not isExists:
                os.makedirs(dir)
            cat_imgs_to(image_list, hit_list, text_list, img)
        elif hit_list == [True,True,True,True,True,True,True,True,True,False,False]:
            dir = os.path.join(save_dir, '8', id_dir)
            img = os.path.join(dir, temp)
            isExists = os.path.exists(dir)
            if not isExists:
                os.makedirs(dir)
            cat_imgs_to(image_list, hit_list, text_list, img)
        elif hit_list == [True,True,True,True,True,True,True,True,False,False,False]:
            dir = os.path.join(save_dir, '7', id_dir)
            img = os.path.join(dir, temp)
            isExists = os.path.exists(dir)
            if not isExists:
                os.makedirs(dir)
            cat_imgs_to(image_list, hit_list, text_list, img)
        elif hit_list == [True,True,True,True,True,True,True,False,False,False,False]:
            dir = os.path.join(save_dir, '6', id_dir)
            img = os.path.join(dir, temp)
            isExists = os.path.exists(dir)
            if not isExists:
                os.makedirs(dir)
            cat_imgs_to(image_list, hit_list, text_list, img)
        elif hit_list == [True,True,True,True,True,True,False,False,False,False,False]:
            dir = os.path.join(save_dir, '5', id_dir)
            img = os.path.join(dir, temp)
            isExists = os.path.exists(dir)
            if not isExists:
                os.makedirs(dir)
            cat_imgs_to(image_list, hit_list, text_list, img)
        elif hit_list == [True,True,True,True,True,False,False,False,False,False,False]:
            dir = os.path.join(save_dir, '4', id_dir)
            img = os.path.join(dir, temp)
            isExists = os.path.exists(dir)
            if not isExists:
                os.makedirs(dir)
            cat_imgs_to(image_list, hit_list, text_list, img)
        elif hit_list == [True,True,True,True,False,False,False,False,False,False,False]:
            dir = os.path.join(save_dir, '3', id_dir)
            img = os.path.join(dir, temp)
            isExists = os.path.exists(dir)
            if not isExists:
                os.makedirs(dir)
            cat_imgs_to(image_list, hit_list, text_list, img)
        elif hit_list == [True,True,True,False,False,False,False,False,False,False,False]:
            dir = os.path.join(save_dir, '2', id_dir)
            img = os.path.join(dir, temp)
            isExists = os.path.exists(dir)
            if not isExists:
                os.makedirs(dir)
            cat_imgs_to(image_list, hit_list, text_list, img)
        elif hit_list == [True,True,False,False,False,False,False,False,False,False,False]:
            dir = os.path.join(save_dir, '1', id_dir)
            img = os.path.join(dir, temp)
            isExists = os.path.exists(dir)
            if not isExists:
                os.makedirs(dir)
            cat_imgs_to(image_list, hit_list, text_list, img)

        else:
            continue
        print(counts, img)
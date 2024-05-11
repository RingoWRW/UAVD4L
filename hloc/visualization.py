from matplotlib import cm
import random
import numpy as np
import pickle
import pycolmap
from .utils.viz import (
        plot_images, plot_keypoints, plot_matches, cm_RdGn, add_text, save_plot)
from .utils.io import read_image, get_keypoints, get_matches, get_matches_loftr
from .utils.parsers import parse_retrieval, names_to_pair
from .match_features import find_unique_new_pairs
def visualize_sfm_2d(reconstruction, image_dir, color_by='visibility',
                     selected=[], n=1, seed=0, dpi=75):
    assert image_dir.exists()
    if not isinstance(reconstruction, pycolmap.Reconstruction):
        reconstruction = pycolmap.Reconstruction(reconstruction)

    if not selected:
        image_ids = reconstruction.reg_image_ids()
        selected = random.Random(seed).sample(
                image_ids, min(n, len(image_ids)))

    for i in selected:
        image = reconstruction.images[i]
        keypoints = np.array([p.xy for p in image.points2D])
        visible = np.array([p.has_point3D() for p in image.points2D])

        if color_by == 'visibility':
            color = [(0, 0, 1) if v else (1, 0, 0) for v in visible]
            text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
        elif color_by == 'track_length':
            tl = np.array([reconstruction.points3D[p.point3D_id].track.length()
                           if p.has_point3D() else 1 for p in image.points2D])
            max_, med_ = np.max(tl), np.median(tl[tl > 1])
            tl = np.log(tl)
            color = cm.jet(tl / tl.max()).tolist()
            text = f'max/median track length: {max_}/{med_}'
        elif color_by == 'depth':
            p3ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
            z = np.array([image.transform_to_image(
                reconstruction.points3D[j].xyz)[-1] for j in p3ids])
            z -= z.min()
            color = cm.jet(z / np.percentile(z, 99.9))
            text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
            keypoints = keypoints[visible]
        else:
            raise NotImplementedError(f'Coloring not implemented: {color_by}.')

        name = image.name
        plot_images([read_image(image_dir / name)], dpi=dpi)
        plot_keypoints([keypoints], colors=[color], ps=4)
        add_text(0, text)
        add_text(0, name, pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')


def visualize_loc(results, image_dir, reconstruction=None,
                  selected=[], n=1, seed=0, prefix=None, **kwargs):
    assert image_dir.exists()

    with open(str(results)+'_logs.pkl', 'rb') as f:
        logs = pickle.load(f)

    if not selected:
        queries = list(logs['loc'].keys())
        if prefix:
            queries = [q for q in queries if q.startswith(prefix)]
        selected = random.Random(seed).sample(queries, min(n, len(queries)))

    if reconstruction is not None:
        if not isinstance(reconstruction, pycolmap.Reconstruction):
            reconstruction = pycolmap.Reconstruction(reconstruction)

    for qname in selected:
        loc = logs['loc'][qname]
        visualize_loc_from_log(image_dir, qname, loc, reconstruction, **kwargs)


def visualize_loc_from_log(image_dir, query_name, loc, reconstruction=None,
                           top_k_db=2, dpi=75):

    q_image = read_image(image_dir / query_name)
    if loc.get('covisibility_clustering', False):
        # select the first, largest cluster if the localization failed
        loc = loc['log_clusters'][loc['best_cluster'] or 0]

    inliers = np.array(loc['PnP_ret']['inliers'])
    mkp_q = loc['keypoints_query']
    n = len(loc['db'])
    if reconstruction is not None:
        # for each pair of query keypoint and its matched 3D point,
        # we need to find its corresponding keypoint in each database image
        # that observes it. We also count the number of inliers in each.
        kp_idxs, kp_to_3D_to_db = loc['keypoint_index_to_db']
        counts = np.zeros(n)
        dbs_kp_q_db = [[] for _ in range(n)]
        inliers_dbs = [[] for _ in range(n)]
        for i, (inl, (p3D_id, db_idxs)) in enumerate(zip(inliers,
                                                         kp_to_3D_to_db)):
            track = reconstruction.points3D[p3D_id].track
            track = {el.image_id: el.point2D_idx for el in track.elements}
            for db_idx in db_idxs:
                counts[db_idx] += inl
                kp_db = track[loc['db'][db_idx]]
                dbs_kp_q_db[db_idx].append((i, kp_db))
                inliers_dbs[db_idx].append(inl)
    else:
        # for inloc the database keypoints are already in the logs
        assert 'keypoints_db' in loc
        assert 'indices_db' in loc
        counts = np.array([
            np.sum(loc['indices_db'][inliers] == i) for i in range(n)])

    # display the database images with the most inlier matches
    db_sort = np.argsort(-counts)
    for db_idx in db_sort[:top_k_db]:
        if reconstruction is not None:
            db = reconstruction.images[loc['db'][db_idx]]
            db_name = db.name
            db_kp_q_db = np.array(dbs_kp_q_db[db_idx])
            kp_q = mkp_q[db_kp_q_db[:, 0]]
            kp_db = np.array([db.points2D[i].xy for i in db_kp_q_db[:, 1]])
            inliers_db = inliers_dbs[db_idx]
        else:
            db_name = loc['db'][db_idx]
            kp_q = mkp_q[loc['indices_db'] == db_idx]
            kp_db = loc['keypoints_db'][loc['indices_db'] == db_idx]
            inliers_db = inliers[loc['indices_db'] == db_idx]

        db_image = read_image(image_dir / db_name)
        color = cm_RdGn(inliers_db).tolist()
        text = f'inliers: {sum(inliers_db)}/{len(inliers_db)}'

        plot_images([q_image, db_image], dpi=dpi)
        plot_matches(kp_q, kp_db, color, a=0.1)
        add_text(0, text)
        opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')
        add_text(0, query_name, **opts)
        add_text(1, db_name, **opts)


def visualize_from_h5(db_dir, query_dir, pairs_path, features_db, feature_query, matches_path, outputs, top_k_db=2, dpi=175):

    pairs_all = parse_retrieval(pairs_path)
    
    pairs_all = [(q, r) for q, rs in pairs_all.items() for r in rs]
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = sorted(list(pairs))

    for i in range(len(pairs)):
        # import pdb;pdb.set_trace()
        query_name, db_name = pairs[i][0], pairs[i][1]
        kp_db = get_keypoints(features_db, db_name)
        kp_q = get_keypoints(feature_query, query_name)
        matches, scores = get_matches(matches_path, db_name, query_name)
        mkpts_db = kp_db[matches[:,0]]
        mkpts_q = kp_q[matches[:,1]]
        color = cm_RdGn(scores).tolist()
        text = f'inliers: {mkpts_db.shape[0]}'
        q_image = read_image(query_dir / query_name)
        db_image = read_image(db_dir / db_name)
        plot_images([db_image, q_image], dpi=dpi)
        plot_matches(mkpts_db, mkpts_q, a=0.5)
    
        
        add_text(0, text)
        opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')
        add_text(0, db_name, **opts)
        add_text(1, query_name, **opts)
        save_path = outputs / (query_name.split('.')[0] + str(i%3) + '.png')
        save_plot(save_path)
        print("save matches!")
        # import pdb;pdb.set_trace()
        
    
def visualize_loftr_from_h5(db_dir, query_dir, pairs_path, matches, outputs,  top_k_db=2, dpi=175):
    
    pairs_all = parse_retrieval(pairs_path)
    pairs_all = [(q, r) for q, rs in pairs_all.items() for r in rs]
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = sorted(list(pairs))
    
    # display the database images with the most inlier matches
    for i in range(len(pairs)):
        query_name, db_name = pairs[i][0], pairs[i][1]
        mkpts_q, mkpts_db, scores = get_matches_loftr(matches, query_name, db_name)
        color = cm_RdGn(scores).tolist()
        text = f'inliers: {mkpts_db.shape[0]}'
        q_image = read_image(query_dir / query_name)
        db_image = read_image(db_dir / db_name)
        plot_images([db_image, q_image], dpi=dpi)
        
        plot_matches(mkpts_db, mkpts_q, a=0.1)
        
        # plot_images([db_image], dpi=dpi)
        # plot_keypoints([kp_db])
        
        add_text(0, text)
        opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')
        add_text(0, db_name, **opts)
        add_text(1, query_name, **opts)
        save_path = outputs / (query_name.split('.')[0] + '.png')
        save_plot(save_path)
        print("save matches!")


def visualiza_retrieval_from_txt(db_dir, query_dir, pairs_path, outputs, dpi=175):

    pairs_all = parse_retrieval(pairs_path)
    pairs_all = [(q, r) for q, rs in pairs_all.items() for r in rs]
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = sorted(list(pairs))

    for i in range(len(pairs)):
        query_name, db_name = pairs[i][0], pairs[i][1]
        q_image = read_image(query_dir / query_name)
        db_image = read_image(db_dir / db_name)
        plot_images([db_image, q_image], dpi=dpi)
        opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')
        add_text(0, db_name, **opts)
        add_text(1, query_name, **opts)
        save_path = outputs / (query_name.split('.')[0] + str(i%3) + '.png')
        # save_path = outputs / (query_name.split('.')[0] + '.png')
        save_plot(save_path)
        print("save matches!")
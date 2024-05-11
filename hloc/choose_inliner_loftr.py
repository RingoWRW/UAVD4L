from .utils.parsers import parse_retrieval, names_to_pair
from .utils.parsers import parse_image_lists
from pathlib import Path
from . import logger
from .utils.io import get_matches_loftr, get_matches,get_keypoints
from tqdm import tqdm
import numpy as np

def main(queries: Path,
         retrieval: Path,
         matches_path: Path,
         results:Path,
         num_loc:int):

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)
    # choose index
    all_index = []
    # import pdb;pdb.set_trace()
    qname_list = []
    dbname_list = []
    for qname, query_camera in tqdm(queries):
        point_list = []
        if qname not in retrieval_dict:
            logger.warning(f'No images retrieved for query image {qname}. Skipping...')
            continue
        db_names = retrieval_dict[qname]
        db_name_list = db_names[:num_loc]#!top3
        #get 2D correspondences
        # import pdb;pdb.set_trace()
        for db_index in range(num_loc):
            # matches, _ = get_matches(matches_path, qname, db_name_list[db_index])
            # import pdb;pdb.set_trace()
            # features_query_path = output / f'{features_query}.h5'
            # points2D_q = get_keypoints(features_query_path, qname)
            # points2D_q += 0.5 
            # points2D_q = points2D_q[matches[:, 0]]
            points2D_q, points2D_db, _ = get_matches_loftr(matches_path, qname, db_name_list[db_index])
            point_list.append(len(points2D_q))
        max_index = np.argmax(point_list)
        all_index.append(max_index)
        qname_list.append(qname)
        # import pdb;pdb.set_trace()
        dbname_list.append(db_name_list[max_index])

    with open(results, 'w') as fr:
        # import pdb;pdb.set_trace()
        for i in range(len(qname_list)):
            info = qname_list[i] + ' ' + dbname_list[i] + '\n'
            fr.write(info)

    return all_index
import os
import sys
import numpy as np
import anndata
import scanpy as sc
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from scipy.sparse import issparse, csr_matrix
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd
# import stlearn
from _compat import Literal
import scanpy
import scipy
import matplotlib.pyplot as plt

_QUALITY = Literal["fulres", "hires", "lowres"]
_background = ["black", "white"]


def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])


def read_10X_Visium(path,
                    data_name=None,
                    genome=None,
                    count_file='filtered_feature_bc_matrix.h5',
                    library_id=None,
                    load_images=True,
                    quality='hires',
                    image_path=None):
    adata = sc.read_visium(
        path,
        genome=genome,
        count_file=count_file,
        library_id=library_id,
        load_images=load_images,
    )
    adata.var_names_make_unique()

    # print(adata.obs)
    # time.sleep(3000)
    # print(type(adata))
    # ahn=adata.X.todense()
    # # print(ahn.size)
    # print("hello")
    # print(ahn.shape)
    # print(adata)
    # print(type(ahn))
    # print(adata.X.size)
    # print(adata.X.shape)
    # print(type(adata.X))

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
        print("library_id:")
        #
        # library_id:
        # JBO22/151673#这是从源文件中读出来的
        print(library_id)
        print(type(library_id))

    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        impa = os.path.join(path, "spatial", data_name + image_path)
        Image.MAX_IMAGE_PIXELS = 200000000
        img = plt.imread(impa, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_" +
                                                                 quality +
                                                                 "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]

    # print(adata.obsm["spatial"])
    # print(adata.obs["imagecol"])
    # print(adata.obs["imagerow"])
    # time.sleep(3000)

    adata.uns["spatial"][library_id]["use_quality"] = quality

    # region #todo读取gt标签
    anno_path = "./datasets/HumanPilot-master/Analysis/Layer_Guesses/First_Round/metadata_" + data_name + ".csv"
    # labels_df = pd.read_csv(anno_path)
    labels_df = pd.read_csv(anno_path, header=0, index_col=0)

    labels_df.columns = ['Ground_Truth']
    labels_df = labels_df.reindex(adata.obs_names)
    adata.obs['Ground_Truth'] = labels_df.loc[adata.obs_names, 'Ground_Truth']
    # print(labels_df)
    # labels_df = pd.read_csv("labels.csv")
    # assert labels_df["spot_name"].is_unique, "Error: spot_name在labels中不唯一"
    # # 到这里都没问题
    # labels_df = labels_df.set_index("spot_name")

    # 确保adata对象已加载，且索引为spot_name
    # 获取索引交集
    # common_spots = adata.obs.index.intersection(labels_df.index)

    # # 创建子集（保留原始所有数据）
    # cdata = adata[common_spots].copy()

    # # 合并layer信息（通过索引自动对齐）
    # cdata.obs = cdata.obs.join(labels_df[["layer"]])

    # # 结果验证
    # print(f"原始细胞数: {adata.n_obs}, 新细胞数: {cdata.n_obs}")
    # print(cdata.obs[["layer"]].head())  # 显示合并后的前5行

    # adata = cdata
    # endregion 借鉴读取gt标签

    return adata


def read_SlideSeq(
    path,
    library_id=None,
    scale=None,
    quality="hires",
    spot_diameter_fullres=50,
    background_color="white",
):

    count = pd.read_csv(os.path.join(path, "count_matrix.count"))
    meta = pd.read_csv(os.path.join(path, "spatial.idx"))

    # adata = AnnData(count.iloc[:, 1:].set_index("gene").T)
    adata = AnnData(count.iloc[:, :].set_index("Row").T)

    adata.var["ENSEMBL"] = count["Row"].values

    # adata.var["ENSEMBL"] = count["ENSEMBL"].values

    # adata.obs["index"] = meta["index"].values
    adata.obs["index"] = meta["barcodes"].values

    if scale == None:
        max_coor = np.max(meta[["xcoord", "ycoord"]].values)
        scale = 2000 / max_coor

    adata.obs["imagecol"] = meta["xcoord"].values * scale
    adata.obs["imagerow"] = meta["ycoord"].values * scale

    # Create image
    max_size = np.max(
        [adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "Slide-seq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality +
                                                     "_scalef"] = scale

    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"] = spot_diameter_fullres
    adata.obsm["spatial"] = meta[["xcoord", "ycoord"]].values

    return adata


def read_merfish(
    path,
    library_id=None,
    scale=None,
    quality="hires",
    spot_diameter_fullres=50,
    background_color="white",
):

    counts = sc.read_csv(os.path.join(path, 'counts.csv')).transpose()
    locations = pd.read_excel(os.path.join(path, 'spatial.xlsx'), index_col=0)
    if locations.min().min() < 0:
        locations = locations + np.abs(locations.min().min()) + 100
    adata = counts[locations.index, :]
    adata.obsm["spatial"] = locations.to_numpy()

    if scale == None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    # Create image
    max_size = np.max(
        [adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "MERSEQ"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality +
                                                     "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"] = spot_diameter_fullres

    return adata


def read_seqfish(
    path,
    library_id=None,
    scale=1.0,
    quality="hires",
    field=0,
    spot_diameter_fullres=50,
    background_color="white",
):

    count = pd.read_table(os.path.join(path, 'counts.matrix'), header=None)
    spatial = pd.read_table(os.path.join(path, 'spatial.csv'), index_col=False)

    count = count.T
    count.columns = count.iloc[0]
    count = count.drop(count.index[0]).reset_index(drop=True)
    count = count[count["Field_of_View"] == field].drop(count.columns[[0, 1]],
                                                        axis=1)
    spatial = spatial[spatial["Field_of_View"] == field]

    # cells = set(count[''])
    # obs = pd.DataFrame(index=cells)
    adata = AnnData(count)

    if scale == None:
        max_coor = np.max(spatial[["X", "Y"]])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = spatial["X"].values * scale
    adata.obs["imagerow"] = spatial["Y"].values * scale

    adata.obsm["spatial"] = spatial[["X", "Y"]].values

    # Create image
    max_size = np.max(
        [adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "SeqFish"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality +
                                                     "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"] = spot_diameter_fullres

    return adata


def read_stereoSeq(
    path,
    bin_size=100,
    is_sparse=True,
    library_id=None,
    scale=None,
    quality="hires",
    spot_diameter_fullres=1,
    background_color="white",
):

    from scipy import sparse
    count = pd.read_csv(os.path.join(path, "count.txt"),
                        sep='\t',
                        comment='#',
                        header=0)
    count.dropna(inplace=True)
    if "MIDCounts" in count.columns:
        count.rename(columns={"MIDCounts": "UMICount"}, inplace=True)
    count['x1'] = (count['x'] / bin_size).astype(np.int32)
    count['y1'] = (count['y'] / bin_size).astype(np.int32)
    count['pos'] = count['x1'].astype(str) + "-" + count['y1'].astype(str)
    bin_data = count.groupby(['pos', 'geneID'])['UMICount'].sum()
    cells = set(x[0] for x in bin_data.index)
    genes = set(x[1] for x in bin_data.index)
    cellsdic = dict(zip(cells, range(0, len(cells))))
    genesdic = dict(zip(genes, range(0, len(genes))))
    rows = [cellsdic[x[0]] for x in bin_data.index]
    cols = [genesdic[x[1]] for x in bin_data.index]
    exp_matrix = sparse.csr_matrix((bin_data.values, (rows, cols))) if is_sparse else \
                 sparse.csr_matrix((bin_data.values, (rows, cols))).toarray()
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=genes)
    adata = AnnData(X=exp_matrix, obs=obs, var=var)
    pos = np.array(list(adata.obs.index.str.split('-', expand=True)),
                   dtype=np.int)
    adata.obsm['spatial'] = pos

    if scale == None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 20 / max_coor

    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    # Create image
    max_size = np.max(
        [adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "StereoSeq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality +
                                                     "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"] = spot_diameter_fullres

    return adata


def ReadOldST(
    count_matrix_file: Union[str, Path] = None,
    spatial_file: Union[str, Path] = None,
    image_file: Union[str, Path] = None,
    library_id: str = None,
    scale: float = 1.0,
    quality: str = "fulres",
    spot_diameter_fullres: float = 100,
    visium=False,
    dataname: str = None,
    num_cluster=3,
):
    """\
    Read Old Spatial Transcriptomics data
    Parameters
    ----------
    count_matrix_file
        Path to count matrix file.
    spatial_file
        Path to spatial location file.
    image_file
        Path to the tissue image file
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in anndata.obs['imagecol' & 'imagerow']
    spot_diameter_fullres
        Diameter of spot in full resolution
    Returns
    -------
    AnnData
    """

    # dataname="A"
    if library_id is None:
        library_id = dataname
    # library_id="shit"

    count_matrix_file = "./datasets/her2/count-matrices/ut_" + dataname + "1_stdata_filtered.tsv"

    adata = scanpy.read_text(count_matrix_file)
    print("AnnData: ")
    print(adata)  #AnnData object with n_obs × n_vars = 346 × 15045

    sc.pp.filter_genes(adata, min_cells=3)

    # sc.pp.filter_cells(adata, min_genes = 200)
    print(adata)
    # print(adata.X)
    # print(type(adata.X))
    coordinates_file = "./datasets/her2/spot-selections/" + dataname + "1_selection.tsv"
    # coordinates_file="./meta/A1_labeled_coordinates.tsv"
    # copy=True

    #parsing:
    new_coordinates = dict()
    with open(coordinates_file, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            assert len(tokens) >= 6 or len(tokens) == 4
            if tokens[
                    0] != "x":  #tokens[0]是 "x"的话表示是第一行，也就是说还在读行标题，排除以后下面才是正常的数据
                old_x = int(tokens[0])
                old_y = int(tokens[1])  #读x和y

                if len(tokens) >= 6:
                    pixel_x = float(tokens[4])
                    pixel_y = float(tokens[5])  #读pixel_x和y
                    new_coordinates[(old_x, old_y)] = (pixel_x, pixel_y
                                                       )  #建一个索引
                else:
                    raise ValueError(
                        "Error, output format is pixel coordinates but\n "
                        "the coordinates file only contains 4 columns\n")

    counts_table = adata.to_df()
    # Remove genes that have now a total count of zero
    # counts_table = counts_table.transpose()[counts_table.sum(axis=0) > 0].transpose() #筛掉部分不表达的基因
    # new_index_values = list()
    imgcol = []
    imgrow = []
    for index in counts_table.index:
        tokens = index.split("x")  #把index分割了一下，分成了x,y坐标（原本是x\times y的形式）
        x = int(tokens[0])
        y = int(tokens[1])
        try:
            new_x, new_y = new_coordinates[(x, y)]
            imgcol.append(new_x)
            imgrow.append(new_y)

            # new_index_values.append("{0}x{1}".format(new_x, new_y)) #这个list是像素上的坐标点
        except KeyError:
            counts_table.drop(index, inplace=True)

    # Assign the new indexes
    # counts_table.index = new_index_values

    adata = AnnData(counts_table)

    # print(adata)

    adata.var_names_make_unique()
    # print(adata)

    adata.obs["imagecol"] = imgcol
    adata.obs["imagerow"] = imgrow

    # result = [list(pair) for pair in zip(imgcol, imgrow)]
    # adata.obsm["spatial"] = np.array(result)

    # imgcol = []
    # imgrow = []
    adata.obsm["spatial"] = np.c_[imgcol, imgrow]  #也就是说最终用的坐标是在照片位置上的像素值
    print("adata with coordinates: ")
    print(adata)
    # print(adata.obsm['spatial'])
    # print(adata.obsm['spatial'].shape)
    # print((type(adata.obsm['spatial'])))
    # print(adata.obs['imagecol'])
    # print(type(adata.obs['imagecol']))
    # print(adata.obs['imagerow'])
    # print(type(adata.obs['imagerow']))

    #图像的部分
    imgpath = "./datasets/her2/images/HE/" + dataname + "1.jpg"

    if imgpath is not None and os.path.isfile(imgpath):
        try:
            # from PIL import Image
            Image.MAX_IMAGE_PIXELS = 200000000
            img = plt.imread(imgpath, 0)

            if visium:
                adata.uns["spatial"][library_id]["images"][quality] = img
            else:
                adata.uns["spatial"] = {}
                adata.uns["spatial"][library_id] = {}
                adata.uns["spatial"][library_id]["images"] = {}
                adata.uns["spatial"][library_id]["images"][quality] = img
                adata.uns["spatial"][library_id]["use_quality"] = quality
                adata.uns["spatial"][library_id]["scalefactors"] = {}
                adata.uns["spatial"][library_id]["scalefactors"][
                    "tissue_" + quality + "_scalef"] = scale
                adata.uns["spatial"][library_id]["scalefactors"][
                    "spot_diameter_fullres"] = spot_diameter_fullres
                adata.obsm["spatial"] = adata.obs[["imagecol",
                                                   "imagerow"]].values
                adata.obs[["imagecol",
                           "imagerow"]] = adata.obsm["spatial"] * scale

            print("Added tissue image to the object!")

        except:
            raise ValueError(f"""\
            {imgpath!r} does not end on a valid extension.
            """)
    else:
        raise ValueError(f"""\
        {imgpath!r} does not end on a valid extension.
        """)

    print("complete AnnData: ")
    print(adata)
    # AnnData object with n_obs × n_vars = 346 × 14532
    #     obs: 'imagecol', 'imagerow'
    #     uns: 'spatial'
    #     obsm: 'spatial'
    # print(adata.uns["spatial"])
    # print("another check")
    # print(adata.obsm["spatial"])
    # print(adata.obs["imagecol"])
    # print(adata.obs["imagerow"])
    # import random
    # n=len(adata.obs["imagecol"])
    # shit_list=[random.randint(0, 4) for _ in range(n)]

    # adata.obs["ground_truth"]=shit_list
    # scanpy.pl.spatial(adata, color='ground_truth',frameon=False,spot_size=150,show=True,img_key='fulres') #默认的img_key是hires和lowres,所以要把图片叫“fulres”的话，这里必须要指定img_key为fulres
    #读ground_truth
    # data_name = "A"
    anno_path = "./datasets/her2/meta/" + dataname + "1_labeled_coordinates.tsv"

    labels = dict()
    # label_dict = {"invasive": 0, "connective": 1, "immune": 3,"undetermined":3}#先不管这个小聚类
    if dataname=='D' or dataname=='E' or dataname=='F':
        if num_cluster == 2:
            label_dict = {
                "invasive": 0,
                "connective": 1,
                "immune": np.nan,
                "undetermined": np.nan
            }  #先不管这个小聚类
        elif num_cluster == 3:
            label_dict = {
                "invasive": 0,
                "connective": 1,
                "immune": 2,
                "undetermined": np.nan
            }
    elif dataname=='C':
        if num_cluster == 2:
            label_dict = {
                "invasive": 0,
                "connective": 1,
                "adipose": np.nan,
                "undetermined": np.nan
            }  #先不管这个小聚类
        elif num_cluster == 3:
            label_dict = {
                "invasive": 0,
                "connective": 1,
                "adipose": 2,
                "undetermined": np.nan
            }
    elif dataname=='B':
        
        # label_dict = {
        #     "invasive": 0,
        #     "connective": 1,
        #     "adipose": 2,
        #     "breast": 3,
        #     "undetermined": np.nan
        # }
        label_dict = {
            "invasive": 3,
            "connective": 0,
            "adipose": 1,
            "breast": 2,
            "undetermined": np.nan
        }
    #有5个细胞是undetermined

    with open(anno_path, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            assert len(tokens) >= 6 or len(tokens) == 4
            if tokens[0] != "Row.names":
                old_x = round(float(tokens[1]))
                old_y = round(float(tokens[2]))

                if len(tokens) >= 6:
                    clabel = tokens[5]
                    # print(clabel)
                    tmp = label_dict[clabel]
                    labels[(old_x, old_y)] = tmp
                    # print(tmp)

                else:
                    raise ValueError(
                        "Error, output format is pixel coordinates but\n "
                        "the coordinates file only contains 4 columns\n")

            # if tokens[0] != "x":
            #     old_x = int(tokens[0])
            #     old_y = int(tokens[1])

            #     if len(tokens) >= 6:
            #         pixel_x = float(tokens[4])
            #         pixel_y = float(tokens[5])
            #         new_coordinates[(old_x, old_y)] = (pixel_x, pixel_y)
            #     else:
            #         raise ValueError(
            #             "Error, output format is pixel coordinates but\n "
            #             "the coordinates file only contains 4 columns\n"
            #         )

    counts_table = adata.to_df()
    # new_index_values = list()

    gt_labels = []
    for index in counts_table.index:
        tokens = index.split("x")  #把index分割了一下，分成了x,y坐标（原本是x\times y的形式）
        x = int(tokens[0])
        y = int(tokens[1])
        try:
            mylabel = labels[(x, y)]
            gt_labels.append(mylabel)

            # new_index_values.append("{0}x{1}".format(new_x,
            #                                          new_y))  #这个list是像素上的坐标点
        except KeyError:
            counts_table.drop(index, inplace=True)

    adata.obs["Ground_Truth"] = gt_labels
    print(adata.obs["Ground_Truth"])
    print(adata)

    # adata.obs['Ground_Truth'] = adata.obs['Ground_Truth'].astype('int')
    # print(adata.obs["Ground_Truth"])

    # cdata = adata[adata.obs["Ground_Truth"] != 3].copy()
    # print("after")
    # print(cdata)

    return adata


def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print(
            "Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data."
        )
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp[0:num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred


def mclust_R(adata,
             num_cluster,
             modelNames='EEE',
             used_obsm='emb_pca',
             random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]),
                  num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata,
               n_clusters=7,
               radius=50,
               key='emb',
               method='mclust',
               start=0.1,
               end=3.0,
               increment=0.01,
               refinement=False):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1.
    end : float 
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.   
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=20, random_state=42)
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm['emb_pca'] = embedding

    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    # elif method == 'leiden':
    #    res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
    #    sc.tl.leiden(adata, random_state=0, resolution=res)
    #    adata.obs['domain'] = adata.obs['leiden']
    # elif method == 'louvain':
    #    res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
    #    sc.tl.louvain(adata, random_state=0, resolution=res)
    #    adata.obs['domain'] = adata.obs['louvain']

    if refinement:
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['domain'] = new_type


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    #calculate distance
    position = adata.obsm['spatial']
    import ot
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    #adata.obs['label_refined'] = np.array(new_type)

    return new_type


def analyze_tensor(tensor):
    """分析张量的正负值比例"""
    # 计算各类元素数量
    total = tensor.numel()
    positive = (tensor > 0).sum().item()
    negative = (tensor < 0).sum().item()
    zero = (tensor == 0).sum().item()

    # 计算比例 (避免除零错误)
    positive_ratio = positive / total if total > 0 else 0
    negative_ratio = negative / total if total > 0 else 0
    zero_ratio = zero / total if total > 0 else 0

    # 计算正负值比例 (当负值存在时)
    pos_neg_ratio = positive / negative if negative > 0 else float('inf')

    return {
        "total_elements": total,
        "positive": positive,
        "negative": negative,
        "zeros": zero,
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio,
        "zero_ratio": zero_ratio,
        "positive_to_negative": pos_neg_ratio
    }


class ClusteringEarlyStopper:

    def __init__(self, patience=5, min_delta=0.001, max_epochs=100):
        """
        参数说明:
        - patience: 允许连续无改进的迭代次数
        - min_delta: 轮廓系数最小提升阈值（低于此值视为无改进）
        - max_epochs: 最大迭代轮次（防止无限循环）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = max_epochs
        self.best_score = -1  # 轮廓系数范围 [-1, 1]
        self.counter = 0
        self.epoch = 0
        self.best_model = None

    def should_stop(self, current_score, model):
        """
        判断是否应提前停止训练
        返回:
        - True/False: 是否停止
        """
        self.epoch += 1

        # 首次记录最佳模型
        if self.best_model is None:
            self.best_model = model
            self.best_score = current_score
            return False

        # 判断是否提升超过阈值
        if current_score - self.best_score > self.min_delta:
            self.best_score = current_score
            self.best_model = model
            self.counter = 0
        else:
            self.counter += 1

        # 停止条件检查
        if self.counter >= self.patience:
            print(f"Early stopping at epoch {self.epoch}")
            return True
        # if self.epoch >= self.max_epochs:
        #     print(f"Reached max epochs {self.max_epochs}")
        #     return True
        return False

import random
import itertools
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from tqdm.auto import tqdm
from scipy.spatial.qhull import QhullError
from scipy.spatial import ConvexHull
import tempfile
import zipfile
import os
import struct
import codecs


# taken from https://github.com/inferno-pytorch/inferno/blob/master/inferno/io/volumetric/volumetric_utils.py
def slidingwindowslices(shape, window_size, strides,
                        ds=1, shuffle=True, rngseed=None,
                        dataslice=None, add_overhanging=True):
    # only support lists or tuples for shape, window_size and strides
    assert isinstance(shape, (list, tuple))
    assert isinstance(window_size, (list, tuple)), "%s" % (str(type(window_size)))
    assert isinstance(strides, (list, tuple))

    dim = len(shape)
    assert len(window_size) == dim
    assert len(strides) == dim

    # check for downsampling
    assert isinstance(ds, (list, tuple, int))
    if isinstance(ds, int):
        ds = [ds] * dim
    assert len(ds) == dim

    # Seed RNG if a seed is provided
    if rngseed is not None:
        random.seed(rngseed)

    # sliding windows in one dimenstion
    def dimension_window(start, stop, wsize, stride, dimsize, ds_dim):
        starts = range(start, stop + 1, stride)
        slices = [slice(st, st + wsize, ds_dim) for st in starts if st + wsize <= dimsize]

        # add an overhanging window at the end if the windoes
        # do not fit and `add_overhanging`
        if slices[-1].stop != dimsize and add_overhanging:
            slices.append(slice(dimsize - wsize, dimsize, ds_dim))

        if shuffle:
            random.shuffle(slices)
        return slices

    # determine adjusted start and stop coordinates if we have a dataslice
    # otherwise predict the whole volume
    if dataslice is not None:
        assert len(dataslice) == dim, "Dataslice must be a tuple with len = data dimension."
        starts = [sl.start for sl in dataslice]
        stops  = [sl.stop - wsize for sl, wsize in zip(dataslice, window_size)]
    else:
        starts = dim * [0]
        stops  = [dimsize - wsize if wsize != dimsize else dimsize
                  for dimsize, wsize in zip(shape, window_size)]

    assert all(stp > strt for strt, stp in zip(starts, stops)),\
        "%s, %s" % (str(starts), str(stops))
    nslices = [dimension_window(start, stop, wsize, stride, dimsize, ds_dim)
               for start, stop, wsize, stride, dimsize, ds_dim
               in zip(starts, stops, window_size, strides, shape, ds)]
    return itertools.product(*nslices)


def array_to_slices(array):
    assert array.shape[-1] == 3
    shape = array.shape[:-1]
    return np.array([slice(*args) for args in array.reshape(-1, 3)]).reshape(*shape)


def slices_to_array(slice_list):
    slice_list = np.array(slice_list)
    shape = slice_list.shape
    return np.array([[sl.start, sl.stop, sl.step] for sl in slice_list.flatten()]).reshape(*shape, 3) # changed dtype from int32


class VolumeLoader(Dataset):
    """
    Minimal Volume Loader
    """

    def __init__(self, volume, base_sequence, transforms):
        super(VolumeLoader, self).__init__()
        # Validate volume
        assert isinstance(volume, np.ndarray), str(type(volume))
        self.volume = volume
        self.base_sequence = base_sequence
        self.transforms = transforms

    def __getitem__(self, index):
        # Casting to int would allow index to be IndexSpec objects.
        index = int(index)
        slices = self.base_sequence[index]
        sliced_volume = self.volume[tuple(slices)]
        return self.transforms(sliced_volume)

    def __len__(self):
        return len(self.base_sequence)


def parse_prediction(prediction):
    sigma_dim = 1
    emb = prediction[:, 1+sigma_dim:]

    # add coordinates
    xy = np.stack(np.mgrid[:emb.shape[-2], :emb.shape[-1]]) / 100
    xy = torch.FloatTensor(xy).to(emb.device)
    emb[:, :2] += xy[None]

    sigmas = prediction[:, 1:1+sigma_dim]
    sigmas = torch.exp(sigmas)
    seed_maps = torch.sigmoid(prediction[:, :1])

    return emb, sigmas, seed_maps


def gaussian_mask_growing_segmentation(embedding, seed_map, sigma_map, seed_score_threshold=0.5, min_cluster_size=1):
    """
    Compute a segmentation by sequentially growing masks based on a gaussian similarity.
    :param embedding: torch.FloatTensor
    Shape should be (E D H W) or (E H W) (embedding dimension + spatial dimensions).
    :param seed_map: torch.FloatTensor
    Shape should be (D H W) or (H W) (arbitrary number of spatial dimensions).
    :param sigma_map: torch.FloatTensor
    Shape should be (D H W) or (H W) (arbitrary number of spatial dimensions).
    :param seed_score_threshold float
    If no seeds with at least this score are present, clustering stops.
    :param min_cluster_size int
    Clulsters with a smaller number of pixels will be discarded.
    :return: torch.LongTensor
    the computed segmentation
    """
    spatial_shape = embedding.shape[1:]
    segmentation = torch.zeros(spatial_shape).long().to(embedding.device)
    mask = seed_map > 0.5

    if mask.sum() < min_cluster_size:  # nothing to do
        return segmentation

    masked_embedding = embedding[:, mask]
    masked_seed_map = seed_map[mask]
    masked_sigma_map = sigma_map[mask]
    masked_segmentation = torch.zeros_like(masked_seed_map).long()
    unclustered = torch.ones_like(masked_segmentation).bool()

    current_id = 1
    while unclustered.sum() >= min_cluster_size:
        # get position of seed with highest score
        seed = (masked_seed_map * unclustered.float()).argmax().item()
        seed_score = masked_seed_map[seed]

        if seed_score < seed_score_threshold:
            break

        center = masked_embedding[:, seed]
        unclustered[seed] = 0  # just to ensure convergence
        sigma = masked_sigma_map[seed]
        smooth_mask = torch.exp(-1 * ((((masked_embedding - center[:, None]) / sigma) ** 2).sum(0)))

        proposal = smooth_mask > 0.5

        if proposal.sum() > min_cluster_size:
            if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:  # half of proposal not assigned yet
                masked_segmentation[proposal] = current_id
                current_id += 1

        unclustered[proposal] = 0

    segmentation[mask] = masked_segmentation

    return segmentation


def compute_global_segmentation(raw_volume, model, use_cuda=False, min_cluster_size=100):
    if len(raw_volume.shape) == 2:
        raw_volume = raw_volume[None]

    # get window size and overlap
    window_size = (1, 512, 512)
    n_dim = len(window_size)
    # 96 is more than maximum FOV in any direction, and multiple of 16
    overlap = [0, 96 * 2, 96 * 2]
    stride = [w - o for w, o in zip(window_size, overlap)]

    fovs_per_dim = [2 + int((raw_volume.shape[i] - window_size[i] - 0.5) // stride[i])
                    for i in range(len(raw_volume.shape))]
    padded_shape = [w + (n_fov - 1) * s
                    for w, n_fov, s in zip(window_size, fovs_per_dim, stride)]
    raw_volume_padded = np.pad(raw_volume, [(0, p - s) for p, s in zip(padded_shape, raw_volume.shape)])

    # compute slices for inference
    inference_slices = list(
        slidingwindowslices(padded_shape, window_size=window_size, strides=stride, shuffle=False))

    # compute slices for global assignment
    slice_array = slices_to_array(inference_slices).reshape(fovs_per_dim + [3, 3])
    local_slice_array = np.ones_like(slice_array)
    local_slice_array[..., 0] = np.array(overlap) / 2  # start
    local_slice_array[..., 1] = window_size - np.array(overlap) / 2  # stop
    global_slice_array = slice_array.copy()

    def empty_slice():
        return [slice(None), ] * n_dim

    for i in range(n_dim):
        # fix local slices at the border of the volume
        sl = empty_slice()
        sl[i] = 0  # extend first fov in that direction
        local_slice_array[tuple(sl) + (i, 0)] = 0
        sl[i] = -1  # extend last fov
        local_slice_array[tuple(sl) + (i, 1)] = window_size[i]

        # set global slices
        sl = empty_slice()
        sl[i] = slice(1, None)  # everything apart from last
        global_slice_array[tuple(sl) + (i, 0)] += overlap[i] // 2
        sl[i] = slice(None, -1)  # everything apart from first
        global_slice_array[tuple(sl) + (i, 1)] -= overlap[i] // 2

    local_slices = array_to_slices(local_slice_array.reshape(-1, 3, 3).astype(np.int32))
    global_slices = array_to_slices(global_slice_array.reshape(-1, 3, 3).astype(np.int32))

    # initialize loader
    def transforms(array):
        return torch.Tensor(array).float()

    loader = VolumeLoader(raw_volume_padded, base_sequence=inference_slices, transforms=transforms)

    # apply the model
    patch_predictions = []
    if use_cuda:
        model.cuda()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting on patches'):
            batch = batch[:, None]
            if use_cuda:
                batch = batch.cuda()
            out = model(batch)
            # The None is a fake depth just because the rest of the processing thinks the data is 3D..
            out = list(np.array(out.cpu())[:, :, None])
            patch_predictions.extend(out)
    patch_predictions = np.stack(patch_predictions)

    # construct the global prediction
    global_prediction = np.zeros([patch_predictions.shape[1]] + padded_shape)
    for local_slice, global_slice, patch in zip(local_slices, global_slices, patch_predictions):
        global_prediction[(slice(None),) + tuple(global_slice)] = patch[(slice(None),) + tuple(local_slice)]
    global_prediction = global_prediction[(slice(None),) + tuple((slice(0, s) for s in raw_volume.shape))]

    # parse prediction
    embedding, sigma_map, seed_map = parse_prediction(torch.FloatTensor(global_prediction).transpose(0, 1))

    # compute the segmentation
    global_segmentation = np.stack(
        [gaussian_mask_growing_segmentation(
            embedding[i], seed_map[i, 0], sigma_map[i, 0],
            min_cluster_size=min_cluster_size,
            seed_score_threshold=0.6,  # TODO: make a parameter

        ).cpu().numpy()
         for i in tqdm(range(embedding.shape[0]), desc='Computing segmentations')])

    return global_segmentation


def extract_convex_hulls(segmentation, ignore_label=0, labels=None):
    xy = np.stack(np.mgrid[:segmentation.shape[0], :segmentation.shape[1]], axis=-1)[:, :, ::-1]
    polygons = []
    hulls = []
    good_labels = []  # to store labels of segments for which the computation of the convex hull was successful
    for label in np.unique(segmentation) if labels is None else labels:
        if label == ignore_label:
            continue
        points = xy[segmentation == label]
        try:
            hull = ConvexHull(points)
            hulls.append(hull)
            polygons.append(points[hull.vertices])
            good_labels.append(label)
        except QhullError as e:
            print('Skipping segment due to Error while trying to compute its convex hull:')
            print(e)
            print()

    return hulls, polygons, good_labels


def get_flattened_border(array, width=1):
    """
    returns a flat array containing all pixels in a width-d margin of the array
    """
    left_slice, right_slice = slice(0, width), slice(-width, None)
    middle_slice = slice(width,-width)
    assert all([s >= 2*width for s in array.shape]), f'Array too small: {array.shape}, {width}'

    def directional_border_slices(i):
        return [(middle_slice,) * i + (left_slice,), (middle_slice,) * i + (right_slice,)]

    return np.concatenate([array[s].flatten()
                           for i in range(len(array.shape))
                           for s in directional_border_slices(i)])


def postprocess_segmentation(segmentation, area_ratio_threshold=0.9):
    ignore_label = 0

    # remove segments on border
    for label in np.unique(get_flattened_border(segmentation, width=3)):
        if label == ignore_label:
            continue
        segmentation[segmentation == label] = ignore_label

    labels = np.array([label for label in np.unique(segmentation) if label != ignore_label])

    hulls, polys, hull_labels = extract_convex_hulls(segmentation, ignore_label=ignore_label, labels=labels)
    hull_areas = np.array([hull.volume for hull in hulls])
    segment_areas = np.array([np.sum(segmentation == label)
                              for label in hull_labels])

    # good labels are those for which we can compute a convex hull, and the area of the hull is not too big.
    good_labels = np.array(hull_labels)[segment_areas / hull_areas >= area_ratio_threshold]
    for label in labels:
        if label not in good_labels:
            segmentation[segmentation == label] = ignore_label
    return segmentation


_hex_decoder = codecs.getdecoder("hex_codec")
decode_hex = lambda s: _hex_decoder(s)[0]


def int_to_hex(i):  # for 16 bit integers
    return decode_hex('{:04x}'.format(i)) #int(i).to_bytes(2, 'big') #


def float_to_hex(f):  # for 32 bit floats
    print(struct.unpack('<I', struct.pack('<f', f))[0])
    return struct.unpack('<I', struct.pack('<f', f))[0].to_bytes(4, 'big')


def write_imagej_polygon_roi(coords, name, path, stroke_width=2, stroke_col='88FFFF00', fill_col='00000000'):
    """
    Creates ImageJ compatible ROI files for points or circles
    :param coords: tuple
    for circles (t,l,b,r) or (x,y,r) for points (x,y)
    :param name: str
    Name of the roi (and exported file)
    :param path: str
    path to write the output files
    :param stroke_width: int
    :param stroke_col: str
    hex code of stroke color
    :param fill_col: str
    hex code of fill color
    :return: str
    Path to output file
    """
    coords = np.array(coords, dtype=int)
    assert len(coords.shape) == 2, coords.shape[-1] == 2

    offset_filename = 64 + 4 * len(coords) + 64
    left, top, right, bottom = np.min(coords[:, 0]), np.min(coords[:, 1]), np.max(coords[:, 0]), np.max(coords[:, 1])
    coords[:, 0] -= left
    coords[:, 1] -= top
    filelength = offset_filename + len(name) * 2
    data = bytearray(filelength)

    data[0:4] = b'\x49\x6F\x75\x74'  # "Iout" 0-3
    data[4:6] = b'\x00\xE3'  # Version 4-5

    data[6:8] = b'\x00\x00'                 # roi type   6-7
    data[8:10] = int_to_hex(top)            # top      8-9
    data[10:12] = int_to_hex(left)          # left    10-11
    data[12:14] = int_to_hex(bottom)        # bottom  12-13
    data[14:16] = int_to_hex(right)         # right   14-15
    data[34:36] = int_to_hex(stroke_width)  # Stroke Width  34-35
    data[40:44] = decode_hex(stroke_col)    # Stroke Color 40-43
    data[44:48] = decode_hex(fill_col)      # Fill Color 44-47
    data[60:64] = int(offset_filename - 64).to_bytes(4, 'big') #b'\x00\x00\x00\x50'       # header2offset 60-63

    # number of coords
    data[16:18] = int_to_hex(len(coords))   # n_coords 16-17

    base1 = 64
    base2 = base1 + 2*len(coords)
    for i, (x, y) in enumerate(coords):
        data[base1 + 2*i:base1 + 2*(i+1)] = int_to_hex(x)
        data[base2 + 2*i:base2 + 2*(i+1)] = int_to_hex(y)

    off = base2 + 2 * (i+1) + 1
    data[off:off+2] = b'\x00\x80'  # Name offset
    data[off+4:off+6] = int_to_hex(len(name))  # Name Length

    p = offset_filename  # add name
    for c in name:
        data[p] = 0x00
        data[p + 1] = ord(c)
        p = p + 2

    filename = os.path.join(path, name + ".roi")  # write file
    file = open(filename, 'wb')
    file.write(data)
    file.close()
    return filename


def save_rois(segmentation, out_file='roiset.zip'):
    ignore_label = 0
    hulls, polys, _ = extract_convex_hulls(segmentation, ignore_label=ignore_label)
    with tempfile.TemporaryDirectory() as tempdir:
        for i, poly in enumerate(np.array(polys)):
            write_imagej_polygon_roi(poly, f'cell_{str(i).rjust(4, "0")}', tempdir)
        with zipfile.ZipFile(out_file, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in os.listdir(tempdir):
                path = os.path.join(tempdir, file)
                archive.write(path, arcname=file)
                os.remove(path)


if __name__ == '__main__':
    import sys
    import h5py
    from imageio import imread
    import argparse

    parser = argparse.ArgumentParser(description='Predict ImageJ ROIs for Yeast Cells.')
    parser.add_argument('--input_location', '-i', type=str, default='data',
                        help='Input data location. Can either be a directory or a path to a TIF file.')
    parser.add_argument('--output_directory', '-o', type=str, default=None,
                        help='Directory in which the predicted ROIs will be saved. '
                             'By default identical to the input directory.')
    parser.add_argument('--model', '-m', type=str, default='model.pytorch',
                        help='Location of the file containing the model to be evaluated.')
    parser.add_argument('--use_gpu', '-g', dest='use_gpu', action='store_const',
                        const=True, default=False,
                        help='Whether or not to use the GPU. ')
    parser.add_argument('--min_size', type=int, default=100)
    args = parser.parse_args()

    model_path = args.model
    data_path = args.input_location
    output_dir = args.output_directory

    print('Loading model')
    model = torch.load(model_path)
    model.cpu()
    print('Loading input images')
    assert os.path.exists(data_path), f'Data not found: Path {data_path} does not exist.'
    if os.path.isfile(data_path):
        data_dir = os.path.dirname(data_path)
        if data_path.endswith('.h5'):
            print('Loading hdf5 data volume')
            with h5py.File(data_path, 'r') as f:
                raw_volume = f['raw_normalized'][()]
            image_names = [f'{i:04d}' for i in range(len(raw_volume))]
        elif data_path.endswith('.tif') or data_path.endswith('.tiff'):
            print(f'Loading .tif image at {data_path}')
            raw_volume = np.array(imread(data_path))[None]
            image_names = ['.'.join(os.path.basename(data_path).split('.')[:-1])]
        else:
            assert False, f'Bad data type {data_path.split(".")[-1]}. Need directory, tif or h5 file.'
    elif os.path.isdir(data_path):
        print(f'Searching for tif files in {data_path}')
        image_files = [f for f in os.listdir(data_path) if f.endswith('.tif') or f.endswith('.tiff')]
        assert len(image_files) > 0, f'Found no tif files in {data_path}.'

        images = []
        for filename in image_files:
            print(f'Found {filename}')
            image = imread(os.path.join(data_path, filename))
            assert len(image.shape) == 2, f'Need one-channel tif file. Got shape {image.shape}'
            images.append(image)
        raw_volume = np.stack(images)
        image_names = ['.'.join(filename.split('.')[:-1]) for filename in image_files]
        data_dir = data_path
    else:
        assert False, f'{data_path} is neither a directory nor a file.'

    print('Normalizing input data')
    raw_volume = raw_volume.astype(np.float32)
    # subtract median
    raw_volume -= np.median(raw_volume.reshape(raw_volume.shape[0], -1), axis=-1)[:, None, None]
    # divide by std
    raw_volume /= np.std(raw_volume.reshape(raw_volume.shape[0], -1), axis=-1)[:, None, None]

    # compute segmentation
    seg = compute_global_segmentation(
        raw_volume, model,
        use_cuda=args.use_gpu,
        min_cluster_size=args.min_size,
    )
    seg = np.stack([postprocess_segmentation(s)
                    for s in tqdm(seg, desc='Postprocessing segmentations')])

    # save results
    if output_dir is None:
        output_dir = data_dir

    for i, s in enumerate(seg):
        out_file = os.path.join(output_dir, image_names[i] + '_roiset.zip')
        save_rois(s, out_file)
        print(f'Writing predicted ROIs to {out_file}')
    print('Done.')




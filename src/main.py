import os

import cv2
from dotenv import load_dotenv
import numpy as np
from pprint import pprint
import supervisely as sly
from supervisely.project.project_type import ProjectType
from supervisely._utils import batched


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

# get variables from enviroment
workspace_id = sly.env.workspace_id()

# create new project at Supervisely platform
project = api.project.create(
    workspace_id,
    "Volume tutorial",
    ProjectType.VOLUMES,
    change_name_if_conflict=True,
)
print(f"Project ID: {project.id}")

# create new dataset at Supervisely platform
dataset = api.dataset.create(project.id, "dataset_1")
print(f"Dataset ID: {dataset.id}")

# prepare nrrd files and place them in local directory ("src/upload/nrrd/")
upload_path = "src/upload/nrrd/MRHead.nrrd"

# upload 1 nnrd volume as nrrd from local directory to Supervisely platform
nrrd_info = api.volume.upload_nrrd_serie_path(
    dataset.id,
    "MRHead.nrrd",
    upload_path,
)
print(f'"{nrrd_info.name}" volume uploaded to Supervisely with ID:{nrrd_info.id}')


# upload volume as NumPy array to Supervisely platform
np_volume, meta = sly.volume.read_nrrd_serie_volume_np(upload_path)
nrrd_info_np = api.volume.upload_np(
    dataset.id,
    "MRHead_np.nrrd",
    np_volume,
    meta,
)
print(f"Volume uploaded as NumPy array to Supervisely with ID:{nrrd_info_np.id}")


# upload list of nrrd files from local directory to Supervisely
upload_dir_name = "src/upload/nrrd/"
all_nrrd_names = os.listdir(upload_dir_name)
names = [f"1_{name}" for name in all_nrrd_names]
paths = [os.path.join(upload_dir_name, name) for name in all_nrrd_names]

infos = api.volume.upload_nrrd_series_paths(dataset.id, names, paths)
print(f"All volumes has been uploaded with IDs: {[x.id for x in infos]}")


# get list of all volumes from current dataset from Supervisely
volume_infos = api.volume.get_list(dataset.id)
volumes_ids = [x.id for x in volume_infos]
print(f"List of volumes`s IDs: {volumes_ids}")

# get volume info from Supervisely platform by volume id
volume_id = volume_infos[0].id
volume_info_by_id = api.volume.get_info_by_id(id=volume_id)
print(f"Volume name: ", volume_info_by_id.name)

# get volume info from Supervisely platform by volume name
volume_info_by_name = api.volume.get_info_by_name(dataset.id, name="MRHead.nrrd")
print(f"Volume name: ", volume_info_by_name.name)


# prepare dicom files and place them in local directory ("src/upload/MRHead_dicom/")
dicom_dir_name = "src/upload/MRHead_dicom/"

# inspect you local directory and collect all dicom series.
series_infos = sly.volume.inspect_dicom_series(root_dir=dicom_dir_name)

# upload DICOM volume from local directory to Supervisely platform
for serie_id, files in series_infos.items():
    item_path = files[0]
    name = f"{sly.fs.get_file_name(path=item_path)}.nrrd"
    dicom_info = api.volume.upload_dicom_serie_paths(
        dataset_id=dataset.id,
        name=name,
        paths=files,
        anonymize=True,  # hide patient's name and ID before uploading to Supervisely platform
    )
    print(f"DICOM volume has been uploaded to Supervisely with ID: {dicom_info.id}")


# download current volume from Supervisely to local path
volume_id = volume_infos[0].id
volume_info = api.volume.get_info_by_id(id=volume_id)
download_dir_name = "src/download/"

path = os.path.join(download_dir_name, volume_info.name)
if os.path.exists(path):
    os.remove(path)

api.volume.download_path(volume_info.id, path)

if os.path.exists(path):
    print(f"Volume (ID {volume_info.id}) successfully downloaded.")


# read nrrd file from local directory
nrrd_path = os.path.join(download_dir_name, "MRHead.nrrd")
volume_np, meta = sly.volume.read_nrrd_serie_volume_np(nrrd_path)
pprint(meta)

# get all sagittal slices
sagittal_slices = {}
dimension = volume_np.shape[0]  # indexes: 0 - sagittal, 1 - coronal, 2 - axial
for batch in batched(list(range(dimension))):
    for i in batch:
        if i >= dimension:
            continue
        pixel_data = volume_np[i, :, :]  # indexes: 0 - sagittal, 1 - coronal, 2 - axial
        sagittal_slices[i] = pixel_data

print(f"{len(sagittal_slices.keys())} slices has been received from current volume")

for i, s in sagittal_slices.items():
    frame = np.array(s, dtype=np.uint8)
    cv2.imshow(f"frame #{i}", frame)
    cv2.waitKey(10)
    cv2.destroyAllWindows()

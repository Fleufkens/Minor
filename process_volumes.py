import os
from os.path import isfile, abspath, join
from glob import glob
import pydicom
import numpy as np
import SimpleITK as sitk
import subprocess
from tkinter import filedialog


def parse_dcm_metadata(dcm):
    unpacked_data = {}
    group_elem_to_keywords = {}
    # iterating here to force conversion from lazy RawDataElement to DataElement
    for d in dcm:
        pass

    # keys are pydicom.tag.BaseTag, values are pydicom.dataelem.DataElement
    for tag, elem in dcm.items():
        tag_group = tag.group
        tag_elem = tag.elem
        keyword = elem.keyword
        group_elem_to_keywords[(tag_group, tag_elem)] = keyword
        value = elem.value
        if keyword != '' and keyword != 'PixelData':
            unpacked_data[keyword] = value

    return unpacked_data, group_elem_to_keywords

def filtering_dcm_files_by_phase(dcm_files, unknown_flag=1):
    output_dcm_files = []
    slice_locations = []
    if unknown_flag:
        dcm_files = [f for f in dcm_files if 'unknown' not in f.split('/')[-1].split('--')[-1]]
    sorted_dcm_files = sorted(dcm_files, key=lambda x:x.split('/')[-1].split('--')[0])
    for filename in sorted_dcm_files:
        try:
            dicom_raw = pydicom.dcmread(filename)
            meta_dict, _ = parse_dcm_metadata(dicom_raw)
            ipp = meta_dict['ImagePositionPatient']
            cosines = meta_dict['ImageOrientationPatient']
            slice_locations.append(compute_slice_location(cosines, ipp))
        except:
            print("{} has A PROBLEM!".format(filename))
            continue

    output_dcm_files = [dcm for _, dcm in sorted(zip(slice_locations, sorted_dcm_files))]
    return output_dcm_files

def compute_slice_location(cosines, ipp):
    ipp_a = np.array(ipp)
    normal = np.zeros(ipp_a.shape)
    normal[0] = float(cosines[1]*cosines[5]-cosines[2]*cosines[4])
    normal[1] = float(cosines[2]*cosines[3]-cosines[0]*cosines[5])
    normal[2] = float(cosines[0]*cosines[4]-cosines[1]*cosines[3])
    return np.dot(normal,ipp_a)

def load_dicoms(dcm_files, unknown_flag=1):
    locs = []
    dcms = []
    dcm_files = filtering_dcm_files_by_phase(dcm_files, unknown_flag)

    hdr_out = {}
    for i in range(len(dcm_files)):
        from_pydicom = 0
        ds = pydicom.read_file(dcm_files[i])
        try:
            data = ds.pixel_array
            from_pydicom = 1
        except:
            try:
                file = sitk.ReadImage(dcm_files[i])
                data = sitk.GetArrayFromImage(file)
                data = np.squeeze(data)
            except:
                print("CANNOT READ {}".format(dcm_files[i]))
                continue

        pixel_spacing = ds.PixelSpacing
        ipp = ds.ImagePositionPatient
        cosines = ds.ImageOrientationPatient

        if ds.RescaleSlope and from_pydicom:
            data = data * ds.RescaleSlope + ds.RescaleIntercept
        
        hdr = {}
        hdr['PixelSpacing'] = [str(pixel_spacing[0]), str(pixel_spacing[1])]

        if i == 0:
            hdr_out = hdr

        real_slice_location = compute_slice_location(cosines, ipp)
        if not real_slice_location in locs:
            dcms += [(data, hdr)]
            locs += [real_slice_location]

    if len(locs) > 1:
        hdr_out['SliceSpacing'] = str(np.abs(locs[1]-locs[0]))
    else:
        hdr_out['SliceSpacing'] = str(1.0)

    return dcms, hdr_out

if __name__ == '__main__':
    dicom_path = filedialog.askdirectory(title = "Open dicom series or folder of studies")
    vol_seg_path = filedialog.askdirectory(title = "Path to volume segmentation package")
    print('Dicom path: ', dicom_path)
    print('Volume segmentation path: ', vol_seg_path)

    if len(glob(dicom_path + "/*.dcm")) > 0:
        folder_paths = [dicom_path]
    elif len(glob(dicom_path + "/*/*.dcm")) > 0:
        folder_paths = glob(dicom_path + '/*')
    elif len(glob(dicom_path + "/*/*/*.dcm")) > 0:
        folder_paths = glob(dicom_path + '/*/*')
    else:
        print('Could not find DICOM files')
        exit()

    if not isfile(join(vol_seg_path, 'volumeSegmentationLeftHeart.exe')):
        print('Could not find volume segmentation executable')
        exit()
    
    len_studies = len(folder_paths)
    for ind, study_path in enumerate(folder_paths):
        # Read dicom data
        study_path = study_path.replace("\\", "/")
        dcm_list = glob(study_path + "/*.dcm")
        if len(dcm_list) == 1:
            continue
        dcm_list = [dcm_name.replace("\\", "/") for dcm_name in dcm_list]
        study_name = dcm_list[0].split('/')[-2]
        
        print('Processing Study {}/{}: {}'.format(ind+1, len_studies, study_name))
        dcms, hdr = load_dicoms(dcm_list)

        # Create volume
        volume = None
        for dcm in dcms:
            slice = np.array(dcm[0])
            slice = np.expand_dims(np.expand_dims(slice, axis=0), axis=-1)
            volume = slice if volume is None else np.concatenate((volume, slice), axis=0)

        # Normalize intensities according to CT window used by network
        volume[volume < -100] = -100
        volume[volume > 1500] = 1500
        volume = (volume + 100) / 1600

        # Write volume as MetaImage for volume segmentation package
        image_array = sitk.GetImageFromArray(volume.astype(np.float32))
        image_array.SetSpacing([
            float(hdr['PixelSpacing'][0]), float(hdr['PixelSpacing'][1]), float(hdr['SliceSpacing']) ])
        sitk.WriteImage(image_array, study_name + '.mhd')
        
        input_file = abspath(study_name + '.mhd')[:-4].replace("\\", "/")
        output_file = abspath(study_name + '_seg.mhd')[:-4].replace("\\", "/")

        # Run Left Heart Segmentation using volume segmentation package
        # Write segmentation output (as MetaImage)
        args = ['volumeSegmentationLeftHeart.exe', '-i', input_file, '-o', output_file]
        wd = os.getcwd()
        os.chdir(abspath(vol_seg_path))
        sub_process = subprocess.Popen(args)
        sub_process.wait()
        sub_process.terminate()
        os.chdir(wd)


import argparse
from glob import glob
import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor

# Radiomics class for extracting radiomic features from images and masks.
class Radiomics:

    # Initialization method used to define patient details and path information.
    def __init__(self, patient, path):
        self.patient = patient
        self.path = path
        # Paths for parameters are set here.
        self.params_full = os.path.join(self.path, "params_full.yaml")
        self.params_rb = os.path.join(self.path, "params_rb.yaml")
        self.params_bb = os.path.join(self.path, "params_bb.yaml")
        self.series = str(glob(os.path.join(self.path, "Series", self.patient, "ST*"))[0])
        # Variables for images, masks, radiomic biopsies, and bounding boxes are initialized.
        self.image = None
        self.mask = None
        self.rb = None
        self.bb = None
        self.nnunet = None

    # Method to convert DICOM series to NIfTI format.
    def dicom_series_to_nii(self):
        reader = sitk.ImageSeriesReader()
        series_dict = {series_id: reader.GetGDCMSeriesFileNames(self.series, series_id) for series_id in
                       reader.GetGDCMSeriesIDs(self.series)}
        dicoms = [item for sub in series_dict.values() for item in sub]

        if self.patient in ["PA_004", "PA_027", "PA_068"]:
            if self.patient in ["PA_068"]:
                dicoms = dicoms[17::]
            else:
                dicoms = dicoms[1::]
        else:
            dicoms = dicoms
        reader.SetFileNames(dicoms)
        self.image = reader.Execute()

    # Method to read the mask and preprocess it.
    def read_mask(self):
        reader = sitk.ImageFileReader()

        if os.path.exists(os.path.join(self.path, "Segmentations", self.patient + "_msk.nii.gz")):
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(os.path.join(self.path, "Segmentations", self.patient + "_msk.nii.gz"))
        elif os.path.exists(os.path.join(self.path, "Segmentations", self.patient + "_msk.nrrd")):
            reader.SetImageIO("NrrdImageIO")
            reader.SetFileName(os.path.join(self.path, "Segmentations", self.patient + "_msk.nrrd"))
        self.mask = reader.Execute()

        if self.patient in ["PA_004", "PA_027", "PA_068"]:
            if self.patient in ["PA_068"]:
                mask = self.mask[:, :, 17::]
            else:
                mask = self.mask[:, :, 1::]
            mask.SetSpacing(self.image.GetSpacing())
            mask.SetOrigin(self.image.GetOrigin())
            mask.SetDirection(self.image.GetDirection())
            self.mask = mask
        else:
            self.mask = self.mask

    # Method to read radiomic biopsies and preprocess them.
    def read_rb(self):
        reader = sitk.ImageFileReader()

        if os.path.exists(os.path.join(self.path, "Radiomic_Biopsies", self.patient + "_rb.nii.gz")):
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(os.path.join(self.path, "Radiomic_Biopsies", self.patient + "_rb.nii.gz"))
        elif os.path.exists(os.path.join(self.path, "Radiomic_Biopsies", self.patient + "_rb.nrrd")):
            reader.SetImageIO("NrrdImageIO")
            reader.SetFileName(os.path.join(self.path, "Radiomic_Biopsies", self.patient + "_rb.nrrd"))
        self.rb = reader.Execute()

        if self.patient in ["PA_004", "PA_027", "PA_068"]:
            if self.patient in ["PA_068"]:
                rb = self.rb[:, :, 17::]
            else:
                rb = self.rb[:, :, 1::]
            rb.SetSpacing(self.image.GetSpacing())
            rb.SetOrigin(self.image.GetOrigin())
            rb.SetDirection(self.image.GetDirection())
            self.rb = rb
        else:
            self.rb = self.rb

    # Method to read nnunet images and preprocess them.
    def read_nnunet(self):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(os.path.join(self.path, "nnUNet", self.patient + ".nii.gz"))
        self.nnunet = reader.Execute()

    # Method to write the image after processing.
    def write_image(self):
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(self.path, "Samples", self.patient, self.patient + "_img.nii.gz"))
        writer.Execute(self.image)

    # Method to write the mask after processing.
    def write_mask(self):
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(self.path, "Samples", self.patient, self.patient + "_msk.nii.gz"))
        writer.Execute(self.mask)

    # Method to write radiomic biopsy after processing.
    def write_rb(self):
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(self.path, "Samples", self.patient, self.patient + "_rb.nii.gz"))
        writer.Execute(self.rb)

    # Write the processed bounding box images.
    def write_bb(self):
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(self.path, "Samples", self.patient, self.patient + "_bb.nii.gz"))
        writer.Execute(self.bb)

    # Write the processed nnunet results.
    def write_nnunet(self):
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(self.path, "Samples", self.patient, self.patient + "_nnunet.nii.gz"))
        writer.Execute(self.nnunet)

    # Generate the connected components for the images.
    def connected_components(self):
        cc = sitk.ConnectedComponentImageFilter()
        cc.SetFullyConnected(True)

        stats = sitk.LabelShapeStatisticsImageFilter()

        self.mask = cc.Execute(self.mask)
        stats.Execute(self.mask)
        label_size = sorted([stats.GetPhysicalSize(label) for label in stats.GetLabels()], reverse=True)
        relabel_map = {i: (1 if stats.GetPhysicalSize(i) == label_size[0] else 0) for i in stats.GetLabels()}
        self.mask_vol = label_size[0]
        self.mask = sitk.ChangeLabel(self.mask, changeMap=relabel_map)
        self.mask = sitk.Clamp(self.mask, upperBound=1)

        self.rb = cc.Execute(self.rb)
        stats.Execute(self.rb)
        label_size = sorted([stats.GetPhysicalSize(label) for label in stats.GetLabels()], reverse=True)
        relabel_map = {i: (1 if stats.GetPhysicalSize(i) == label_size[0] else 0) for i in stats.GetLabels()}
        self.rb = sitk.ChangeLabel(self.rb, changeMap=relabel_map)
        self.rb = self.rb != 0
        self.rb = sitk.Clamp(self.rb, upperBound=1)

    # Method to resample the images.
    def resample_image(self, is_label=False):
        out_spacing = [1.0, 1.0, 1.0]
        original_spacing = self.image.GetSpacing()
        original_size = self.image.GetSize()
        out_size = [int(round(1 + (osz - 1) * ospc / out_spc)) for out_spc, osz, ospc in
                    zip(out_spacing, original_size, original_spacing)]
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(self.image.GetDirection())
        resample.SetOutputOrigin(self.image.GetOrigin())
        resample.SetTransform(sitk.Transform())

        if is_label:
            resample.SetOutputPixelType(self.mask.GetPixelID())
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
            self.mask = resample.Execute(self.mask)
            self.rb = resample.Execute(self.rb)
            self.nnunet = resample.Execute(self.nnunet)
        else:
            resample.SetOutputPixelType(self.image.GetPixelID())
            resample.SetInterpolator(sitk.sitkBSpline)
            self.image = resample.Execute(self.image)

    # Method that generates a bounding box based on the mask.
    def make_bb(self):
        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(self.mask)
        bb = lsif.GetBoundingBox(1)

        roi = sitk.RegionOfInterestImageFilter()
        roi.SetRegionOfInterest(bb)
        self.bb = roi.Execute(self.mask)
        self.bb = sitk.Mask(self.bb, sitk.Not(self.bb != 1), 1)
        self.bb = sitk.Resample(self.bb, self.image, sitk.Transform(), sitk.sitkNearestNeighbor, 0,
                                self.bb.GetPixelID())

    # Method to perform dilation operation on the mask.
    def dilate(self, radius):
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius(radius)
        dilate_filter.SetKernelType(sitk.sitkBall)
        dilate_filter.SetForegroundValue(1)
        self.mask = dilate_filter.Execute(self.mask)
        self.rb = dilate_filter.Execute(self.rb)
        self.bb = dilate_filter.Execute(self.bb)

    # Method to perform erosion operation on the mask.
    def erode(self, radius):
        erode_filter = sitk.BinaryErodeImageFilter()
        erode_filter.SetKernelRadius(radius)
        erode_filter.SetKernelType(sitk.sitkBall)
        erode_filter.SetForegroundValue(1)
        self.mask = erode_filter.Execute(self.mask)
        self.rb = erode_filter.Execute(self.rb)
        self.bb = erode_filter.Execute(self.bb)

    # Method to Calculate DICE coefficient.
    def get_dice(self):
        masks = [self.nnunet]
        df = []
        for i in range(len(masks)):
            im1 = sitk.GetArrayFromImage(self.mask).astype(bool)
            im2 = sitk.GetArrayFromImage(masks[i]).astype(bool)

            if im1.shape != im2.shape:
                raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

            # Compute Dice coefficient
            intersection = np.logical_and(im1, im2)

            dice = 2. * intersection.sum() / (im1.sum() + im2.sum())
            df.append(dice)
        df1 = pd.DataFrame({'Dice': df})
        df1.insert(0, "Patient", self.patient)
        df1.to_csv(os.path.join(self.path, "Dice", self.patient + ".csv"))

    # Method to compute the volume of the masks
    def get_volume(self):
        mask_list = [self.mask, self.rb, self.bb]
        df = []
        for i in range(len(mask_list)):
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(mask_list[i])
            volume = stats.GetPhysicalSize(1)
            df.append(volume)
        df1 = pd.DataFrame({'Volume': df})
        df1.insert(0, "Patient", self.patient)
        df1.to_csv(os.path.join(self.path, "Volume", self.patient + ".csv"))

    # Method to pad the images to a uniform size
    def pad_image(self, pad=64):
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound([pad, pad, 0])
        pad_filter.SetPadUpperBound([pad, pad, 0])
        self.image = pad_filter.Execute(self.image)
        pad_filter.SetConstant(0)
        self.mask = pad_filter.Execute(self.mask)
        self.rb = pad_filter.Execute(self.rb)
        self.bb = pad_filter.Execute(self.bb)
        self.nnunet = pad_filter.Execute(self.nnunet)

    # Method to crop the images to a desired size.
    def crop_image(self, crop=64):
        lssif = sitk.LabelShapeStatisticsImageFilter()
        lssif.Execute(self.mask)
        center = lssif.GetCentroid(1)
        index = self.mask.TransformPhysicalPointToIndex(center)
        lsif = sitk.LabelStatisticsImageFilter()
        lsif.Execute(self.mask, self.mask)
        boundingbox = lsif.GetBoundingBox(1)
        xyzminbounds = np.array([index[0] - crop, index[1] - crop, boundingbox[4]])
        xyzmaxbounds = np.array(self.bb.GetSize()) - np.array(
            [index[0] + crop, index[1] + crop, boundingbox[5]])
        cif = sitk.CropImageFilter()
        cif.SetLowerBoundaryCropSize(xyzminbounds.tolist())
        cif.SetUpperBoundaryCropSize(xyzmaxbounds.tolist())
        self.image = cif.Execute(self.image)
        self.mask = cif.Execute(self.mask)
        self.rb = cif.Execute(self.rb)
        self.bb = cif.Execute(self.bb)
        self.nnunet = cif.Execute(self.nnunet)

    # Method to apply windowing operation on the images for intensity normalization.
    def window(self, width=1600, level=-500):
        iwif = sitk.IntensityWindowingImageFilter()
        iwif.SetWindowMinimum(level - width / 2)
        iwif.SetWindowMaximum(level + width / 2)
        self.image = iwif.Execute(self.image)

    # Method to calculate radiomic features.
    def get_features(self):
        mask_list = [self.mask, self.rb, self.bb, self.nnunet]
        dir_list = ["Full", "Radiomic_Biopsy", "Bounding_Box", "nnUNet"]
        for i in range(len(mask_list)):
            settings = {}
            settings['binWidth'] = 25
            settings['resampledPixelSpacing'] = None
            settings['padDistance'] = 10
            settings['verbose'] = True
            settings['label'] = 1
            # settings['normalize'] = True
            # settings['normalizeScale'] = 500
            settings['voxelArrayShift'] = 0
            extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
            extractor.enableImageTypes(Original={})
            extractor.enableAllFeatures()
            feature = pd.Series(extractor.execute(self.image, mask_list[i]),
                                name=self.patient).to_frame().transpose()
            feature = feature[feature.columns.drop(list(feature.filter(regex='diagnostics')))]
            feature.to_csv(os.path.join(self.path, "Pyradiomics", dir_list[i], self.patient + "_features.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("patients")
    args = parser.parse_args()
    patients = args.patients

    workdir = os.getcwd()

    sample = Radiomics(patients, workdir)

    # dice = pd.read_csv(os.path.join(sample.path, "Dice", "HPUNet", sample.patient + ".csv"))
    # max_idx = dice["Dice"].idxmax(axis=0)
    # max_dice = dice["Mean"][max_idx]

    # os.makedirs(os.path.join(sample.path, "CC"), exist_ok=True)

    os.makedirs(os.path.join(sample.path, "Samples"), exist_ok=True)
    os.makedirs(os.path.join(sample.path, "Samples", sample.patient), exist_ok=True)

    os.makedirs(os.path.join(sample.path, "Dice"), exist_ok=True)
    os.makedirs(os.path.join(sample.path, "Volume"), exist_ok=True)

    os.makedirs(os.path.join(sample.path, "Pyradiomics"), exist_ok=True)
    os.makedirs(os.path.join(sample.path, "Pyradiomics", "Full"), exist_ok=True)
    os.makedirs(os.path.join(sample.path, "Pyradiomics", "Radiomic_Biopsy"), exist_ok=True)
    os.makedirs(os.path.join(sample.path, "Pyradiomics", "Bounding_Box"), exist_ok=True)
    os.makedirs(os.path.join(sample.path, "Pyradiomics", "nnUNet"), exist_ok=True)

    sample.dicom_series_to_nii()

    sample.read_mask()
    sample.read_rb()
    sample.read_nnunet()

    sample.connected_components()
    sample.resample_image(is_label=True)
    sample.resample_image(is_label=False)
    sample.make_bb()

    sample.pad_image()
    sample.crop_image()

    sample.window()

    sample.get_dice()

    sample.write_image()
    sample.write_mask()
    sample.write_rb()
    sample.write_bb()
    sample.write_nnunet()

    sample.get_features()




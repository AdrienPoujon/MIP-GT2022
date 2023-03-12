import matplotlib.pyplot as plt
import nibabel as nib
def main():
    dir = '/data/Datasets/Alzheimers/ADNI1_Annual_2_Yr_3T/ADNI/002_S_0413/MPR____N3__Scaled/2006-05-19_16_17_47.0/I40657/ADNI_002_S_0413_MR_MPR____N3__Scaled_Br_20070216232854688_S14782_I40657.nii'
    img = nib.load(dir)
    data = img.get_data()

    plt.imshow(data[:,:,80],cmap='gray')
    plt.show()
    print("hello")
if __name__ == "__main__":
    main()
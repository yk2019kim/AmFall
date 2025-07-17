# AmFall: WiFi CSI Amplitude-Based Fall Detection Using Denoised Scalograms
AmFall: WiFi CSI Amplitude-Based Fall Detection Using Denoised Scalograms

This is a repository with source code for the paper "AmFall: WiFi CSI Amplitude-Based Fall Detection Using Denoised Scalograms" .

The main goal of this thesis is to explore a novel WiFi CSI amplitude-based fall detection system for non-line-of-sight (NLOS), through-the-wall (TTW), and cross-domain environment with high detection accuracy.

To cope with NLOS/TTW situations, AmFall selectively uses the subcarriers and principal components containing valuable information.
Considering that the continuous wavelet transform (CWT) is suitable for time-frequency analysis of non-stationary signals, such as those generated during falls, AmFall generates a scalogram by applying CWT to the resulting CSI signal with a novel denoising algorithm and extracts the speed information for segmentation.
The segmented scalograms are fed into a deep learning-based classifier.
In this paper, we also suggest a simple yet effective dataset augmentation method to generate multiple scalograms from a single CSI observation.
The proposed system, implemented on commercial WiFi devices, achieves a high fall detection accuracy of 95.28% to 99.67%
in TTW and cross-domain environments, outperforming state-of-the-art WiFi-based methods.

# Files
- AmFall_Github.m : MATLAB (R2024a) code for preprocessing of CSI log file to generate scalogram images. CSI tool for MATLAB/Octave (https://dhalperi.github.io/linux-80211n-csitool/) should be preinstalled on MATLAB to execuit it.

- AmFall_Colab_Github.ipynb : Colab Notebook python code for CNN classification (CNN_SNU1 (LW-CNN), CNN_R18 (ResNet-18), CNN_R34 (ResNet-34)).  
   Image directory structure : image-dir / env-name / user-name / fall or nonfall / image-files (jpg)  
   Ex, csi_AmFall/enva/bh/fall/*.jpg or csi_AmFall/envb/su/nonfall/*.jpg

- b_bh_fall_front3.log : a sample CSI log file

# Execution
1) Install the CSI tool for MATLAB/Octave
2) Use AmFall_Github.m on MATLAB to generate image files from CSI logs
3) Use AmFall_Colab_Github.ipynb on Colab to classify the images

# Cite
Y. Kim, W. S. Jeon, and D. G. Jeong, "AmFall: WiFi CSI Amplitude-Based Fall Detection Using Denoised Scalograms," in IEEE Internet of Things Journal, doi: 10.1109/JIOT.2025.3585401. (Early access as of 16JUL25).

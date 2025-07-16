# AmFall
AmFall: WiFi CSI Amplitude-Based Fall Detection Using Denoised Scalograms

This is a repository with source code for the paper "AmFall: WiFi CSI Amplitude-Based Fall Detection Using Denoised Scalograms" .

The main goal of this thesis is to explore a novel WiFi CSI amplitude-based fall detection system for non-line-of-sight (NLOS), through-the-wall (TTW), and cross-domain environment with high detection accuracy.

To cope with NLOS/TTW situations, AmFall selectively uses the subcarriers and principal components containing valuable information.
Considering that the continuous wavelet transform (CWT) is suitable for time-frequency analysis of non-stationary signals, such as those generated during falls, AmFall generates a scalogram by applying CWT to the resulting CSI signal with a novel denoising algorithm and extracts the speed information for segmentation.
The segmented scalograms are fed into a deep learning-based classifier.
In this paper, we also suggest a simple yet effective dataset augmentation method to generate multiple scalograms from a single CSI observation.
The proposed system, implemented on commercial WiFi devices, achieves a high fall detection accuracy of 95.28% to 99.67%
in TTW and cross-domain environments, outperforming state-of-the-art WiFi-based methods.

# Cite
Y. Kim, W. S. Jeon, and D. G. Jeong, "AmFall: WiFi CSI Amplitude-Based Fall Detection Using Denoised Scalograms," in IEEE Internet of Things Journal, doi: 10.1109/JIOT.2025.3585401. (Early access as of 16JUL25).

# File
AmFall_Github.m : MATLAB (R2024a) code for preprocessing of CSI log file (to generate image file). CSI tool for MATLAB/Octave (https://dhalperi.github.io/linux-80211n-csitool/) should be preinstalled on MATLAB to execuit it.

b_bh_fall_front3.log : a sample CSI log file

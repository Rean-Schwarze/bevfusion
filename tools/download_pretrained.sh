mkdir -p pretrained && \
cd pretrained && \
wget -O bevfusion-det.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained_updated/bevfusion-det.pth && \
wget -O bevfusion-seg.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained_updated/bevfusion-seg.pth && \
wget -O lidar-only-det.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained/lidar-only-det.pth && \
wget -O lidar-only-seg.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained/lidar-only-seg.pth && \
wget -O camera-only-det.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained_updated/camera-only-det.pth && \
wget -O camera-only-seg.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained_updated/camera-only-seg.pth && \
wget -O swint-nuimages-pretrained.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained_updated/swint-nuimages-pretrained.pth
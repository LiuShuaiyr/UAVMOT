# UAVMOT: Multi-Object Tracking Meets Moving UAV 

This work is built on the MCMOT, which adapts to multi-object tracking on moving UAV videos.
You can refer to origin fork [MCMOT](https://github.com/CaptainEven/MCMOT).

##I train

1 Prepare dataset

   Run  gen_dataset_visdrone.py or gen_dataset_UAVDT.py  
   
2 Run train.py

##II test

Run python track.py

The multi-object tracking results saved in **.txt, and evaluate it by the official toolkits.

For visdrone2019 dataset, you can refer to the link(https://github.com/VisDrone/VisDrone2018-MOT-toolkit)

For UAVDT dataset, you can refer to the link(https://github.com/VisDrone/VisDrone2018-MOT-toolkit）

##III demo

Run python demo.py



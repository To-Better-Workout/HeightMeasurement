# HeightMeasurement
./HeightMeasurement/camcalibration/
1. pattern.png를 출력하여 평평한 판에 붙임.
2. 고정된 위치의 카메라 앞에서 출력물을 들고 checkBoard 폴더의 사진처럼 카메라에 대해 checkBoard가 여러방향, 다양한 각도에서 촬영될 수 있도록 함. 이 때 흐리게 촬영된 사진들은 포함하지 않는 게 좋음.(valid_frame.py를 사용해도 좋음)
3. intrinsicMatrix.py 실행, valid_images에 2.에서 실행한 결과물을 넣어서 경로 변경.
4. intrinsic_calibration_result.npz가 잘 저장되었고, 재투영 오차가 0.08mm, 0.1pixel 부근이라면 비교적 정확하게 내재 파라미터가 구해진 것. cali_frame의 예시처럼 결과물이 보여야 함.
5. extrinsic.py 를 실행하여 카메라 정면에 매트를 맞춘 프레임(예: mat-frame.png)에 대해 extrinsix matrix와 재투영 오차 결과를 확인.(reprojection.png처럼 초록색 4개의 점을 manual로 찍음.) 재투영 오차가 0.6 픽셀정도라면 잘 나온 것.

./HeightMeasurement/

6. seg_height_folder.py 에서 intrinsic, extrinsic 경로를 맞추고 실험 프레임에 대해 실험 진행. DeepLab_v3를 이용하여 segmentation을 진행했으나, segmentation 모델은 변경 가능.

* 자세한 사항은 논문을 참고해 주세요.

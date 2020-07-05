import os
import shutil
import mmcv
import os.path as osp
from argparse import ArgumentParser
from mmdet.apis import init_detector, inference_detector


def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # test a video and show the results
    video = mmcv.VideoReader(args.video)
    result_root = "./results"
    frame_dir = osp.join(result_root, 'frame')
    frame_id = 0
    for frame in video:
        result = inference_detector(model, frame)
        model.show_result(frame, result, score_thr=args.score_thr, theme='black',
                          out_file=osp.join(frame_dir, '{:06d}.jpg'.format(frame_id)))
        frame_id += 1

    output_video_path = osp.join(result_root, args.video.split('/')[-1])
    mmcv.frames2video(frame_dir, output_video_path, fourcc='mp4v', filename_tmpl='{:06d}.jpg')
    shutil.rmtree(frame_dir)


if __name__ == '__main__':
    main()

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir', type=str,
                        help='Directory where videos are stored.')
    parser.add_argument('--layout', default='linear', help='Which layout to use to position videos in the canvas')
    parser.add_argument('--how_many', default=10, help='How many videos to use')
    args = parser.parse_args()

    video_paths = os.listdir(args.video_dir)
    video_paths = [os.path.join(args.video_dir, v) for v in video_paths if '.npy.mp4' in v]
    video_paths = sorted(video_paths, key=lambda x: int(os.path.split(x)[1].split('_')[0]))

    how_many_vids = min(len(video_paths), args.how_many)

    if args.layout == 'linear':
        layout = ["0_0"] + ['+'.join(['w'+str(i) for i in range(j)])+'_0' for j in range(1, how_many_vids)]
    elif args.layout == 'grid':
        layout = ["0_0", "w0_0", "w0+w1_0", "w0+w1+w2_0", "0_h0", "w4_h0", "w4+w5_h0", "w4+w5+w6_h0", "0_h0+h4", "w8_h0+h4", "w8+w9_h0+h4", "w8+w9+w10_h0+h4", "0_h0+h4+h8", "w12_h0+h4+h8", "w12+w13_h0+h4+h8", "w12+w13+w14_h0+h4+h8"]

    # read the description from file
    with open(os.path.join(args.video_dir, 'desc.txt'), 'r') as f:
        desc, rank = f.read().splitlines()

    # prepare the ffmpeg command dynamically
    inputs = " ".join(["-i {}".format(v) for v in video_paths])
    input_stream_names = "".join(["[{}:v]".format(i) for i in range(how_many_vids)])
    
    layout = "|".join(layout[:how_many_vids])
    output_file = os.path.join(args.video_dir, 'output.mp4')

    text = '{} (Rank of correct result = {})'.format(desc, rank)

    cmd = "ffmpeg {} -y -filter_complex \"{}xstack=inputs={}:layout={}, pad=iw:ih+40:0:40:blue, fps=25[h];[h]drawtext=font='monospace':text='{}':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10\" -t 00:00:07 {}".format(inputs, input_stream_names, how_many_vids, layout, text, output_file)

    os.system(cmd)



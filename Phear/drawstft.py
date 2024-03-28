import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
import os

for i in tqdm(range(100)):
    filename = "../result/" + str(i).zfill(3) + "/"
    # get file under this directory
    namelist = ["gt", "mixture", "output"]
    filename_gt = filename + "gt.wav"
    filename_mixture = filename + "mixture.wav"
    filename_output = filename + "output.wav"
    filelist = [filename_gt, filename_mixture, filename_output]

    for n, f in enumerate(filelist):
        if os.path.exists(filename + "spec_nw_" + namelist[n] + ".png"):
            continue
        x, sr = librosa.load(f, sr=None)

        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))

        # fig = plt.Figure()
        # canvas = FigureCanvas(fig)
        # ax = fig.add_subplot(111)
        # p = librosa.display.specshow(Xdb, ax=ax, y_axis='log', x_axis='time')
        # fig.savefig(filename+'spec_'+namelist[n]+'.png')
        plt.figure()
        plt.axis("off")  # no axis
        plt.axes(
            [0.0, 0.0, 1.0, 1.0], frameon=False, xticks=[], yticks=[]
        )  # Remove the white edge
        p = librosa.display.specshow(Xdb, y_axis="log")
        plt.savefig(
            filename + "spec_nw_" + namelist[n] + ".png", bbox_inches=None, pad_inches=0
        )

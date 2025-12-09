import matplotlib.pyplot as plot
import numpy as np
import os
import sys
import argparse
import re

def find_values(fp, channel) :
    found_header = False
    field = 0.0
    for line in fp :
        if re.match(".*MAGNETIC Z FIELD =.*", line) :
            toks = [tok.strip() for tok in line.split("=")]
            second = [tok.strip() for tok in toks[1].split(" ") if len(tok) != 0]
            field = float(second[0])
            continue
        if re.match("\\s*K-DEPENDENT SCATTERING LENGTHS/VOLUMES/HYPERVOLUMES FOR CHANNELS WITH LOW KINETIC ENERGY\\s*", line) :
            found_header = True
            continue
        if found_header :
            stripped = line.strip()
            if len(stripped) == 0 :
                found_header = False
                continue
            toks = [tok for tok in stripped.split(" ") if len(tok) != 0]
            curr_channel = 0
            try :
                curr_channel = int(toks[0])
            except ValueError :
                continue
            if curr_channel == channel :
                found_header = False
                yield (field, int(toks[0]), int(toks[1]), int(toks[2]), float(toks[3]), complex(float(toks[4]), float(toks[5])))
    return

def plot_file(fname, channel) :
    data = []
    with open(fname, "r") as fp :
        data = list(find_values(fp, channel))
    print(data)
    X = [d[0] for d in data]
    Y1 = [d[4] for d in data]
    Y2 = [d[5].real for d in data]
    Y3 = [d[5].imag for d in data]
    Y4 = [abs(d[5]) for d in data]
    figure1 = plot.figure()
    plot.title("Wvec")
    plot.plot(X, Y1, label = "Wvec")
    figure2 = plot.figure()
    plot.title("re(A)")
    plot.plot(X, Y2, label = "re(A)")
    figure3 = plot.figure()
    plot.title("im(A)")
    plot.plot(X, Y3, label = "im(A)")
    figure4 = plot.figure()
    plot.title("|A|")
    plot.plot(X, Y4, label = "|A|")
    plot.show()

def main() :
    parser = argparse.ArgumentParser("Plot the contents of a molscat output.")
    parser.add_argument("--channel", help="The channel to plot", type = int, default = 1)
    parser.add_argument("file", help = "The file to plot")

    args = vars(parser.parse_args())

    plot_file(args["file"], args["channel"])

if __name__ == "__main__" :
    main()

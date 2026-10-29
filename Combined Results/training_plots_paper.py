# import json
# import numpy as np
# import matplotlib.pyplot as plt

# names = ["4q_2l", "6q_4l", "8q_6l", "10q_8l", "12q_10l", "14q_10l"]
# lrs = ["lr_0.001", "lr_0.01", "lr_0.1", "lr_0.5"]
# styles = [":", "-.", "--", "-"]  # one line style per learning rate
# steps = np.arange(1, 51)

# # Average of the 5 runs
# def avg(d, key):
#     return np.mean([d[f"{key}{i}"] for i in range(1, 6)], axis=0)

# fig, axes = plt.subplots(1, 6, figsize=(30, 6))

# for ax, name in zip(axes, names):
#     path = f"./QAOA/Cycle/{name}.json"
#     with open(path) as f:
#         data = json.load(f)

#     for opt, color in zip(["sgd", "adam", "qng"], ["C0", "C1", "C2"]):
#         for lr, ls in zip(lrs, styles):
#             ax.plot(steps, avg(data[opt][lr], "loss"), color=color, linestyle=ls,
#                     label=f"{opt.upper()} {lr[3:]}")

#     ax.plot(steps, avg(data["polyak_gd"], "best_loss"), color="C3", linewidth=2.5, label="Polyak GD")
#     ax.plot(steps, avg(data["polyak_qng"], "best_loss"), color="C4", linewidth=2.5, label="Polyak QNG")

#     num_qubits = data["num_qubits"]
#     e_min = -num_qubits if num_qubits % 2 == 0 else -(num_qubits - 2)
#     ax.axhline(e_min, color="black", linestyle="--", label="GSE")

#     ax.set_title(name)
#     ax.set_xlabel("Number of steps")
#     ax.set_ylabel("Loss")
#     ax.legend(fontsize=7, ncol=2)

# plt.tight_layout()
# plt.savefig("loss_plot.png", dpi=150)
# plt.show()



import json
import numpy as np

# LaTeX macro names can only contain letters, so: Six, Ten, Fourteen and a, b, c, d
files = {"Six": "6q_4l", "Ten": "10q_8l", "Fourteen": "14q_10l"}
lrs = {"a": "lr_0.001", "b": "lr_0.01", "c": "lr_0.1", "d": "lr_0.5"}

# Average of the 5 runs
def avg(d, key):
    return np.mean([d[f"{key}{i}"] for i in range(1, 6)], axis=0)

# ---------- 1. Data objects ----------
for prefix, name in files.items():
    with open(f"./QAOA/Cycle/{name}.json") as f:
        data = json.load(f)

    lines = {}
    for opt in ["sgd", "adam", "qng"]:
        for suffix, lr in lrs.items():
            lines[opt.upper() + suffix] = avg(data[opt][lr], "loss")
    lines["PolyakGD"] = avg(data["polyak_gd"], "best_loss")
    lines["PolyakQNG"] = avg(data["polyak_qng"], "best_loss")

    n = data["num_qubits"]
    e_min = -n if n % 2 == 0 else -(n - 2)
    ymin = np.floor(min(e_min, min(v.min() for v in lines.values())))
    ymax = np.ceil(max(v.max() for v in lines.values()))
    yticks = np.linspace(ymin, ymax, 6)

    print(f"% {name}")
    for key, values in lines.items():
        print(f"\\def\\{prefix}{key}{{" + ",".join(f"{v:.3f}" for v in values) + "}")
    print(f"\\def\\{prefix}GSE{{{e_min}}}")
    print(f"\\def\\{prefix}ymin{{{ymin:.0f}}}")
    print(f"\\def\\{prefix}ymax{{{ymax:.0f}}}")
    print(f"\\def\\{prefix}yticks{{" + ",".join(f"{t:.2f}" for t in yticks) + "}")
    print()

# ---------- 2. Plotting code ----------
print(r"""\makeatletter
\newcount\stepnum \newcount\mk
% Markers
\def\mcirc{\circle*{1.6}}
\def\msq{\polygon*(-0.75,-0.75)(0.75,-0.75)(0.75,0.75)(-0.75,0.75)}
\def\mtri{\polygon*(-0.9,-0.7)(0.9,-0.7)(0,0.9)}

% \plotline{data name}{colour}{thickness}{marker}  (marker every 5 steps)
\newcommand\plotline[4]{%
  \begingroup
  \color{#2}\linethickness{#3}%
  \expandafter\let\expandafter\data\csname\prefix#1\endcsname
  \stepnum=0 \mk=0 \def\coords{}%
  \@for\val:=\data\do{%
    \edef\px{\fpeval{round(\xs*\the\stepnum,2)}}%
    \edef\py{\fpeval{round(\ys*((\val)-(\ymn)),2)}}%
    \edef\coords{\coords(\px,\py)}%
    \advance\mk by 1 \ifnum\mk=5 \put(\px,\py){#4}\mk=0 \fi
    \advance\stepnum by 1
  }%
  \expandafter\polyline\coords
  \endgroup}

\def\pw{44} % panel width
\def\ph{30} % panel height
\def\lw{1pt}   % line width for SGD/Adam/QNG
\def\plw{2.5pt} % line width for Polyak

% \panel{Six/Ten/Fourteen}{title}
\newcommand\panel[2]{%
  \def\prefix{#1}%
  \edef\ymn{\csname#1ymin\endcsname}%
  \edef\ymx{\csname#1ymax\endcsname}%
  \edef\xs{\fpeval{\pw/49}}%
  \edef\ys{\fpeval{\ph/((\ymx)-(\ymn))}}%
  \put(0,0){\framebox(\pw,\ph){}}%
  \put(\fpeval{\pw/2},\fpeval{\ph+2.5}){\makebox(0,0){\small #2}}%
  % y ticks
  \expandafter\let\expandafter\yt\csname#1yticks\endcsname
  \@for\val:=\yt\do{%
    \edef\py{\fpeval{\ys*((\val)-(\ymn))}}%
    \put(0,\py){\line(1,0){1}}%
    \put(-0.8,\py){\makebox(0,0)[r]{\tiny\val}}}%
  % x ticks
  \@for\val:=1,10,20,30,40,50\do{%
    \edef\px{\fpeval{\xs*(\val-1)}}%
    \put(\px,0){\line(0,1){1}}%
    \put(\px,-2.2){\makebox(0,0){\tiny\val}}}%
  % curves
  \plotline{SGDa}{blue}{\lw}{}\plotline{SGDb}{blue}{\lw}{\mcirc}%
  \plotline{SGDc}{blue}{\lw}{\msq}\plotline{SGDd}{blue}{\lw}{\mtri}%
  \plotline{ADAMa}{orange}{\lw}{}\plotline{ADAMb}{orange}{\lw}{\mcirc}%
  \plotline{ADAMc}{orange}{\lw}{\msq}\plotline{ADAMd}{orange}{\lw}{\mtri}%
  \plotline{QNGa}{green!60!black}{\lw}{}\plotline{QNGb}{green!60!black}{\lw}{\mcirc}%
  \plotline{QNGc}{green!60!black}{\lw}{\msq}\plotline{QNGd}{green!60!black}{\lw}{\mtri}%
  \plotline{PolyakGD}{red}{\plw}{}\plotline{PolyakQNG}{violet}{\plw}{}%
  % GSE (dashed)
  \put(0,\fpeval{\ys*((\csname#1GSE\endcsname)-(\ymn))}){%
    \linethickness{1.2pt}\multiput(0,0)(2,0){22}{\line(1,0){1}}}%
}
\makeatother

% Legend helpers: \sample{colour}{thickness}{marker}, \legitem{x}{y}{sample}{text}
\newcommand\sample[3]{\put(0,0){\color{#1}\linethickness{#2}\put(0,0){\line(1,0){6}}\put(3,0){#3}}}
\newcommand\legitem[4]{\put(#1,#2){#3\put(7.5,0){\makebox(0,0)[l]{\scriptsize #4}}}}

\setlength{\unitlength}{\dimexpr\linewidth/159\relax}
\begin{picture}(159,52)(-9,-18)
  \put(0,0){\panel{Six}{6 qubits}}
  \put(53,0){\panel{Ten}{10 qubits}}
  \put(106,0){\panel{Fourteen}{14 qubits}}
  % Legend row 1: optimizers
  \legitem{0}{-8}{\sample{blue}{\lw}{}}{SGD}
  \legitem{25}{-8}{\sample{orange}{\lw}{}}{Adam}
  \legitem{50}{-8}{\sample{green!60!black}{\lw}{}}{QNG}
  \legitem{75}{-8}{\sample{red}{\plw}{}}{Polyak GD}
  \legitem{100}{-8}{\sample{violet}{\plw}{}}{Polyak QNG}
  \legitem{125}{-8}{\put(0,0){\linethickness{1.2pt}\multiput(0,0)(2,0){3}{\line(1,0){1}}}}{GSE}
  % Legend row 2: step sizes
  \legitem{0}{-14}{\sample{black}{\lw}{}}{step size 0.001}
  \legitem{38}{-14}{\sample{black}{\lw}{\mcirc}}{step size 0.01}
  \legitem{75}{-14}{\sample{black}{\lw}{\msq}}{step size 0.1}
  \legitem{112}{-14}{\sample{black}{\lw}{\mtri}}{step size 0.5}
\end{picture}""")
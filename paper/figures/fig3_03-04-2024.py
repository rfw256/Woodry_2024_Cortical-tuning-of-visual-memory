# %% [markdown]
# # 0. Load data and dependencies

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
import scipy.signal as sp
from scipy.interpolate import griddata
import os
import warnings
from scipy.special import iv
from scipy.optimize import curve_fit
import matplotlib
from interstellar import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

warnings.filterwarnings('ignore')

# Set path to dataframe directory
df_dir = "../../data/dataframes/"

# %% [markdown]
# Load example subject data.

# %%
angdist_df_subj127 = pd.read_csv(os.path.join(df_dir, 'angdist_df.tsv'), sep = '\t', index_col = 1)

# %% [markdown]
# # 1. Figure 3 
# ### Panel: Single-trial estimates in visual space
# 
# Plot single-trial estimates in visual space, averaged for each target position. From example subject 127. 

# %%
r = 2
t = 'perception'
stim_ecc = 7

df = angdist_df_subj127.query("roi == @r & task == @t")
# df = df[np.abs(stim_ecc - df.eccen) <= df.sigma]
df = df[df.vexpl >= 0.1]
df.beta = (df.beta - np.nanmean(df.beta)) / np.nanstd(df.beta)

for c in df.condition.unique():
    pos = df[df.condition == c]
    pos = pos.groupby(['condition', 'x', 'y'], as_index = False).mean()
    plt.scatter(pos.x, -pos.y, c = pos.beta, alpha = 1, s = 10, cmap = 'RdBu_r', vmin = -3, vmax = 3)
    plt.axis('equal')
    
    x, y = pol2cart(stim_ecc, np.radians(pos.stim_angle.values[0]))
    print(pos.stim_angle.values[0])
    
    circle = plt.Circle((0, 0), radius = stim_ecc, fill = False, lw = 4, linestyle = ":")
    plt.gca().add_patch(circle)
    
    plt.tick_params(which = 'major', labelsize = 30)
    
    plt.scatter(x, y, s = 2000, marker = "x", color = 'black', linewidths = 7)
    plt.scatter(x, y, s = 500, marker = "o", color = 'black')
    
    plt.vlines(0, -12, 12)
    plt.hlines(0, -13, 13)
    
    plt.gcf().set_size_inches([12, 10])
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.yticks([-10, -5, 0, 5, 10])
    
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-3, vmax=3))
    cb = plt.colorbar(sm)
    cb.ax.tick_params(labelsize=30)
    
    plt.show()
    
    

# %% [markdown]
# ### Panel: Target-aligned average in visual space
# 
# Load example subject's data.

# %%
angdist_df_subj136 = pd.read_csv(os.path.join(df_dir, 'subj136_angdist_df.tsv'), sep = '\t', index_col = 1)

# %% [markdown]
# Use custom functions found in interstellar.py to rotate, interpolate, and average estimates in visual space. For example subject 136

# %%
degs_lim = 12
interp_df = []
rot_data = []
roi = 3

for task in angdist_df_subj136.task.unique():
    df = angdist_df_subj136[angdist_df_subj136.task == task]

    for c in df.condition.unique():
        data = df[df.roi == roi]
        data = data[data.condition == c]
        data = data[data.vexpl >= 0.1]
        # data.beta = data.beta * data.vexpl
        idf, rd = interpolate_activity(data, normalize=True, offset = 90)
        interp_df.append(idf)
        rot_data.append(rd)

v1 = pd.concat(interp_df).reset_index(drop=True)
v1.activity_map_rot[np.isnan(v1.activity_map_rot)] = 0
v1_rot = pd.concat(rot_data).reset_index(drop=True)
ax_data = avg_activity_figure(v1, vmax = 2, degs_lim = degs_lim, stim_radius = 7, 
                              tasks = ['perception', 'wm', 'ltm'], colors = ['b', 'g', 'orange'])
plt.gcf().set_size_inches([15, 5])



import matplotlib as mpl
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import itertools
from pylab import *
from matplotlib.colors import LogNorm
import pylab
import math

from scipy.signal import find_peaks

from scipy.optimize import curve_fit

from matplotlib.colors import LinearSegmentedColormap


from matplotlib import rcParams
from matplotlib import font_manager as fm
from matplotlib.ticker import AutoMinorLocator

from scipy import polyval, polyfit

#mpl.colormaps.register(LinearSegmentedColormap('BlueRed2', cdict2))
colors = [(0, 0, 0), (1, 0.6, 0), (0, 0, 1)]  # R -> G -> B
n_bin = 1000  # Discretizes the interpolation into bins
cmap_name = 'my_list'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

# Plot transition energies based on the many-body model:
#from nmEnergyCalc import makePlot as mp

figurefile='PLE maps'
spectrafile='PLE spectra'


rcParams['font.family'] = 'Comic Sans'
myFont = fm.FontProperties(family='Arial', size=15)
myFont2 = fm.FontProperties(family='Arial', size=12.5)
myFont3 = fm.FontProperties(family='Arial', size=18)
myFont4 = fm.FontProperties(family='Arial', size=22)


interp_methods = ['none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

interp_methods_new = [string[:3] for string in interp_methods]
#print(interp_methods_new)

def gaussianf(x, A, x0, sigma):

    return A * np.exp(-np.power(x - x0, 2) / (2 * np.power(sigma, 2)))

def gaussian_fit(x_min, x_max):

    n_min = np.argmin(ExPLE, x_min, axis=0)
    n_max = np.argmax(ExPLE, x_max, axis=0)
    nx_min_grounded = np.argmin(Em, x_min, axis=0)
    nx_max_grounded = np.argmin(Em, x_max, axis=0)
    x_grounded = Em[nx_min_grounded:nx_max_grounded]
    y = data[:,n_min:n_max]
    x0 = 0.5* (x_min + x_max)



def FindASite(Excitation, Emission, data):
    EmPosition = (Emission - Em1) / EmStep - 2
    ExPosition = (Excitation - Ex1) / ExStep - 2
    #PickedIntensity = data[int(EmPosition), int(ExPosition)]
    PickedIntensity = data[int(ExPosition), int(EmPosition)]

    return PickedIntensity

def FindASite_S22(Excitation, Emission, data):
    EmPosition = (Emission - Em1) / EmStep
    ExPosition = (Excitation - 500) / ExStep
    #PickedIntensity = data[int(EmPosition), int(ExPosition)]
    PickedIntensity = data[int(ExPosition), int(EmPosition)]

    return PickedIntensity

def SubBandEnergies(Excitation=565, Emission=970):
    Em_eV = 1240.0 / Emission
    Ex_eV = 1240.0 / Excitation

    return Em_eV, Ex_eV

def PlotPLE(Emission, data, Color, Ex_peak_estimate, unfilled=True):
    #f = np.zeros((len(ExPLE_s22), 2))
    EmPosition = (Emission - Em1) / EmStep
    estimate_position = (Ex_peak_estimate - Ex11) / ExStep

    Integration = data[:,int(EmPosition-1):int(EmPosition+1)].sum(axis=1)
    Integration /= max(Integration)
    Integration_diff = np.diff(Integration, n=1)

    #Integration = data[:, :].sum(axis=1)
    #Integration = data[int(EmPosition-1):int(EmPosition+1),:].sum(axis=0)
    #PLEprofile = Integration[22:41]      # [0:61] for 500-800      # [0:91] for 330-800    # Integration[0:90] (for 350-800)
    #NormalizedPLE = PLEprofile / max(PLEprofile)
    #f[:,0] = Emission
    #f[:,1] = NormalizedPLE

    peak_count = 0
    peak_idx = []

    for id_x in np.arange(10, len(Integration_diff)-10, 1):
        if np.sign(Integration_diff[id_x]) - np.sign(Integration_diff[id_x+1]) == 2:
            peak_idx.append(id_x)
            peak_count += 1
        elif Integration_diff[id_x] == 0:
            peak_idx.append(id_x)
            peak_count += 1

    peak_idx_required, n = min((n, peak_idx_required) for peak_idx_required, n in enumerate(np.abs(np.array(peak_idx) - estimate_position)))
    peak_wavelength = ExPLE_s22[peak_idx[n]]

    #print(peak_idx)
    print('estimated no. of peaks:', peak_count)
    #print(peak_idx_required)
    print('Ex wavelength of the acquired peak:', peak_wavelength, 'nm')



    if unfilled:
        ax2.plot(ExPLE_s22, Integration/Integration[peak_idx[n]+1], '-', color=Color, linewidth=2.5,
                 label='unfilled {0}-{1} nm'.format(str(Emission - 5), str(Emission + 5)))      # NormalizedPLE * ScaleFactor + Offset
        #ax2.scatter(ExPLE_s22[int(peak)], NormalizedPLE[int(peak)])

    else:
        ax2.plot(ExPLE_s22, Integration/Integration[peak_idx[n]+1], '-', color=Color, linewidth=2.5,
                 label='HgTe-filled {0}-{1} nm'.format(str(Emission - 5), str(Emission + 5)))       # NormalizedPLE * ScaleFactor * 0.7 + Offset



    plt.xlim(peak_wavelength-30, peak_wavelength+30)
    plt.ylim(0.2, 1.2)

def PlotPL(Excitation, data, color, Em_peak_estimate, unfilled=True):
    #f = np.zeros((len(Em), 2))
    ExPosition = (Excitation - 500) / ExStep
    estimate_position = (Em_peak_estimate - Em1) / EmStep

    Integration = data[int(ExPosition-1):int(ExPosition+1),:].sum(axis=0)
    Integration /= max(Integration)
    Integration_diff = np.diff(Integration, n=1)

    #Integration = data[:, :].sum(axis=1)
    #Integration = data[int(EmPosition-1):int(EmPosition+1),:].sum(axis=0)
    #PLspectrum = Integration[20:60]    # Integration[0:90] (for 350-800), Integration[0:101] (for 330-800)
    #NormalizedPL = PLspectrum / max(PLspectrum)
    #f[:,0] = Emission
    #f[:,1] = NormalizedPL

    peak_count = 0
    peak_idx = []

    for id_x in np.arange(10, len(Integration_diff) - 10, 1):
        if np.sign(Integration_diff[id_x]) - np.sign(Integration_diff[id_x + 1]) == 2:
            peak_idx.append(id_x)
            peak_count += 1
        elif Integration_diff[id_x] == 0:
            peak_idx.append(id_x)
            peak_count += 1

    peak_idx_required, n = min(
        (n, peak_idx_required) for peak_idx_required, n in enumerate(np.abs(np.array(peak_idx) - estimate_position)))
    peak_wavelength = EmPL[peak_idx[n]]

    # print(peak_idx)
    print('estimated no. of peaks:', peak_count)
    # print(peak_idx_required)
    print('Em wavelength of the acquired peak:', peak_wavelength, 'nm')

    if unfilled:
        ax2.plot(EmPL, Integration/Integration[peak_idx[n]+1], '--', color=color, linewidth=2.5,
                 label='unfilled {0}-{1} nm'.format(str(Excitation - 5), str(Excitation + 5)))      # NormalizedPL * ScaleFactor + Offset + 0.14

    else:
        ax2.plot(EmPL, Integration/Integration[peak_idx[n]+1], '--', color=color, linewidth=2.5,
                 label='HgTe-filled {0}-{1} nm'.format(str(Excitation - 5), str(Excitation + 5)))       # NormalizedPL * ScaleFactor + Offset + 0.14



    #return NormalizedPL
    plt.xlim(peak_wavelength - 30, peak_wavelength + 30)
    plt.ylim(0.2, 1.2)

# Input Lambda1050 Absorbance data:
def PlotAbs(AbsFile):

    wavelength = []
    Abs = []
    with open(AbsFile) as f:
        for line in itertools.islice(f, 2, 1042):
            d1 = line.split(',')[0]
            d2 = line.split(',')[1]
            wavelength.append(d1)
            Abs.append(d2)

            for i in range(len(wavelength)):
                wavelength[i] = float(wavelength[i])
                Abs[i] = float(Abs[i])

    fig, ax = plt.subplots()
    ax.plot(wavelength, Abs, '--', color='blue', label='985 nm')

    return Abs

def filter_negative_signal(d, correct_exvariation=True):

    range_of_raw = len(d[:,0])
    range_of_column = len(d[0,:])

    for i in range(range_of_raw):
        for j in range(range_of_column):

            if correct_exvariation:
                d[i,j] -= d[i,0] - (d[i,0]-d[i,-1]) * j / range_of_column   # assuming tht background signal varies linearly with Emission wavelength

            if d[i,j] < 0.1:    # 0.35:
                d[i,j] = 0.1    # 0.35

            #elif d[i,j] > 10.0:
             #   d[i,j] = 10.0

    #fig, ax3 = plt.subplots(figsize=(8, 7))
    #bg_start = d[:,0]
    #bg_end = d[:,-1]
    #ax3.plot(ExPLE_s22, bg_start, ExPLE_s22, bg_end)

    return d

def bg_profile(d):

    ax3 = fig2.add_subplot(211)
    bg_start = d[:, 0]
    bg_end = d[:, -1]
    ax3.plot(ExPLE_s22, bg_start, label='bg @ 850 nm')
    ax3.plot(ExPLE_s22, bg_end, label='bg @ 1350 nm')

def save_spectra(Excitation, data1, data2):

    ExPosition = (Excitation - Ex11) / ExStep
    spectrum_unfilled = data1[int(ExPosition),:]
    spectrum_filled = data2[int(ExPosition),:]
    saved_spectrum = np.column_stack((spectrum_unfilled, spectrum_filled))
    # print('transient at 0.8-1.5 ps:', transient_vs_wavelength)
    Result = location + 'CoMo76 semi PL spectra' + '.txt'
    pylab.savetxt(Result, saved_spectrum, fmt='%0.9f', delimiter='   ')

    return ExPosition

def round_up(raw_val):
#raw_val = 1122.6
    decimals = 0
    round_up_val = 5 * (math.ceil((raw_val/5) * 10**decimals) / 10**decimals)
    if raw_val % 5 < 2.5:
        round_up_val += -5

    else: round_up_val = round_up_val

    return int(round_up_val)

Em1 = 850       #850
Em2 = 1350      #1350
Ex1 = 330       #330
Ex11 = 500      # 350      # where no graph combination
Ex2 = 800

Ex_start = 330
Ex_stop = 750
Em_start = 850
Em_stop = 1350
#Ex_range = np.arange(Ex_start, Ex_stop, 5)
#Ex_no = Ex_range.shape[0]
#Ex_steps = int(Ex_no/10)
#Ex_positions = np.arange(0, Ex_no, Ex_steps)
#Ex_labels = Ex_range[::Ex_steps]
#Ex_labels = [round(y, 50) for y in Ex_labels]

ExStep = 5     #5
EmStep = 5     #5
Em = np.arange(Em1, Em2, EmStep)
#Em1 = np.arange(Em1, int(Em2-5), EmStep)
Ex = np.arange(Ex1, Ex2, ExStep)

EM, EX = np.meshgrid(Em, Ex, sparse=True)

# Input S22_S11 PLE contour files:
location = 'C:/Users/Eric Hu/TEAS data plot/TEASanalysis/'
file1 = location + 'Unfilled CoMo76 semi PLE contour' + '.csv'      # Unfilled CoMo76 semi PLE contour / unfilled semi gelatin 350 to 800
                                       # /unfilled raw CHASM PLE contour
data = loadtxt(file1, delimiter=',')
data1 = np.transpose(data)

file2 = location + 'HgTe-filled CoMo76 semi PLE contour' + '.csv'      # HgTe-filled CoMo76 semi PLE contour / HgTefilled semi gelatin 350 to 800
                                    # /HgTe CHASM 0.5wtSC PLE contour
data2 = loadtxt(file2, delimiter=',')
data22 = np.transpose(data2)

# Input S33_S11 contour files:
fileS1 = location + 'Unfilled CoMo76 semi S33 PLE contour' + '.csv'    # Unfilled HiPco C3 S33 PLE contour
                                                                    # /unfilled raw CHASM S33 PLE contour
dataS = loadtxt(fileS1, delimiter=',')
dataS1 = np.transpose(dataS)

fileS2 = location + 'HgTe-filled CoMo76 semi S33 PLE contour' + '.csv'    # Unfilled HiPco C2 S33 PLE contour
dataS2 = loadtxt(fileS2, delimiter=',')
dataS22 = np.transpose(dataS2)


NumberOfraws = len(data1[:,1])
NumberOfColumns = len(data1[1,:])
#print(NumberOfraws, NumberOfColumns)

ExPLE = np.arange(350, 805, 5)
ExPLE_s22 = np.arange(500, 805, 5)
ExPLE_s33 = np.arange(330, 505, 5)
EmPL = np.arange(850,1355,5)

s22_s33 = True
correct_exvariation = True

Max0 = max(data[0,:])
for i in np.arange(1, NumberOfraws, 1):
    Max1 = max(data[i,:])
    if Max1 > Max0:
        Max0 = Max1

#NormalizedPL = data1 / Max0
NormalizedPL1 = data1 / FindASite_S22(round_up(502), round_up(1122), data1)
NormalizedPL2 = data22 / FindASite_S22(round_up(502), round_up(1122), data22)

NormalizedPL_S1 = dataS1 / FindASite(round_up(502), round_up(1122), dataS1)
NormalizedPL_S2 = dataS22 / FindASite(round_up(502), round_up(1122), dataS22)

#save PL spectra:
# E = save_spectra(650, data1, data22)    #NormalizedPL1, NormalizedPL2)

# Merge S22_S11 and S33_S11 maps
#rows = len(NormalizedPL1[:,0]) + len(NormalizedPL_S1[:,0])
#columns = len(NormalizedPL1[0,:])
#Map = np.zero(rows, columns)
NormalizedPLCombined1_raw = np.concatenate((NormalizedPL_S1, NormalizedPL1), axis=0)
NormalizedPLCombined2_raw = np.concatenate((NormalizedPL_S2, NormalizedPL2), axis=0)
NormalizedPLCombined1 = filter_negative_signal(NormalizedPLCombined1_raw, correct_exvariation)
NormalizedPLCombined2 = filter_negative_signal(NormalizedPLCombined2_raw, correct_exvariation)


EmEnergy1_eV = 1240.0 / Em1
EmEnergy2_eV = 1240.0 / Em2
ExEnergy1_eV = 1240.0 / Ex1
ExEnergy2_eV = 1240.0 / Ex2


# Transition energies calculated by the empirical model:
Chirality = ['(6,5)', '(7,5)', '(7,6)', '(9,4)', '(8,4)', '(10,2)', '(8,6)', '(8,3)', '(9,5)',
             '(10,3)', '(11,1)', '(8,7)', '(6,4)']  #'(5,4)'     # '(6,4)',
S11 = [976, 1024, 1120, 1101, 1111, 1053, 1173, 952, 1241, 1249, 1265, 1265, 873]   # 835       # 873,
S22 = [566, 645, 648, 722, 589, 737, 718, 665, 672, 632, 610, 728, 578]     # 483      # 578,

empirical_energies = np.concatenate((Chirality, S11, S22), axis=0).reshape(3,len(S11))

# Input Cary60 absorbance files:
Result = "C:/Users/Eric Hu/TEAS data plot/TEASanalysis/21-02-12 SteadyState Absorbance/file1.txt"
wavelength = []
Abs = []
#f1='C:/Users/Eric Hu/TEAS data plot/TEASanalysis/21-02-12 SteadyState Absorbance/CoMoCAT76 C1 col 1_1(1) 1.5 wtSDS.csv'

# /21-3-2 Lambda 1050/HgTe-filled CHASM 5wtSC.Sample.csv
with open('C:/Users/Eric Hu/TEAS data plot/TEASanalysis/21-02-12 SteadyState Absorbance/HgTe-filled CHASM C3 col11 2 wt SDS.csv') as f:
    #Col1 = f[0,2:912]
    #Col2 = f[1,2:912]
    # /21-02-12 SteadyState Absorbance/HgTe-filled CHASM C2 (6,5) more than 1 wt SDS.csv'
    for line in itertools.islice(f, 2, 912):       # 912 for Cary60   1102 for Lambda1050
        d1 = line.split(',')[0]
        d2 = line.split(',')[1]
        #print(line)
        #d = loadtxt(line, delimiter=',')
        wavelength.append(float(d1))
        Abs.append(float(d2))
        #wave = float(wavelength)
        #A = float(Abs)
        #pylab.savetext(Result,line,fmt='%0.9f', delimiter='    ')

#for i in range(len(wavelength)):
 #   wavelength[i] = float(wavelength[i])
  #  Abs[i] = float(Abs[i])


# Input Lambda1050 Absorbance data:
#AbsFile1 = 'C:/Users/Eric Hu/TEAS data plot/TEASanalysis/21-3-2 Lambda 1050/HgTe-filled CoMoCAT76 metallic.Sample.csv'
#AbsFile2 = 'C:/Users/Eric Hu/TEAS data plot/TEASanalysis/21-3-2 Lambda 1050/unfilled CoMoCAT76 metallic.Sample.csv'

#d = PlotAbs(AbsFile1)
#d2 = PlotAbs(AbsFile2)

ScaleFactor = 0.25
ScaleFactor2 = 0.50
Offset = 0.4

# Fig1
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,2,1)

levels = np.arange(0.5, 20, 0.8)
if s22_s33:
    im = ax.imshow(NormalizedPLCombined1, interpolation='bilinear', origin='lower',
                   cmap=cmap, extent=(Em1, Em2, Ex1, Ex2))    # (EM, EM, EX, EX)) cm.rainbow
    CS = ax.contour(NormalizedPLCombined1, levels, origin='lower', cmap='flag',
                    linewidths=0.02, extent=(Em1, Em2, Ex1, Ex2))
else:
    im = ax.imshow(filter_negative_signal(NormalizedPL1, correct_exvariation), interpolation='bilinear', origin='lower',
                   cmap=cmap, extent=(Em1, Em2, Ex11, Ex2))
    CS = ax.contour(NormalizedPL1, levels, origin='lower', cmap='flag',
                    linewidths=0.02, extent=(Em1, Em2, Ex11, Ex2))


#plt.xtick(np.arange(EmEnergy1_eV, EmEnergy2_eV, 0.15))
#ax.set_aspect('equal')


#CS = ax.contour(NormalizedPLCombined1, levels, origin='lower', cmap='flag',
#                linewidths=0.2, extent=(Em1, Em2, Ex1, Ex2))

ax.tick_params(direction='out', length=5, width=1.5, colors='black',
               grid_color='black', grid_alpha=1, labelsize=12)

for side in ['bottom', 'left', 'top', 'right']:
    ax.spines[side].set_linewidth(1.2)

ax.xaxis.set_major_locator(plt.MultipleLocator(100))
ax.yaxis.set_major_locator(plt.MultipleLocator(50))
ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
ax.yaxis.set_minor_locator(plt.MultipleLocator(25))
#ax.yaxis.set_minor_formatter(ScalarFormatter("%.3f"))

#plt.xticks(fontproperties=myFont)
#plt.yticks(fontproperties=myFont)   #Ex_positions, Ex_labels)
#plt.rcParams["axes.linewidth"]  = 2.0

ax.set_aspect(2)
# We can still add a colorbar for the image, too.
# CBI = fig.colorbar(im, orientation='vertical', shrink=0.8)

#ax.set_autoscale_on(True)
#ax.annotate('', xy=(1.2,2.0), xycoords='data', xytext=('Em','Ex'), textcoords='axes fraction')
#plt.annotate('{}'.format('(6,5)'), (SubBandEnergies(565, 970)), arrowprops=dict(color='white', arrowstyle="->",
                                                                                       #connectionstyle="angle,angleA=0,angleB=-100,rad=20", alpha=0.9))


   ### ax.text(S11[i], S22[i], '.', fontsize=30, color='black')
    #plt.annotate('{}'.format(Chirality[i]), (S11[i], S22[i]), color='white',
                 #arrowprops=dict(color='white', arrowstyle='->',
                                 #connectionstyle="angle,angleA=0,angleB=20,rad=3", alpha=0.9))
    #plt.annotate('{}'.format(Chirality[i]), (SubBandEnergies(S22[i], S11[i])), color='white', arrowprops=dict(color='white', arrowstyle="->",
                                                                                       #connectionstyle="angle,angleA=0,angleB=-120,rad=25", alpha=0.7))

#ax.set_title('PLE Contour, Ex(500-800)/Em(850-1350)', color='r', fontproperties=myFont)

# plot the Eii energies of SWCNTs in a solution state
#S33_S11_chirality = ['(7,6)', '(8,6)']
#S33_S11_ex_unfilled = [372.1, 371.6]   # 337.4, 347,
#S33_S11_em_unfilled = [1125.9, 1184.4]   # 1026.3, 994.4,
#S22_S11_ex_unfilled = [647.7, 585.7, 721.6, 645.5, 570.3, 671.8, 733.9, 716.3]
#S22_S11_em_unfilled = [1126.2, 1121.3, 1120.8, 1028, 989.9, 955, 1042.1, 1180.7]

for i in range(empirical_energies.shape[1]):
    ax.text(int(empirical_energies[1,i])-10, int(empirical_energies[2,i])-17, '{}'.format(empirical_energies[0,i]), fontsize=12, color='w', fontproperties=myFont2)

    ax.text(int(empirical_energies[1,i]), int(empirical_energies[2,i]), '.', fontsize=30, color='white')

#for i in range(len(Chirality)):
#    ax.text(S11[i]-10, S22[i]-17, '{}'.format(str(Chirality[i])), fontsize=12, color='w', fontproperties=myFont2)

#    ax.text(S11[i], S22[i], '.', fontsize=30, color='white')
   # except Chirality[i]=='(10,2)'



##for k, energy in enumerate(S33_S11_ex_unfilled):
  ##  ax.text(S33_S11_em_unfilled[k], energy, '.', fontsize=30, color='w')
    #ax.text(S33_S11_em_unfilled[k] - 0, energy[k] + 20, '{}'.format(S33_S11_chirality[k]), fontsize=0.5, color='white', fontproperties=myFont2)

##for h, energy in enumerate(S22_S11_ex_unfilled):
##    ax.text(S22_S11_em_unfilled[h], energy, '.', fontsize=30, color='w')


# compare the exp energies with the model considering many-body effect (nmEnergyCalc.py):
##for m in range(12, 23, 1):  #[14, 16, 19, 22, 24]:
  ##  a = mp(familyvalue=m)

##for i in range(len(Chirality)):
  ##  ax.text(S11[i]-0, S22[i]+20, '{}'.format(Chirality[i]), fontsize=0.5, color='black', fontproperties=myFont2)

plt.xlabel('Emission (nm)', fontproperties=myFont)
plt.ylabel('Excitation (nm)', fontproperties=myFont)

plt.xlim(Em_start, Em_stop)
plt.ylim(Ex_start, Ex_stop)



ax = fig.add_subplot(1,2,2)
# Fig2

levels = np.arange(0.5, 20, 0.8)
if s22_s33:
    im = ax.imshow(NormalizedPLCombined2, interpolation='bilinear', origin='lower',
                   cmap=cmap, extent=(Em1, Em2, Ex1, Ex2))    # EmEnergy1_eV, EmEnergy2_eV, ExEnergy1_eV, ExEnergy2_eV
    CS = ax.contour(NormalizedPLCombined2, levels, origin='lower', cmap='flag',
                    linewidths=0.02, extent=(Em1, Em2, Ex11, Ex2))
else:
    im = ax.imshow(filter_negative_signal(NormalizedPL2, correct_exvariation), interpolation='bilinear', origin='lower',
                   cmap=cmap, extent=(Em1, Em2, Ex11, Ex2))
    CS = ax.contour(NormalizedPL2, levels, origin='lower', cmap='flag',
                    linewidths=0.02, extent=(Em1, Em2, Ex11, Ex2))


ax.tick_params(direction='out', length=5, width=1.5, colors='black',
               grid_color='black', grid_alpha=1, labelsize=12)

for side in ['bottom', 'left', 'top', 'right']:
    ax.spines[side].set_linewidth(1.2)

ax.xaxis.set_major_locator(plt.MultipleLocator(100))
ax.yaxis.set_major_locator(plt.MultipleLocator(50))
ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
ax.yaxis.set_minor_locator(plt.MultipleLocator(25))
#ax.yaxis.set_minor_formatter(ScalarFormatter("%.3f"))
#plt.xticks(fontproperties=myFont2)
#plt.yticks(fontproperties=myFont2)

ax.set_aspect(2)

# compare the exp energies with the model considering many-body effect (nmEnergyCalc.py):

#for m in range(12, 23, 1):  #[14, 16, 19, 22, 24]:
 #   a = mp(familyvalue=m)

for i in range(empirical_energies.shape[1]):
    ax.text(int(empirical_energies[1,i])-10, int(empirical_energies[2,i])-17, '{}'.format(empirical_energies[0,i]), fontsize=12, color='w', fontproperties=myFont2)

    ax.text(int(empirical_energies[1,i]), int(empirical_energies[2,i]), '.', fontsize=30, color='white')


#for i in range(len(Chirality)):
#    ax.text(S11[i]-10, S22[i]-17, '{}'.format(str(Chirality[i])), fontsize=12, color='w', fontproperties=myFont2)

#    ax.text(S11[i], S22[i], '.', fontsize=30, color='white')

#for i in range(len(Chirality)):
    #plt.annotate('{}'.format(Chirality[i]), (S11[i], S22[i]), color='white',
                 #arrowprops=dict(color='white', arrowstyle="->",
                                 #connectionstyle="angle,angleA=0,angleB=-120,rad=25", alpha=0.7))
#for i in range(len(Chirality)):
    #plt.annotate('{}'.format(Chirality[i]), (SubBandEnergies(S22[i], S11[i])), color='white', arrowprops=dict(color='white', arrowstyle="->",
                                                                                       #connectionstyle="angle,angleA=0,angleB=-120,rad=25", alpha=0.7))

#ax.set_title('PLE Contour, Ex(500-800)/Em(850-1350)', color='r')

# plot the Eii energies of SWCNTs in a solution state
#S33_S11_ex_HgTefilled = [369.7, 337.6, 371.8, 374.2]    # 346.5,
#S33_S11_em_HgTefilled = [1127, 1037.3, 1053.5, 1203]    # 986.3,
#S22_S11_ex_HgTefilled = [649, 590, 731.2, 646.5, 569.4, 671.1, 736.2, 727.3]
#S22_S11_em_HgTefilled = [1131, 1123, 1124.6, 1036.4, 989.9, 959.6, 1063.6, 1202.3]


##for k, energy in enumerate(S33_S11_ex_HgTefilled):
  ##  ax.text(S33_S11_em_HgTefilled[k], energy, '.', fontsize=30, color='w')

##for h, energy in enumerate(S22_S11_ex_HgTefilled):
  ##  ax.text(S22_S11_em_HgTefilled[h], energy, '.', fontsize=30, color='w')

plt.xlabel('Emission (nm)', fontproperties=myFont)
#plt.ylabel('Excitation (nm)', fontproperties=myFont)

plt.xlim(Em_start, Em_stop)
plt.ylim(Ex_start,Ex_stop)

savefig(figurefile, dpi=150)

# Fig3
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,2,1)
levels = np.arange(0, 20, 0.5)
if s22_s33:
    im = ax.imshow(NormalizedPLCombined1, interpolation='bicubic', origin='lower',
                   cmap=cm.rainbow, norm = LogNorm(), extent=(Em1, Em2, Ex1, Ex2))
    CS = ax.contour(NormalizedPLCombined1, levels, origin='lower', cmap='flag',
                    linewidths=0.2, extent=(Em1, Em2, Ex1, Ex2))
else:
    im = ax.imshow(filter_negative_signal(NormalizedPL1, correct_exvariation), interpolation='bicubic', origin='lower',
                   cmap=cm.rainbow, norm = LogNorm(), extent=(Em1, Em2, Ex11, Ex2))
    CS = ax.contour(NormalizedPL1, levels, origin='lower', cmap='flag',
                    linewidths=0.2, extent=(Em1, Em2, Ex11, Ex2))

#CS = ax.contour(NormalizedPLCombined1, levels, origin='lower', cmap='flag',
#                linewidths=0.2, extent=(Em1, Em2, Ex1, Ex2))

# We can still add a colorbar for the image, too.
CBI = fig.colorbar(im, orientation='vertical', shrink=0.8)

ax.set_aspect(2)

ax.tick_params(direction='out', length=3, width=1.0, colors='black',
               grid_color='black', grid_alpha=1, labelsize=15)

for side in ['bottom', 'left', 'top', 'right']:
    ax.spines[side].set_linewidth(1.2)

ax.xaxis.set_major_locator(plt.MultipleLocator(100))
ax.yaxis.set_major_locator(plt.MultipleLocator(100))
ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
ax.yaxis.set_minor_locator(plt.MultipleLocator(50))
#ax.yaxis.set_minor_formatter(ScalarFormatter("%.3f"))
plt.xticks(fontproperties=myFont4)
plt.yticks(fontproperties=myFont4)
plt.rcParams["axes.linewidth"]  = 2.0

for i in range(empirical_energies.shape[1]):
    #ax.text(S11[i]-0, S22[i]+20, '{}'.format(Chirality[i]), fontsize=0.5, color='black', fontproperties=myFont2)
    ax.text(int(empirical_energies[1,i]), int(empirical_energies[2,i]), '.', fontsize=30, color='black')


#ax.set_title('', color='b')     #Level in Log scale, Ex(500-800)/Em(850-1350)

plt.xlabel('Emission (nm)', fontproperties=myFont3)
plt.ylabel('Excitation (nm)', fontproperties=myFont3)


# Fig4
ax = fig.add_subplot(1,2,2)

levels = np.arange(0, 12, 0.5)
if s22_s33:
    im = ax.imshow(NormalizedPLCombined2, interpolation='bilinear', origin='lower',
                   cmap=cm.rainbow, norm = LogNorm(), extent=(Em1, Em2, Ex1, Ex2))
    CS = ax.contour(NormalizedPLCombined2, levels, origin='lower', cmap='flag',
                    linewidths=0.2, extent=(Em1, Em2, Ex1, Ex2))
else:
    im = ax.imshow(filter_negative_signal(NormalizedPL2, correct_exvariation), interpolation='bilinear', origin='lower',
                   cmap=cm.rainbow, norm = LogNorm(), extent=(Em1, Em2, Ex11, Ex2))
    CS = ax.contour(NormalizedPL2, levels, origin='lower', cmap='flag',
                    linewidths=0.2, extent=(Em1, Em2, Ex11, Ex2))
#CS = ax.contour(NormalizedPLCombined2, levels, origin='lower', cmap='flag',
#                linewidths=0.2, extent=(Em1, Em2, Ex1, Ex2))

# We can still add a colorbar for the image, too.
CBI = fig.colorbar(im, orientation='vertical', shrink=0.8)

#ax.set_title('Level in Log scale, Ex(500-800)/Em(850-1350)', color='b')

ax.set_aspect(2)

ax.tick_params(direction='out', length=3, width=1.0, colors='black',
               grid_color='black', grid_alpha=1, labelsize=15)

for side in ['bottom', 'left', 'top', 'right']:
    ax.spines[side].set_linewidth(1.2)

ax.xaxis.set_major_locator(plt.MultipleLocator(100))
ax.yaxis.set_major_locator(plt.MultipleLocator(100))
ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
ax.yaxis.set_minor_locator(plt.MultipleLocator(50))
#ax.yaxis.set_minor_formatter(ScalarFormatter("%.3f"))
plt.xticks(fontproperties=myFont4)
plt.yticks(fontproperties=myFont4)
plt.rcParams["axes.linewidth"]  = 2.0


for i in range(empirical_energies.shape[1]):
    #ax.text(S11[i]-0, S22[i]+20, '{}'.format(Chirality[i]), fontsize=0.5, color='black', fontproperties=myFont2)
    ax.text(int(empirical_energies[1,i]), int(empirical_energies[2,i]), '.', fontsize=30, color='black')

plt.xlabel('Emission (nm)', fontproperties=myFont3)
plt.ylabel('Excitation (nm)', fontproperties=myFont3)

# Plot PLE profile
#fig.subplot_adjust()
fig2 = plt.figure()
#fig,ax2 = plt.subplots(figsize=(8,7))
ax2 = fig2.add_subplot(111)


#f1 = PlotPLE(round_up(1023), filter_negative_signal(NormalizedPL1, correct_exvariation), 'black', Ex_peak_estimate=round_up(646), unfilled=True)
f1_em = PlotPL(round_up(648), filter_negative_signal(NormalizedPL1, correct_exvariation), 'black', Em_peak_estimate=round_up(1123), unfilled=True)
#f2 = PlotPLE(round_up(1023), filter_negative_signal(NormalizedPL2, correct_exvariation), 'red', Ex_peak_estimate=round_up(648), unfilled=False)
f2_em = PlotPL(round_up(648), filter_negative_signal(NormalizedPL2, correct_exvariation), 'red', Em_peak_estimate=round_up(1123), unfilled=False)
##f3 = PlotPLE(1195, data1, 'purple', unfilled=False)
###ax2.plot(wavelength, Abs, '-', color='gray', linewidth=2,  label='Absorbance')
##np.savetxt('PLE unfilled.txt', f1, delimiter='   ')
##np.savetxt('PLE HgTe-filled.txt', f2, delimiter='   ')
#xlim(500, 800)
#ylim(0.35,0.9)
xlabel('Excitation wavelength (nm)', fontproperties=myFont3)
#ylabel('Absorbance or PL intensity (a.u.)', fontproperties=myFont3)
title('', color='r', fontsize=14)       #PLE profile
ax2.legend(loc=3)
plt.legend(prop={'family': 'Arial', 'size': 15})

ax2.tick_params(direction='out', length=7, width=1.5, colors='black',
               grid_color='black', grid_alpha=1, labelsize=15)
for side in ['bottom', 'left', 'top', 'right']:
    ax2.spines[side].set_linewidth(2)

ax2.xaxis.set_major_locator(plt.MultipleLocator(30))
ax2.yaxis.set_major_locator(plt.MultipleLocator(30))
ax2.xaxis.set_minor_locator(plt.MultipleLocator(15))
ax2.yaxis.set_minor_locator(plt.MultipleLocator(15))

#plt.xticks(fontproperties=myFont)
#plt.yticks(fontproperties=myFont)
#plt.rcParams["axes.linewidth"]  = 2.0

ax2.set_aspect(50)

#bg_profile(NormalizedPL2)


savefig(spectrafile, dpi=150)


plt.show()

## Calculating transition energies based on the many-body model:
#for m in range(17, 24, 1):  #[14, 16, 19, 22, 24]:
#    a = mp(familyvalue=m)

